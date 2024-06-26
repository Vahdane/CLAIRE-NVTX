/*************************************************************************
 *  Copyright (c) 2018.
 *  All rights reserved.
 *  This file is part of the CLAIRE library.
 *
 *  CLAIRE is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  CLAIRE is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with CLAIRE. If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#include "DistanceMeasureKernel.hpp"
#include "cuda_helper.hpp"
#include "thrust/device_ptr.h"
#include "thrust/reduce.h"
#include "thrust/execution_policy.h"
#include "cublas_v2.h"

#include "nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();


__global__ void VecSubMulGPU(ScalarType *pL, const ScalarType *pW, const ScalarType *pWts,
    const ScalarType *pMr, const ScalarType *pM, int nl) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int k = blockIdx.y*nl + i;
  
  if (i < nl) {
    pL[k] = pW[i] * pWts[blockIdx.y] * (pMr[k] - pM[k]);
  }
}

__global__ void VecSubGPU(ScalarType *pL, const ScalarType *pWts, const ScalarType *pMr, 
    const ScalarType *pM, int nl) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int k = blockIdx.y*nl + i;
  
  if (i < nl) {
    pL[k] = pWts[blockIdx.y] * (pMr[k] - pM[k]);
  }
}

__global__ void VecMulGPU(ScalarType *pL, const ScalarType *pW, const ScalarType *pWts,
    const ScalarType *pM, int nl) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int k = blockIdx.y*nl + i;
  
  if (i < nl) {
    pL[k] = - pW[i] * pWts[blockIdx.y] * pM[k];
  }
}

__global__ void VecNegGPU(ScalarType *pL, const ScalarType *pWts, const ScalarType *pM, int nl) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int k = blockIdx.y*nl + i;
  
  if (i < nl) {
    pL[k] = - pWts[blockIdx.y] * pM[k];
  }
}

// TODO I would rather use cuBlas functions for reduction for performace reasons
template<int N> inline __device__ void LocalReductionSum(ScalarType *shared) {
  if (threadIdx.x < N) {
    shared[threadIdx.x] += shared[threadIdx.x + N];
  }
  __syncthreads();
  LocalReductionSum<N/2>(shared);
}

template<> inline __device__ void LocalReductionSum<1>(ScalarType *shared) {
  if (threadIdx.x == 0) {
    shared[0] += shared[1];
  }
  __syncthreads();
}

template<int N> __global__ void ReductionSum(ScalarType *res, int n) {
  __shared__ ScalarType value[N];
  
  value[threadIdx.x] = 0.0;
  for (int i=threadIdx.x; i<n; i+=N) {
    value[threadIdx.x] += res[i];
  }
  
  __syncthreads();
  
  LocalReductionSum<N/2>(value);
  
  if (threadIdx.x == 0) {
    res[0] = value[0];
  }
}

template<int N>
__global__ void DistanceMeasureFunctionalGPU(ScalarType *res, 
    const ScalarType *pW, const ScalarType *pWts, 
    const ScalarType *pMr,const ScalarType *pM, 
    int nl) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int k = blockIdx.y*nl + i;
  
  __shared__ ScalarType value[N];
  
  ScalarType tmp;
  
  value[threadIdx.x] = 0.0;
  
  if (i < nl) {
    tmp = pMr[k] - pM[k];
    value[threadIdx.x] = tmp*tmp*pW[i]*pWts[blockIdx.y];
  }
  
  __syncthreads();
    
  LocalReductionSum<N/2>(value);
  
  if (threadIdx.x == 0) {
    res[blockIdx.x + blockIdx.y*gridDim.x] = value[0];
  }
}

template<int N>
__global__ void DistanceMeasureFunctionalGPU(ScalarType *res, 
    const ScalarType *pWts, const ScalarType *pMr, const ScalarType *pM, int nl) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int k = blockIdx.y*nl + i;
  
  __shared__ ScalarType value[N];
  ScalarType tmp;
  
  value[threadIdx.x] = 0.0;  
  if (i < nl) {
    tmp = (pMr[k] - pM[k]);
    value[threadIdx.x] = tmp*tmp*pWts[blockIdx.y];
  }
  
  __syncthreads();
    
  LocalReductionSum<N/2>(value);
  
  if (threadIdx.x == 0) {
    res[blockIdx.x + blockIdx.y*gridDim.x] = value[0];
  }
}

namespace reg {
namespace DistanceMeasureKernel {

/* Compute Masked Registration Functional */
PetscErrorCode EvaluateFunctionalSL2::ComputeFunctionalMask() {
  //PUSH_RANGE("CU_DistanceMeasure_sl2_1",3)
  PetscErrorCode ierr = 0;
  dim3 block(256, 1, 1);
  dim3 grid((nl + 255)/256, nc, 1);
  PetscFunctionBegin;
  //ierr = AllocateMemoryOnce(res, grid.x*grid.y*sizeof(ScalarType)); CHKERRQ(ierr);
  
  DistanceMeasureFunctionalGPU<256><<<grid, block>>>(res, pW, pWts, pMr, pM, nl);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
  
  ReductionSum<1024><<<1, 1024>>>(res, grid.x*grid.y);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
  
  ierr = cudaMemcpy(reinterpret_cast<void*>(&value), reinterpret_cast<void*>(res), sizeof(ScalarType), cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
    //POP_RANGE
  PetscFunctionReturn(ierr);
}

/* Compute the Registration Functional */
PetscErrorCode EvaluateFunctionalSL2::ComputeFunctional() {
 // PUSH_RANGE("CU_DistanceMeasure_sl2_2",3)
  PetscErrorCode ierr = 0;
  dim3 block(256, 1, 1);
  dim3 grid((nl + 255)/256, nc, 1);
  PetscFunctionBegin;
  //ierr = AllocateMemoryOnce(res, grid.x*grid.y*sizeof(ScalarType)); CHKERRQ(ierr);
  
  DistanceMeasureFunctionalGPU<256><<<grid, block>>>(res, pWts, pMr, pM, nl);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
  
  ReductionSum<1024><<<1, 1024>>>(res, grid.x*grid.y);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
  
  ierr = cudaMemcpy(reinterpret_cast<void*>(&value), reinterpret_cast<void*>(res), sizeof(ScalarType), cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
   //POP_RANGE 
  PetscFunctionReturn(ierr);
}

IntType GetTempResSize(IntType nl, IntType nc) {
  return (nl + 255)/256 * nc;
}

/* Final Condition for Adjoint Equation */
PetscErrorCode FinalConditionSL2::ComputeFinalConditionAE() {
 // PUSH_RANGE("CU_DistanceMeasure_sl2_3",3)
  PetscErrorCode ierr = 0;
  dim3 block(256, 1, 1);
  dim3 grid((nl + 255)/256, nc, 1);
  PetscFunctionBegin;
  VecSubGPU<<<grid, block>>>(pL, pWts, pMr, pM, nl);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
 // POP_RANGE
  PetscFunctionReturn(ierr);
}

/* Final Condition for Masked Adjoint Equation */
PetscErrorCode FinalConditionSL2::ComputeFinalConditionMaskAE() {
  //PUSH_RANGE("CU_DistanceMeasure_sl2_4",3)
  PetscErrorCode ierr = 0;
  dim3 block(256, 1, 1);
  dim3 grid((nl + 255)/256, nc, 1);
  PetscFunctionBegin;
  VecSubMulGPU<<<grid, block>>>(pL, pW, pWts, pMr, pM, nl);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

/* Final Condition for Incremental Adjoint Equation */
PetscErrorCode FinalConditionSL2::ComputeFinalConditionIAE() {
 // PUSH_RANGE("CU_DistanceMeasure_sl2_5",3)
  PetscErrorCode ierr = 0;
  dim3 block(256, 1, 1);
  dim3 grid((nl + 255)/256, nc, 1);
  PetscFunctionBegin;
  VecNegGPU<<<grid, block>>>(pL, pWts, pM, nl);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
  
  //Status_t stat;
   cublasHandle_t handle; 
   cublasCreate(&handle);
   cublasSnrm2(handle, nl*nc, pM, 1, &norm_mtilde_loc);
   cublasDestroy(handle);
  //POP_RANGE
  
  PetscFunctionReturn(ierr);
}

/* Final Condition for Masked Incremental Adjoint Equation */
PetscErrorCode FinalConditionSL2::ComputeFinalConditionMaskIAE() {
  //PUSH_RANGE("CU_DistanceMeasure_sl2_6",3)
  PetscErrorCode ierr = 0;
  dim3 block(256, 1, 1);
  dim3 grid((nl + 255)/256, nc, 1);
  PetscFunctionBegin;
  VecMulGPU<<<grid, block>>>(pL, pW, pWts, pM, nl);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
  //POP_RANGE
  PetscFunctionReturn(ierr);
}


////////////////////////////////////////////////////////////////////////
//> NCC Distance metric routines 
///////////////////////////////////////////////////////////////////////
__global__ void FinalConditionAENCC_kernel (ScalarType *pL, const ScalarType *pMr, const ScalarType *pM, ScalarType const1, ScalarType const2, ScalarType mean_m1, ScalarType mean_mR) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  pL[i] = const1*(pMr[i]-mean_mR) - const2*(pM[i]-mean_m1);
}

__global__ void FinalConditionIAENCC_kernel (ScalarType *pLtilde, const ScalarType *pMr, const ScalarType *pM, const ScalarType *pMtilde, ScalarType const1tilde, ScalarType const3tilde, ScalarType const5, ScalarType mean_m1, ScalarType mean_mR) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  pLtilde[i] = const1tilde*(pMr[i]-mean_mR) + const3tilde*(pM[i]-mean_m1) - const5*pMtilde[i];
}

/* Compute the Registration Functional */
PetscErrorCode EvaluateFunctionalNCC::ComputeScaleMask() {
  PetscErrorCode ierr = 0;
  //dim3 block(256, 1, 1);
  //dim3 grid((nl + 255)/256, nc, 1);
  //ScalarType *res = nullptr;
  PetscFunctionBegin;
  
  // not implemented
  
  PetscFunctionReturn(ierr);
}

/* Compute the Registration Functional */
PetscErrorCode EvaluateFunctionalNCC::ComputeScale() {
  PetscErrorCode ierr = 0;
  //Status_t stat;
  cublasHandle_t handle; 
  dim3 block(256, 1, 1);
  dim3 grid((nl + 255)/256, nc, 1);
  ScalarType *res = nullptr;
  //ScalarType sum = 0.0;
  PetscFunctionBegin;
  
  // compute local sums
  sum_mT_loc = thrust::reduce(thrust::device, pMt, pMt+nl*nc);
  sum_mR_loc = thrust::reduce(thrust::device, pMr, pMr+nl*nc);
  
   cublasCreate(&handle);

  cublasSnrm2(handle, nl*nc, pMt, 1, &norm_mT_loc);
  norm_mT_loc *= norm_mT_loc;
  cublasSnrm2(handle, nl*nc, pMr, 1, &norm_mR_loc);
  norm_mR_loc *= norm_mR_loc;
  cublasSdot(handle, nl*nc, pMt, 1, pMr, 1, &inpr_mT_mR_loc);

  cublasDestroy(handle);
  
  ierr = AllocateMemoryOnce(res, grid.x*grid.y*sizeof(ScalarType)); CHKERRQ(ierr);
  
  DistanceMeasureFunctionalGPU<256><<<grid, block>>>(res, pWts, pMr, pMt, nl);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
  
  ReductionSum<1024><<<1, 1024>>>(res, grid.x*grid.y);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
  
  ierr = cudaMemcpy(reinterpret_cast<void*>(&norm_l2_loc), reinterpret_cast<void*>(res), sizeof(ScalarType), cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
    
  FreeMemory(res);
  
  PetscFunctionReturn(ierr);
}

/* Compute the Registration Functional */
PetscErrorCode EvaluateFunctionalNCC::ComputeFunctional() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  // compute local sums
  sum_mR_loc = thrust::reduce(thrust::device, pMr, pMr+nl*nc);
  sum_m1_loc = thrust::reduce(thrust::device, pM, pM+nl*nc);

  //Status_t stat;
   cublasHandle_t handle; 

   cublasCreate(&handle);

   cublasSnrm2(handle, nl*nc, pM, 1, &norm_m1_loc);
  norm_m1_loc *= norm_m1_loc;
   cublasSnrm2(handle, nl*nc, pMr, 1, &norm_mR_loc);
  norm_mR_loc *= norm_mR_loc;
   cublasSdot(handle, nl*nc, pM, 1, pMr, 1, &inpr_m1_mR_loc);

   cublasDestroy(handle);
  
  PetscFunctionReturn(ierr);
}

/* Compute the Registration Functional */
PetscErrorCode EvaluateFunctionalNCC::ComputeFunctionalMask() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
    
  // not implemented 
  PetscFunctionReturn(ierr);
}

/* Final Condition for Adjoint Equation */
PetscErrorCode FinalConditionNCC::ComputeFinalConditionAE() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  dim3 block(256,1,1);
  dim3 grid((nl*nc + 255)/256,1,1);

  FinalConditionAENCC_kernel<<<grid, block>>>(pL, pMr, pM, const1, const2, mean_m1, mean_mR);
  cudaCheckKernelError();
  
  PetscFunctionReturn(ierr);
}

/* Final Condition for Adjoint Equation */
PetscErrorCode FinalConditionNCC::ComputeFinalConditionMaskAE() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  // not implemented 
  PetscFunctionReturn(ierr);
}

PetscErrorCode FinalConditionNCC::ComputeInnerProductsFinalConditionAE() {
  PetscErrorCode ierr = 0;
  //Status_t stat;
   cublasHandle_t handle; 
  PetscFunctionBegin;

  // compute local sums
  sum_mR_loc = thrust::reduce(thrust::device, pMr, pMr+nl*nc);
  sum_m1_loc = thrust::reduce(thrust::device, pM, pM+nl*nc);
  
   cublasCreate(&handle);
  
   cublasSdot(handle, nl*nc, pM, 1, pM, 1, &norm_m1_loc);
   cublasSdot(handle, nl*nc, pMr, 1, pMr, 1, &norm_mR_loc);
   cublasSdot(handle, nl*nc, pM, 1, pMr, 1, &inpr_m1_mR_loc);
  
   cublasDestroy(handle);

  PetscFunctionReturn(ierr);
}

PetscErrorCode FinalConditionNCC::ComputeInnerProductsFinalConditionIAE() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  sum_mR_loc = thrust::reduce(thrust::device, pMr, pMr+nl*nc);
  sum_m1_loc = thrust::reduce(thrust::device, pM, pM+nl*nc);
  sum_mtilde_loc = thrust::reduce(thrust::device, pMtilde, pMtilde+nl*nc);

  //Status_t stat;
   cublasHandle_t handle; 
   cublasCreate(&handle);

  ScalarType norm_mtilde_loc = 0;
   cublasSnrm2(handle, nl*nc, pMtilde, 1, &norm_mtilde_loc);
  norm_mtilde_loc *= norm_mtilde_loc;
  
   cublasSdot(handle, nl*nc, pMr, 1, pMtilde, 1, &inpr_mR_mtilde_loc);
  ierr = Assert(!PetscIsNanReal(inpr_mR_mtilde_loc), "is nan"); CHKERRQ(ierr);
   cublasSdot(handle, nl*nc, pM, 1, pMtilde, 1, &inpr_m1_mtilde_loc);
  ierr = Assert(!PetscIsNanReal(inpr_m1_mtilde_loc), "is nan"); CHKERRQ(ierr);
   cublasSnrm2(handle, nl*nc, pM, 1, &norm_m1_loc);
  norm_m1_loc *= norm_m1_loc;
  ierr = Assert(!PetscIsNanReal(norm_m1_loc), "is nan"); CHKERRQ(ierr);
   cublasSnrm2(handle, nl*nc, pMr, 1, &norm_mR_loc);
  norm_mR_loc *= norm_mR_loc;
  ierr = Assert(!PetscIsNanReal(norm_mR_loc), "is nan"); CHKERRQ(ierr);
   cublasSdot(handle, nl*nc, pM, 1, pMr, 1, &inpr_m1_mR_loc);
  ierr = Assert(!PetscIsNanReal(inpr_m1_mR_loc), "is nan"); CHKERRQ(ierr);
  
   cublasDestroy(handle);

  PetscFunctionReturn(ierr);
}

/* Final Condition for Incremental Adjoint Equation */
PetscErrorCode FinalConditionNCC::ComputeFinalConditionIAE() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  dim3 block(256,1,1);
  dim3 grid((nl*nc + 255)/256,1,1);
    
  ScalarType const1tilde = const1 - const2;
  ScalarType const3tilde = const3 - const4;
  FinalConditionIAENCC_kernel<<<grid, block>>>(pLtilde, pMr, pM, pMtilde, const1tilde, const3tilde, const5, mean_m1, mean_mR);
  cudaCheckKernelError();
  
  PetscFunctionReturn(ierr);
}

/* Final Condition for Incremental Adjoint Equation */
PetscErrorCode FinalConditionNCC::ComputeFinalConditionMaskIAE() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  // not implemented
  
  PetscFunctionReturn(ierr);
}

} // namespace DistanceMeasureKernel
} // namespace reg
