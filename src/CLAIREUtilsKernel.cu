/*************************************************************************
 *  Copyright (c) 2016.
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
 *  along with CLAIRE.  If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#ifndef _CLAIREUTILSKERNEL_CU_
#define _CLAIREUTILSKERNEL_CU_

#include "CLAIREUtils.hpp"
#include "cuda_helper.hpp"

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

// CUDA kernel to evaluate point-wise norm of a vector field
__global__ void VecFieldPointWiseNormKernel(ScalarType *p_m, const ScalarType *p_X1, const ScalarType *p_X2, const ScalarType *p_X3, IntType nl) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < nl) {
        p_m[i] = sqrtf(p_X1[i]*p_X1[i] + p_X2[i]*p_X2[i] + p_X3[i]*p_X3[i]);
    }
}

__global__ void CopyStridedToFlatVecKernel(ScalarType *pX, const ScalarType *p_x1, const ScalarType *p_x2, const ScalarType *p_x3, IntType nl) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < nl) {
        pX[3*i + 0] = p_x1[i];
        pX[3*i + 1] = p_x2[i];
        pX[3*i + 2] = p_x3[i];
    }

}

__global__ void CopyStridedFromFlatVecKernel(ScalarType *p_x1, ScalarType *p_x2, ScalarType *p_x3, const ScalarType* pX, IntType nl) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < nl) {
        p_x1[i] = pX[3*i + 0];
        p_x2[i] = pX[3*i + 1];
        p_x3[i] = pX[3*i + 2];
    }
}

__global__ void SetValueKernel(ScalarType* p, ScalarType v, IntType nl) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < nl) {
        p[i] = v;
    }
}


namespace reg {
  
  
PetscErrorCode SetValue(ScalarType* p, ScalarType v, IntType nl) {
  //PUSH_RANGE("CU_UtilsKernel_SetValue",3)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  dim3 block(256, 1, 1);
  dim3 grid((nl + 255)/256, 1, 1);
  
  SetValueKernel<<<grid, block>>>(p, v, nl);
  cudaDeviceSynchronize();
  cudaCheckKernelError();
  //POP_RANGE
  PetscFunctionReturn(ierr);
}
  

/********************************************************************
 * @brief compute pointwise norm of vector field
 *******************************************************************/
PetscErrorCode VecFieldPointWiseNormGPU(ScalarType* p_m, const ScalarType* p_X1, const ScalarType* p_X2, const ScalarType* p_X3, IntType nl) {
    //PUSH_RANGE("CU_UtilsKernel_VecFieldPointWiseNormGPU",3)
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    
    dim3 block(256, 1, 1);
    dim3 grid((nl + 255)/256, 1, 1);
    
    VecFieldPointWiseNormKernel<<<grid, block>>>(p_m, p_X1, p_X2, p_X3, nl);
    cudaDeviceSynchronize();
    cudaCheckKernelError();
    //POP_RANGE
    PetscFunctionReturn(ierr);

}


/********************************************************************
 * @brief Copy vector field to a flat array in strided fashion
 *******************************************************************/
PetscErrorCode CopyStridedToFlatVec(ScalarType* pX, const ScalarType* p_x1, const ScalarType* p_x2, const ScalarType* p_x3, IntType nl) {
    //PUSH_RANGE("CU_UtilsKernel_CopyStridedToFlatVec",3)
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    
    int threads = 256;
    int blocks = (nl + 255)/threads;

    CopyStridedToFlatVecKernel<<<blocks,threads>>>(pX, p_x1, p_x2, p_x3, nl);
    cudaDeviceSynchronize();
    cudaCheckKernelError();
    //POP_RANGE
    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief Copy vector field to a flat array in strided fashion
 *******************************************************************/
PetscErrorCode CopyStridedFromFlatVec(ScalarType* p_x1, ScalarType* p_x2, ScalarType* p_x3, const ScalarType* pX, IntType nl) {
    //PUSH_RANGE("CU_UtilsKernel_CopyStridedFromFlatVec",3)
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    
    int threads = 256;
    int blocks = (nl + 255)/threads;

    CopyStridedFromFlatVecKernel<<<blocks,threads>>>(p_x1, p_x2, p_x3, pX, nl);
    cudaDeviceSynchronize();
    cudaCheckKernelError();
    //POP_RANGE
    PetscFunctionReturn(ierr);
}

}

#endif
