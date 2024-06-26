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

#ifndef _SPECTRALKERNEL_CPP_
#define _SPECTRALKERNEL_CPP_

#include "SpectralKernel.hpp"
#include "cuda_helper.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include "SpectralKernel.txx"

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


using KernelUtils::SpectralKernelCallGPU;
using KernelUtils::SpectralReductionKernelCallGPU;

namespace reg {
  
PetscErrorCode SpectralKernel::LowPassFilter(ComplexType *pXHat, ScalarType pct) {
  //PUSH_RANGE("CU_SpectralKernel_LowPassFilter",1)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ScalarType l1, l2, l3;
  l1 = static_cast<ScalarType>(nx[0])*0.5*pct;
  l2 = static_cast<ScalarType>(nx[1])*0.5*pct;
  l3 = static_cast<ScalarType>(nx[2])*0.5*pct;
  ierr = SpectralKernelCallGPU<LowPassFilterKernel>(nstart, nx, nl, pXHat, l1, l2, l3, scale); CHKERRQ(ierr);
//POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode SpectralKernel::HighPassFilter(ComplexType *pXHat, ScalarType pct) {
 // PUSH_RANGE("CU_SpectralKernel_HighPassFilter",1)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ScalarType l1, l2, l3;
  l1 = static_cast<ScalarType>(nx[0])*0.5*pct;
  l2 = static_cast<ScalarType>(nx[1])*0.5*pct;
  l3 = static_cast<ScalarType>(nx[2])*0.5*pct;
  ierr = SpectralKernelCallGPU<HighPassFilterKernel>(nstart, nx, nl, pXHat, l1, l2, l3, scale); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode SpectralKernel::Scale(ComplexType *pX, ScalarType val) {
  //PUSH_RANGE("CU_SpectralKernel_Scale",1)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = SpectralKernelCallGPU<ScaleKernel>(nstart, nx, nl, pX, val); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

__global__ void FilterKernel(int3 wave, int3 nl, ComplexType *x, int3 nxc) {
  int i3 = threadIdx.x + blockIdx.x*blockDim.x;
  int i2 = blockIdx.y;
  int i1 = blockIdx.z;
  
  if (i3 < nl.z) {
    wave.x += i1;
    wave.y += i2;
    wave.z += i3;

    int i = i1*nl.y + i2*nl.z + i3;
    
    if (wave.x > nxc.x || wave.y > nxc.y || wave.z > nxc.z) {
      x[i][0] = 0.;
      x[i][1] = 0.;
    }
  }
}

PetscErrorCode SpectralKernel::Norm(ScalarType &norm, ComplexType *pXHat, const IntType w[3]) {
  //PUSH_RANGE("CU_SpectralKernel_Norm",1)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ScalarType l1 = w[0];
  ScalarType l2 = w[1];
  ScalarType l3 = w[2];
  
  ierr = SpectralReductionKernelCallGPU<NormKernel>(norm, pWS, nstart, nx, nl, pXHat, l1, l2, l3); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode SpectralKernel::Restrict(ComplexType *pXc, const ComplexType *pXf, 
                                        const IntType nx_c[3], const IntType osize_c[3], const IntType ostart_c[3]) {
  //PUSH_RANGE("CU_SpectralKernel_Restrict",1)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  cudaMemset(pXc, 0, sizeof(ComplexType)*osize_c[0]*osize_c[1]*osize_c[2]);
  
  size_t pitch_f = nl[2]*sizeof(ComplexType);
  size_t pitch_c = osize_c[2]*sizeof(ComplexType);
  size_t width = (osize_c[2]-1)*sizeof(ComplexType);
  size_t height_l = osize_c[1]/2;
  size_t height_h = osize_c[1]/2;
  
  if (osize_c[1]%2 == 0) height_h -= 1;
  
  IntType half_x = osize_c[0]/2;
  
  // width always fits in pencil or slab decomposition;
  
  for (IntType x=0; x<half_x; ++x) {
    size_t offset_c = osize_c[2]*osize_c[1]*x;
    size_t offset_f = nl[2]*nl[1]*x;
    cudaMemcpy2DAsync(&pXc[offset_c], pitch_c, const_cast<ComplexType*>(&pXf[offset_f]), pitch_f, width, height_l, cudaMemcpyDeviceToDevice);
    offset_c += osize_c[2]*(osize_c[1] - height_h);
    offset_f += nl[2]*(nl[1] - height_h);
    cudaMemcpy2DAsync(&pXc[offset_c], pitch_c, const_cast<ComplexType*>(&pXf[offset_f]), pitch_f, width, height_h, cudaMemcpyDeviceToDevice);
  }
  if (osize_c[0]%2 == 0) half_x -= 1;
  for (IntType x=1; x<=half_x; ++x) {
    size_t offset_c = osize_c[2]*osize_c[1]*(osize_c[0]-x);
    size_t offset_f = nl[2]*nl[1]*(nl[0]-x);
    cudaMemcpy2DAsync(&pXc[offset_c], pitch_c, const_cast<ComplexType*>(&pXf[offset_f]), pitch_f, width, height_l, cudaMemcpyDeviceToDevice);
    offset_c += osize_c[2]*(osize_c[1] - height_h);
    offset_f += nl[2]*(nl[1] - height_h);
    cudaMemcpy2DAsync(&pXc[offset_c], pitch_c, const_cast<ComplexType*>(&pXf[offset_f]), pitch_f, width, height_h, cudaMemcpyDeviceToDevice);
  }
  
  cudaDeviceSynchronize();
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode SpectralKernel::Prolong(ComplexType *pXf, const ComplexType *pXc, 
                                       const IntType nx_c[3], const IntType osize_c[3], const IntType ostart_c[3]) {
  //PUSH_RANGE("CU_SpectralKernel_Prolong",1)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  cudaMemset(pXf, 0, sizeof(ComplexType)*nl[0]*nl[1]*nl[2]);
  
  size_t pitch_f = nl[2]*sizeof(ComplexType);
  size_t pitch_c = osize_c[2]*sizeof(ComplexType);
  size_t width = (osize_c[2]-1)*sizeof(ComplexType);
  size_t height_l = osize_c[1]/2;
  size_t height_h = osize_c[1]/2;
  
  if (osize_c[1]%2 == 0) height_h -= 1;
  
  IntType half_x = osize_c[0]/2;
  
  for (IntType x=0; x<half_x; ++x) {
    size_t offset_c = osize_c[2]*osize_c[1]*x;
    size_t offset_f = nl[2]*nl[1]*x;
    cudaMemcpy2DAsync(&pXf[offset_f], pitch_f, const_cast<ComplexType*>(&pXc[offset_c]), pitch_c, width, height_l, cudaMemcpyDeviceToDevice);
    offset_c += osize_c[2]*(osize_c[1] - height_h);
    offset_f += nl[2]*(nl[1] - height_h);
    cudaMemcpy2DAsync(&pXf[offset_f], pitch_f, const_cast<ComplexType*>(&pXc[offset_c]), pitch_c, width, height_h, cudaMemcpyDeviceToDevice);
  }
  if (osize_c[0]%2 == 0) half_x -= 1;
  for (IntType x=1; x<=half_x; ++x) {
    size_t offset_c = osize_c[2]*osize_c[1]*(osize_c[0]-x);
    size_t offset_f = nl[2]*nl[1]*(nl[0]-x);
    cudaMemcpy2DAsync(&pXf[offset_f], pitch_f, const_cast<ComplexType*>(&pXc[offset_c]), pitch_c, width, height_l, cudaMemcpyDeviceToDevice);
    offset_c += osize_c[2]*(osize_c[1] - height_h);
    offset_f += nl[2]*(nl[1] - height_h);
    cudaMemcpy2DAsync(&pXf[offset_f], pitch_f, const_cast<ComplexType*>(&pXc[offset_c]), pitch_c, width, height_h, cudaMemcpyDeviceToDevice);
  }
  
  cudaDeviceSynchronize();
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

} // namepsace reg

#endif
