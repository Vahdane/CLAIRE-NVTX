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

#include "PreconditionerKernel.hpp"
#include "cuda_helper.hpp"

#include "PreconditionerKernel.txx"

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

namespace reg {
  
using KernelUtils::KernelCallGPU;
using KernelUtils::ReductionKernelCallGPU;

PetscErrorCode H0PrecondKernel::gMgMT2 () {
 
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
   //PUSH_RANGE("CU_PrecondKernel_gMgMT2",4)
  
  ierr = KernelCallGPU<H0Kernel2>(nl, 
                                 pM[0], pM[1], pM[2], 
                                 pVhat[0], pVhat[1], pVhat[2], 
                                 pGmt[0], pGmt[1], pGmt[2]); CHKERRQ(ierr);
  //POP_RANGE  
  PetscFunctionReturn(ierr);
}
PetscErrorCode H0PrecondKernel::res2 (ScalarType &res) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
    //PUSH_RANGE("CU_PrecondKernel_res2",4)
  
  ierr = ReductionKernelCallGPU<H0Kernel2>(res, pWS, nl, 
                                          pM[0], pM[1], pM[2],
                                          pP[0], pP[1], pP[2],
                                          pRes[0], pRes[1], pRes[2],
                                          pGmt[0], pGmt[1], pGmt[2],
                                          diag); CHKERRQ(ierr);
  //POP_RANGE  
  PetscFunctionReturn(ierr);
}
PetscErrorCode H0PrecondKernel::pTAp2 (ScalarType &res) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
    //PUSH_RANGE("CU_PrecondKernel_pTAp2",4)
  
  ierr = ReductionKernelCallGPU<H0Kernel2>(res, pWS, nl, 
                                          pM[0], pM[1], pM[2],
                                          pP[0], pP[1], pP[2],
                                          pGmt[0], pGmt[1], pGmt[2],
                                          diag); CHKERRQ(ierr);
  //POP_RANGE  
  PetscFunctionReturn(ierr);
}

PetscErrorCode H0PrecondKernel::gMgMT () {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
    //PUSH_RANGE("CU_PrecondKernel_gMgMT",4)
  
  ierr = KernelCallGPU<H0Kernel>(nl, 
                                 pM[0], pM[1], pM[2], 
                                 pVhat[0], pVhat[1], pVhat[2], 
                                 pGmt[0], pGmt[1], pGmt[2]); CHKERRQ(ierr);
  //POP_RANGE  
  PetscFunctionReturn(ierr);
}

PetscErrorCode H0PrecondKernel::res (ScalarType &res) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
    //PUSH_RANGE("CU_PrecondKernel_res",4)
  
  ierr = ReductionKernelCallGPU<H0Kernel>(res, pWS, nl, 
                                          pM[0], pM[1], pM[2],
                                          pP[0], pP[1], pP[2],
                                          pRes[0], pRes[1], pRes[2],
                                          pVhat[0], pVhat[1], pVhat[2]); CHKERRQ(ierr);
  // POP_RANGE 
  PetscFunctionReturn(ierr);
}

PetscErrorCode H0PrecondKernel::pTAp (ScalarType &res) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
    //PUSH_RANGE("CU_PrecondKernel_pTAp",4)
  
  ierr = ReductionKernelCallGPU<H0Kernel>(res, pWS, nl, 
                                          pM[0], pM[1], pM[2],
                                          pP[0], pP[1], pP[2]); CHKERRQ(ierr);
   // POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode H0PrecondKernel::CGres (ScalarType &res) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
   // PUSH_RANGE("CU_PrecondKernel_CGres",4)
  
  ScalarType alpha = res;
  
  ierr = ReductionKernelCallGPU<H0KernelCG>(res, pWS, nl, 
                                            pM[0], pM[1], pM[2],
                                            pP[0], pP[1], pP[2],
                                            pRes[0], pRes[1], pRes[2],
                                            pVhat[0], pVhat[1], pVhat[2],
                                            alpha); CHKERRQ(ierr);
   // POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode H0PrecondKernel::CGp (ScalarType alpha) {
 
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // PUSH_RANGE("CU_PrecondKernel_CGp",4)
    
  ierr = KernelCallGPU<H0KernelCG>(nl, 
                                   pP[0], pP[1], pP[2],
                                   pRes[0], pRes[1], pRes[2],
                                   alpha); CHKERRQ(ierr);
    //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode CFLStatKernel::CFLx (ScalarType &ratio) {
 
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
   //PUSH_RANGE("CU_CFLStatKernel_CFLx",4)
  
  ScalarType res;
    
  ierr = ReductionKernelCallGPU<CFLKernel>(res, nl, pV[0], h, dt); CHKERRQ(ierr);
  
  //ratio = res/ng;
  
  ratio = res;
    //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode H0PrecondKernel::Norm (ScalarType &norm) {
  
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
   //PUSH_RANGE("CU_PrecondKernel_Norm",4)
  
  ScalarType res;
    
  ierr = ReductionKernelCallGPU<NormKernel>(res, nl, pGmt[0]); CHKERRQ(ierr);
  
  //ratio = res/ng;
  
  norm = res;
   // POP_RANGE
  PetscFunctionReturn(ierr);
}

} // namespace reg
