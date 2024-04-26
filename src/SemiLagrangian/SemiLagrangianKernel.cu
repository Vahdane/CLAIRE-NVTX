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

#include "SemiLagrangianKernel.hpp"
#include "cuda_helper.hpp"
#include "SemiLagrangianKernel.txx"

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
  
using KernelUtils::SpacialKernelCallGPU;
  
PetscErrorCode TrajectoryKernel::RK2_Step1() {
  //PUSH_RANGE("CU_SemiLagrangian_RK2_Step1",6)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = SpacialKernelCallGPU<RK2Kernel>(istart, isize,
                                         pX[0], pX[1], pX[2],
                                         pV[0], pV[1], pV[2],
                                         ix[0], ix[1], ix[2],
                                         hx[0], hx[1], hx[2]); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode TrajectoryKernel::RK2_Step2() {
   //PUSH_RANGE("CU_SemiLagrangian_RK2_Step2",6)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  const ScalarType half = 0.5;

  ierr = SpacialKernelCallGPU<RK2Kernel>(istart, isize,
                                         pX[0], pX[1], pX[2],
                                         pV[0], pV[1], pV[2],
                                         pVx[0], pVx[1], pVx[2],
                                         ix[0], ix[1], ix[2],
                                         hx[0]*half, hx[1]*half, hx[2]*half); CHKERRQ(ierr);

  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode TrajectoryKernel::RK2_Step2_inplace() {
   //PUSH_RANGE("CU_SemiLagrangian_RK2_Step2_inplace",6)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  const ScalarType half = 0.5;

  ierr = SpacialKernelCallGPU<RK2Kernel>(istart, isize,
                                         pV[0], pV[1], pV[2],
                                         pVx[0], pVx[1], pVx[2],
                                         ix[0], ix[1], ix[2],
                                         hx[0]*half, hx[1]*half, hx[2]*half); CHKERRQ(ierr);

  //POP_RANGE
  PetscFunctionReturn(ierr);
}
} // namespace reg
