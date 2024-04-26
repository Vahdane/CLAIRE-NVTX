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

#include "DeformationKernel.hpp"
#include "cuda_helper.hpp"
#include "DeformationKernel.txx"

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
  
PetscErrorCode DetDefGradKernel::IntegrateSL() {
  PUSH_RANGE("CU_DeformationFields_IntegrateSL",1)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = KernelUtils::KernelCallGPU<DetDefGradSLKernel>(nl, pJ, pJx, pDivV, pDivVx, alpha, ht); CHKERRQ(ierr);
  POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DetDefGradKernel::InitSL(ScalarType val) {
  PUSH_RANGE("CU_DeformationFields_InitSL",1)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = KernelUtils::KernelCallGPU<DetDefGradSLKernel>(nl, pJ, val); CHKERRQ(ierr);
  POP_RANGE
  PetscFunctionReturn(ierr);
}

} // namespace reg
