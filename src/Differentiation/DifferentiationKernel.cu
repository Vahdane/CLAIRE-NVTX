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

#ifndef _DIFFERENTIATIONKERNEL_CPP_
#define _DIFFERENTIATIONKERNEL_CPP_

#include "DifferentiationKernel.hpp"
#include "cuda_helper.hpp"

#include "DifferentiationKernel.txx"

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


namespace reg {

PetscErrorCode DifferentiationKernel::ScalarLaplacian(ScalarType b0) {
  //PUSH_RANGE("CU_Differentiation_ScalarLaplacian",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ierr = SpectralKernelCallGPU<NLaplacianKernel<1> >(nstart, nx, nl, 
    pXHat[0], b0*scale); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}


PetscErrorCode DifferentiationKernel::LaplacianMod(ScalarType b0) {
  //PUSH_RANGE("CU_Differentiation_LaplacianMod",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ierr = SpectralKernelCallGPU<NLaplacianModKernel<1> >(nstart, nx, nl, 
    pXHat[0], pXHat[1], pXHat[2], 
    scale, b0); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}
PetscErrorCode DifferentiationKernel::Laplacian(ScalarType b0, ScalarType b1) {
  //PUSH_RANGE("CU_Differentiation_Laplacian",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  if (b1 == 0.0) {
    ierr = SpectralKernelCallGPU<NLaplacianKernel<1> >(nstart, nx, nl, 
      pXHat[0], pXHat[1], pXHat[2], 
      b0*scale); CHKERRQ(ierr);
  } else {
    ierr = SpectralKernelCallGPU<RelaxedNLaplacianKernel<1> >(nstart, nx, nl, 
      pXHat[0], pXHat[1], pXHat[2], 
      b0*scale, b1); CHKERRQ(ierr);
  }
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::LaplacianTol(ScalarType b0, ScalarType b1) {
  //PUSH_RANGE("CU_Differentiation_LaplacianTol",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ScalarType lognx = 0.;
  lognx += log2(static_cast<ScalarType>(nx[0]));
  lognx += log2(static_cast<ScalarType>(nx[1]));
  lognx += log2(static_cast<ScalarType>(nx[2]));
  
  KernelUtils::array3_t<ComplexType*> v;
  v.x = pXHat[0];
  v.y = pXHat[1];
  v.z = pXHat[2];
  
  if (b1 == 0.0) {
    ierr = SpectralKernelCallGPU<NLaplacianFilterKernel<1> >(nstart, nx, nl, v, 
      b0*scale, tol*lognx); CHKERRQ(ierr);
  } else {
    ierr = SpectralKernelCallGPU<RelaxedNLaplacianKernel<1> >(nstart, nx, nl, 
      pXHat[0], pXHat[1], pXHat[2], 
      b0*scale, b1); CHKERRQ(ierr);
  }
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::Bilaplacian(ScalarType b0, ScalarType b1) {
  //PUSH_RANGE("CU_Differentiation_Bilaplacian",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  if (b1 == 0.0) {
    ierr = SpectralKernelCallGPU<NLaplacianKernel<2> >(nstart, nx, nl, 
      pXHat[0], pXHat[1], pXHat[2], 
      b0*scale); CHKERRQ(ierr);
  } else {
    ierr = SpectralKernelCallGPU<RelaxedNLaplacianKernel<2> >(nstart, nx, nl,
      pXHat[0], pXHat[1], pXHat[2], 
      b0*scale, b1); CHKERRQ(ierr);
  }
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::Trilaplacian(ScalarType b0, ScalarType b1) {
  //PUSH_RANGE("CU_Differentiation_Trilaplacian",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  if (b1 == 0.0) {
    ierr = SpectralKernelCallGPU<NLaplacianKernel<3> >(nstart, nx, nl, 
      pXHat[0], pXHat[1], pXHat[2], 
      b0*scale); CHKERRQ(ierr);
  } else {
    ierr = SpectralKernelCallGPU<RelaxedNLaplacianKernel<3> >(nstart, nx, nl,
      pXHat[0], pXHat[1], pXHat[2], 
      b0*scale, b1); CHKERRQ(ierr);
  }
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::TrilaplacianFunctional(ScalarType b0, ScalarType b1) {
  //PUSH_RANGE("CU_Differentiation_TrilaplacianFunctional",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ierr = ThrowError("trilaplacian operator not implemented"); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::InverseLaplacian(bool usesqrt, ScalarType b0, ScalarType b1) {
  //PUSH_RANGE("CU_Differentiation_InverseLaplacian",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  if (usesqrt) {
    if (b1 == 0.0) {
      ierr = SpectralKernelCallGPU<InverseNLaplacianSqrtKernel<1> >(nstart, nx, nl,
        pXHat[0], pXHat[1], pXHat[2],
        scale, b0); CHKERRQ(ierr);
    } else {
      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianSqrtKernel<1> >(nstart, nx, nl, 
        pXHat[0], pXHat[1], pXHat[2],
        scale, b0, b1); CHKERRQ(ierr);
    }
  } else {
    if (b1 == 0.0) {
      ierr = SpectralKernelCallGPU<InverseNLaplacianKernel<1> >(nstart, nx, nl, 
        pXHat[0], pXHat[1], pXHat[2], 
        scale, b0); CHKERRQ(ierr);
    } else {
      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianKernel<1> >(nstart, nx, nl, 
        pXHat[0], pXHat[1], pXHat[2],
        scale, b0, b1); CHKERRQ(ierr);
    }
  }
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::InverseBilaplacian(bool usesqrt, ScalarType b0, ScalarType b1) {
  //PUSH_RANGE("CU_Differentiation_InverseBilaplacian",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  if (usesqrt) {
    if (b1 == 0.0) {
      /// scale/sqrt(b0*|lapik|^2) = scale/(sqrt(b0)*|lapik|)
      ierr = SpectralKernelCallGPU<InverseNLaplacianKernel<1> >(nstart, nx, nl,
        pXHat[0], pXHat[1], pXHat[2],
        scale, sqrt(b0)); CHKERRQ(ierr);
    } else {
      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianSqrtKernel<2> >(nstart, nx, nl, 
        pXHat[0], pXHat[1], pXHat[2],
        scale, b0, b1); CHKERRQ(ierr);
    }
  } else {
    if (b1 == 0.0) {
      ierr = SpectralKernelCallGPU<InverseNLaplacianKernel<2> >(nstart, nx, nl, 
        pXHat[0], pXHat[1], pXHat[2], 
        scale, b0); CHKERRQ(ierr);
    } else {
      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianKernel<2> >(nstart, nx, nl, 
        pXHat[0], pXHat[1], pXHat[2],
        scale, b0, b1); CHKERRQ(ierr);
    }
  }
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::InverseTrilaplacian(bool usesqrt, ScalarType b0, ScalarType b1) {
  //PUSH_RANGE("CU_Differentiation_InverseTrilaplacian",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  if (usesqrt) {
    if (b1 == 0.0) {
      ierr = SpectralKernelCallGPU<InverseNLaplacianSqrtKernel<3> >(nstart, nx, nl,
        pXHat[0], pXHat[1], pXHat[2],
        scale, sqrt(b0)); CHKERRQ(ierr);
    } else {
      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianSqrtKernel<3> >(nstart, nx, nl, 
        pXHat[0], pXHat[1], pXHat[2],
        scale, b0, b1); CHKERRQ(ierr);
    }
  } else {
    if (b1 == 0.0) {
      ierr = SpectralKernelCallGPU<InverseNLaplacianKernel<3> >(nstart, nx, nl, 
        pXHat[0], pXHat[1], pXHat[2], 
        scale, b0); CHKERRQ(ierr);
    } else {
      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianKernel<3> >(nstart, nx, nl, 
        pXHat[0], pXHat[1], pXHat[2],
        scale, b0, b1); CHKERRQ(ierr);
    }
  }
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::Leray(ScalarType b0, ScalarType b1) {
  //PUSH_RANGE("CU_Differentiation_Leray",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ierr = SpectralKernelCallGPU<LerayKernel>(nstart, nx, nl, 
    pXHat[0], pXHat[1], pXHat[2], 
    scale, b0, b1); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::InvRegLeray(ScalarType b0, ScalarType b1, ScalarType b2) {
  //PUSH_RANGE("CU_Differentiation_InvRegLeray",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ierr = SpectralKernelCallGPU<LerayKernel>(nstart, nx, nl, 
    pXHat[0], pXHat[1], pXHat[2], 
    scale, b0, b1); CHKERRQ(ierr);
  ierr = SpectralKernelCallGPU<InverseNLaplacianKernel<1> >(nstart, nx, nl, 
    pXHat[0], pXHat[1], pXHat[2], 
    1., b2); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::GaussianFilter(const ScalarType c[3]) {
  //PUSH_RANGE("CU_Differentiation_GaussianFilter",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ierr = SpectralKernelCallGPU<GaussianFilterKernel>(nstart, nx, nl, 
    pXHat[0], c[0], c[1], c[2], scale); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

PetscErrorCode DifferentiationKernel::Gradient() {
  //PUSH_RANGE("CU_Differentiation_Gradient",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ierr = SpectralKernelCallGPU<GradientKernel>(nstart, nx, nl, 
    pXHat[0], pXHat[1], pXHat[2], scale); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}
PetscErrorCode DifferentiationKernel::Divergence() {
  //PUSH_RANGE("CU_Differentiation_Divergence",2)
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ierr = SpectralKernelCallGPU<DivergenceKernel>(nstart, nx, nl, 
    pXHat[0], pXHat[1], pXHat[2], scale); CHKERRQ(ierr);
  //POP_RANGE
  PetscFunctionReturn(ierr);
}

} // namespace reg

#endif
