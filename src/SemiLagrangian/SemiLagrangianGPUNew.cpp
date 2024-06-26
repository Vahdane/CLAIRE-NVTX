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
 *  along with CLAIRE. If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#ifndef _SEMILAGRANGIANGPUNEW_CPP_
#define _SEMILAGRANGIANGPUNEW_CPP_

#include "SemiLagrangianGPUNew.hpp"
#include "SemiLagrangianKernel.hpp"
#include <petsc/private/vecimpl.h>

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

/********************************************************************
 * @brief default constructor
 *******************************************************************/
SemiLagrangianGPUNew::SemiLagrangianGPUNew() {
    this->Initialize();
}




/********************************************************************
 * @brief default constructor
 *******************************************************************/
SemiLagrangianGPUNew::SemiLagrangianGPUNew(RegOpt* opt) {
    this->m_Opt = opt;
    this->Initialize();
    
    if (opt->m_Verbosity > 2) {
      DbgMsg("SemiLagrangianGPUNew created");
    }
}




/********************************************************************
 * @brief default destructor
 *******************************************************************/
SemiLagrangianGPUNew::~SemiLagrangianGPUNew() {
    this->ClearMemory();
}




/********************************************************************
 * @brief init class variables
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::Initialize() { 
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    
    this->m_Xstate = nullptr;
    this->m_Xadjoint = nullptr;
    this->m_X = nullptr;
    this->m_WorkScaField1 = nullptr;
    this->m_WorkScaField2 = nullptr;
    this->m_ScaFieldGhost = nullptr;
    this->m_VecFieldGhost = nullptr;
    this->m_WorkVecField1 = nullptr;
    this->m_InitialTrajectory = nullptr;
    this->m_texture = 0;
    
#ifdef REG_HAS_MPICUDA
    this->cuda_aware = true;
#else
    this->cuda_aware = false;
#endif

    this->m_Dofs[0] = 1;
    this->m_Dofs[1] = 3;
    
    this->m_tmpInterpol1 = nullptr;
    this->m_tmpInterpol2 = nullptr;
    
    this->m_StatePlan = nullptr;
    this->m_AdjointPlan = nullptr;
    this->m_GhostPlan = nullptr;

    if (this->m_Opt->rank_cnt > 1) {
      IntType nl = this->m_Opt->m_Domain.nl;
      
      this->nghost = this->m_Opt->m_PDESolver.iporder;
      ierr = AllocateOnce(this->m_GhostPlan, this->m_Opt, this->nghost); CHKERRQ(ierr);
      this->g_alloc_max = this->m_GhostPlan->get_ghost_local_size_x(this->isize_g, this->istart_g);
      this->nlghost  = this->isize_g[0];
      this->nlghost *= this->isize_g[1];
      this->nlghost *= this->isize_g[2];
      
      cudaMalloc((void**)&this->m_VecFieldGhost, this->g_alloc_max); 

      ierr = AllocateOnce(this->m_StatePlan, this->g_alloc_max, this->cuda_aware);
      this->m_StatePlan->allocate(nl, this->m_Dofs, 2, this->nghost, this->isize_g);

    } else {
      ierr = AllocateOnce(this->m_Xstate, this->m_Opt); CHKERRQ(ierr);
      ierr = AllocateOnce(this->m_Xadjoint, this->m_Opt); CHKERRQ(ierr);
    }
    
    ierr = this->InitializeInterpolationTexture(); CHKERRQ(ierr);

    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief clears memory
 * perform everything on the GPU
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::ClearMemory() {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    if (this->m_Xstate != nullptr) {
        delete this->m_Xstate;
        this->m_Xstate = nullptr;
    }
    
    if (this->m_Xadjoint != nullptr) {
        delete this->m_Xadjoint;
        this->m_Xadjoint = nullptr;
    }

    if (this->m_InitialTrajectory != nullptr) {
        delete this->m_InitialTrajectory;
        this->m_InitialTrajectory = nullptr;
    }

    if (this->m_X != nullptr) {
        ierr = VecDestroy(&this->m_X); CHKERRQ(ierr);
        this->m_X = nullptr;
    }
    
    if (this->m_WorkScaField1 != nullptr) {
      cudaFree(this->m_WorkScaField1);
      this->m_WorkScaField1 = nullptr;
    }
    
    if (this->m_WorkScaField2 != nullptr) {
      cudaFree(this->m_WorkScaField2);
      this->m_WorkScaField2 = nullptr;
    }
    
    if (this->m_ScaFieldGhost != nullptr) {
      cudaFree(this->m_ScaFieldGhost);
      this->m_ScaFieldGhost = nullptr;
    }

    if (this->m_VecFieldGhost != nullptr) {
      cudaFree(this->m_VecFieldGhost); 
      this->m_VecFieldGhost = nullptr;
    }

    if (this->m_texture != 0) {
      cudaDestroyTextureObject(this->m_texture);
    }
  
    if (this->m_tmpInterpol1 != nullptr) {
      cudaFree(this->m_tmpInterpol1);
      this->m_tmpInterpol1 = nullptr;
    }
    
    if (this->m_tmpInterpol2 != nullptr) {
      cudaFree(this->m_tmpInterpol2);
    }

    Free(this->m_StatePlan);
    Free(this->m_AdjointPlan);
    Free(this->m_GhostPlan);
    
    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief init empty texture for interpolation on GPU 
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::InitializeInterpolationTexture() {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    if (this->m_Opt->rank_cnt == 1) {
      //int isize[3];
      //isize[0] = static_cast<int>(this->m_Opt->m_Domain.isize[0]);
      //isize[1] = static_cast<int>(this->m_Opt->m_Domain.isize[1]);
      //isize[2] = static_cast<int>(this->m_Opt->m_Domain.isize[2]);
      this->m_texture = gpuInitEmptyTexture(this->m_Opt->m_Domain.isize);
      if (this->m_Opt->m_PDESolver.iporder == 3) {
        cudaMalloc((void**) &this->m_tmpInterpol1, sizeof(float)*this->m_Opt->m_Domain.nl);
        cudaMalloc((void**) &this->m_tmpInterpol2, sizeof(float)*this->m_Opt->m_Domain.nl);
      }
    } else {
      this->m_texture = gpuInitEmptyTexture(this->isize_g);
    }
    
    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief set work vector field to not have to allocate it locally
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::SetWorkVecField(VecField* x) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    ierr = Assert(x != nullptr, "null pointer"); CHKERRQ(ierr);
    this->m_WorkVecField1 = x;

    PetscFunctionReturn(ierr);
}



/********************************************************************
 * @brief compute the trajectory from the velocity field based
 * on an rk2 scheme (todo: make the velocity field a const vector)
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::ComputeTrajectory(VecField* v, std::string flag) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);
    
    if (this->m_Opt->m_Verbosity > 2) {
      ScalarType tmp, max;
      VecNorm(v->m_X1, NORM_INFINITY, &max);
      VecNorm(v->m_X2, NORM_INFINITY, &tmp);
      max = std::max(max, tmp);
      VecNorm(v->m_X3, NORM_INFINITY, &tmp);
      max = std::max(max, tmp);
      
      tmp = std::min(this->m_Opt->m_Domain.hx[0], this->m_Opt->m_Domain.hx[1]);
      tmp = std::min(tmp, this->m_Opt->m_Domain.hx[2]);
      
      char str[200];
      sprintf(str, "CFL %s nt=%i vmax=%e", flag.c_str(), static_cast<int>(max/tmp), max);
      DbgMsg2(str);
    }

    // compute trajectory by calling a CUDA kernel
    if (this->m_Opt->m_PDESolver.rkorder == 2) {
        ierr = this->ComputeTrajectoryRK2(v, flag); CHKERRQ(ierr);
    } else if (this->m_Opt->m_PDESolver.rkorder == 4) {
        ierr = this->ComputeTrajectoryRK4(v, flag); CHKERRQ(ierr);
    } else {
        ierr = ThrowError("rk order not implemented"); CHKERRQ(ierr);
    }

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(ierr);
}


/********************************************************************
 * @brief compute the initial trajectory
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::ComputeInitialTrajectory() {
    PetscErrorCode ierr;
    ScalarType *p_x1=nullptr,*p_x2=nullptr,*p_x3=nullptr;
    IntType isize[3],istart[3];
    
    ScalarType hx[3];
    PetscFunctionBegin;


    for (unsigned int i = 0; i < 3; ++i){
      if (this->m_Opt->rank_cnt == 1) {
        hx[i]     = 1.;
      } else {
        hx[i]     = 1./static_cast<ScalarType>(this->m_Opt->m_Domain.nx[i]);
      }
        isize[i]  = this->m_Opt->m_Domain.isize[i];
        istart[i] = this->m_Opt->m_Domain.istart[i];
    }

    ierr = VecGetArray(this->m_InitialTrajectory->m_X1, &p_x1); CHKERRQ(ierr);
    ierr = VecGetArray(this->m_InitialTrajectory->m_X2, &p_x2); CHKERRQ(ierr);
    ierr = VecGetArray(this->m_InitialTrajectory->m_X3, &p_x3); CHKERRQ(ierr);

#pragma omp parallel
{
#pragma omp for
    for (int i1 = 0; i1 < isize[0]; ++i1){  // x1
        for (int i2 = 0; i2 < isize[1]; ++i2){ // x2
            for (int i3 = 0; i3 < isize[2]; ++i3){ // x3

                // compute coordinates (nodal grid)
                ScalarType x1 = hx[0]*static_cast<ScalarType>(i1 + istart[0]);
                ScalarType x2 = hx[1]*static_cast<ScalarType>(i2 + istart[1]);
                ScalarType x3 = hx[2]*static_cast<ScalarType>(i3 + istart[2]);

                // compute linear / flat index
                IntType linidx = GetLinearIndex(i1,i2,i3,isize);

                // assign values
                p_x1[linidx] = x1;
                p_x2[linidx] = x2;
                p_x3[linidx] = x3;

            } // i1
        } // i2
    } // i3
}// pragma omp for

    ierr=VecRestoreArray(this->m_InitialTrajectory->m_X1,&p_x1); CHKERRQ(ierr);
    ierr=VecRestoreArray(this->m_InitialTrajectory->m_X2,&p_x2); CHKERRQ(ierr);
    ierr=VecRestoreArray(this->m_InitialTrajectory->m_X3,&p_x3); CHKERRQ(ierr);
    
    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief set the initial trajectory from outside the class
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::SetInitialTrajectory(const ScalarType* pX) {
    PetscErrorCode ierr;
    PetscFunctionBegin;
    
    ierr = this->m_InitialTrajectory->SetComponents(pX, "stride"); CHKERRQ(ierr);
    if (this->m_Opt->rank_cnt == 1) {
      ierr = VecScale(this->m_InitialTrajectory->m_X1, 1./this->m_Opt->m_Domain.hx[0]); CHKERRQ(ierr);
      ierr = VecScale(this->m_InitialTrajectory->m_X2, 1./this->m_Opt->m_Domain.hx[1]); CHKERRQ(ierr);
      ierr = VecScale(this->m_InitialTrajectory->m_X3, 1./this->m_Opt->m_Domain.hx[2]); CHKERRQ(ierr);
    } else {
      ierr = VecScale(this->m_InitialTrajectory->m_X1, this->m_Opt->m_Domain.nx[0]); CHKERRQ(ierr);
      ierr = VecScale(this->m_InitialTrajectory->m_X2, this->m_Opt->m_Domain.nx[1]); CHKERRQ(ierr);
      ierr = VecScale(this->m_InitialTrajectory->m_X3, this->m_Opt->m_Domain.nx[2]); CHKERRQ(ierr);
    }
    
    if (m_Opt->m_Verbosity > 1) {
      DbgMsgCall("SetInitialTrajectory completed"); CHKERRQ(ierr);
    }
    
    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief compute the trajectory from the velocity field based
 * on an rk2 scheme (todo: make the velocity field a const vector)
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::ComputeTrajectoryRK2(VecField* v, std::string flag) {
    PetscErrorCode ierr = 0;
    std::stringstream ss;
    VecField *X = nullptr;
    
    TrajectoryKernel kernel;

    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    ierr = Assert(this->m_WorkVecField1 != nullptr, "null pointer"); CHKERRQ(ierr);
    
    kernel.isize[0] = this->m_Opt->m_Domain.isize[0];
    kernel.isize[1] = this->m_Opt->m_Domain.isize[1];
    kernel.isize[2] = this->m_Opt->m_Domain.isize[2];
    kernel.istart[0] = this->m_Opt->m_Domain.istart[0];
    kernel.istart[1] = this->m_Opt->m_Domain.istart[1];
    kernel.istart[2] = this->m_Opt->m_Domain.istart[2];
    
    if (this->m_Opt->rank_cnt == 1) { // Coordinates in [0, Ni)
      kernel.ix[0] = 1.; kernel.ix[1] = 1.; kernel.ix[2] = 1.;
      kernel.hx[0] = this->m_Opt->GetTimeStepSize()/this->m_Opt->m_Domain.hx[0];
      kernel.hx[1] = this->m_Opt->GetTimeStepSize()/this->m_Opt->m_Domain.hx[1];
      kernel.hx[2] = this->m_Opt->GetTimeStepSize()/this->m_Opt->m_Domain.hx[2];
    } else { // Coordinates in [0, 1)
      kernel.ix[0] = 1./static_cast<ScalarType>(this->m_Opt->m_Domain.nx[0]);
      kernel.ix[1] = 1./static_cast<ScalarType>(this->m_Opt->m_Domain.nx[1]);
      kernel.ix[2] = 1./static_cast<ScalarType>(this->m_Opt->m_Domain.nx[2]);
      kernel.hx[0] = this->m_Opt->GetTimeStepSize()/(2.*PETSC_PI);
      kernel.hx[1] = this->m_Opt->GetTimeStepSize()/(2.*PETSC_PI);
      kernel.hx[2] = this->m_Opt->GetTimeStepSize()/(2.*PETSC_PI);
    }
    
    if (flag.compare("state") == 0) {
        X = this->m_Xstate;
    } else if (flag.compare("adjoint") == 0) {
        X = this->m_Xadjoint;
        kernel.hx[0] *= -1.;
        kernel.hx[1] *= -1.;
        kernel.hx[2] *= -1.;
    } else {
        ierr = ThrowError("flag wrong"); CHKERRQ(ierr);
    }
    
    // RK2 stage 1
    if (this->m_Opt->rank_cnt == 1) {
      ierr = X->GetArraysWrite(kernel.pX); CHKERRQ(ierr);
    } else {
      ierr = this->m_WorkVecField1->GetArrays(kernel.pX); CHKERRQ(ierr);
    }
    ierr = v->GetArraysRead(kernel.pV); CHKERRQ(ierr);
    
    ierr = kernel.RK2_Step1(); CHKERRQ(ierr);
    
    if (this->m_Opt->rank_cnt == 1) {
      ierr = X->RestoreArrays(); CHKERRQ(ierr);
    } else {
      ierr = this->m_WorkVecField1->RestoreArrays(); CHKERRQ(ierr);
      ierr = this->MapCoordinateVector(flag);
    }
    ierr = v->RestoreArrays(); CHKERRQ(ierr);
        
    // Interpolate on Euler coordinates
    PUSH_RANGE("Interpolate",2)
    ierr = this->Interpolate(this->m_WorkVecField1, v, flag); CHKERRQ(ierr);
    POP_RANGE
    
    // RK2 stage 2 
    ierr = v->GetArraysRead(kernel.pV); CHKERRQ(ierr);
    ierr = this->m_WorkVecField1->GetArrays(kernel.pVx); CHKERRQ(ierr);

    if (this->m_Opt->rank_cnt == 1) {
      ierr = X->GetArraysWrite(kernel.pX); CHKERRQ(ierr);
      ierr = kernel.RK2_Step2(); CHKERRQ(ierr);
    } else {
      ierr = kernel.RK2_Step2_inplace(); CHKERRQ(ierr);
    }
    
    ierr = this->m_WorkVecField1->RestoreArrays(); CHKERRQ(ierr);
    ierr = v->RestoreArrays(); CHKERRQ(ierr);
    
    if (this->m_Opt->rank_cnt > 1) {
      ierr = this->MapCoordinateVector(flag);
    } else {
      ierr = X->RestoreArrays(); CHKERRQ(ierr);
    }
    
    if (this->m_Opt->m_Verbosity > 2) {
      DbgMsgCall("Trajectory computed");
    }

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(ierr);
}


PetscErrorCode SemiLagrangianGPUNew::ComputeTrajectoryRK4(VecField* v, std::string flag) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief interpolate scalar field
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::Interpolate(Vec* xo, Vec xi, std::string flag) {
    PetscErrorCode ierr = 0;
    ScalarType *p_xo = nullptr, *p_xi = nullptr;

    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    ierr = Assert(*xo != nullptr, "null pointer"); CHKERRQ(ierr);
    ierr = Assert( xi != nullptr, "null pointer"); CHKERRQ(ierr);
    
    ierr = GetRawPointerReadWrite( xi, &p_xi); CHKERRQ(ierr);
    ierr = GetRawPointerReadWrite(*xo, &p_xo); CHKERRQ(ierr);
    
    //PUSH_RANGE("SL_Interpolate",2)
    ierr = this->Interpolate(p_xo, p_xi, flag); CHKERRQ(ierr);
    //POP_RANGE

    ierr = RestoreRawPointerReadWrite(*xo, &p_xo); CHKERRQ(ierr);
    ierr = RestoreRawPointerReadWrite( xi, &p_xi); CHKERRQ(ierr);

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief interpolate scalar field
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::Interpolate(ScalarType* xo, ScalarType* xi, std::string flag) {
    PetscErrorCode ierr = 0;
    //int nx[3];
    IntType nl;
    std::stringstream ss;
    double timers[4] = {0, 0, 0, 0};
    const ScalarType *xq1, *xq2, *xq3;
    Interp3_Plan_GPU* interp_plan = nullptr;

    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    ierr = Assert(xi != nullptr, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(xo != nullptr, "null pointer"); CHKERRQ(ierr);
    
    nl     = this->m_Opt->m_Domain.nl;

    /*for (int i = 0; i < 3; ++i) {
        nx[i]     = static_cast<int>(this->m_Opt->m_Domain.nx[i]);
    }*/
    
    ZeitGeist_define(SL_INTERPOL);
    ZeitGeist_tick(SL_INTERPOL);
    ierr = this->m_Opt->StartTimer(IPSELFEXEC); CHKERRQ(ierr);

    if (this->m_Opt->rank_cnt > 1) {
      ZeitGeist_define(SL_COMM);
      ZeitGeist_tick(SL_COMM);
      this->m_GhostPlan->share_ghost_x(xi, this->m_VecFieldGhost);
      ZeitGeist_tock(SL_COMM);
      ScalarType *wout[1] = {xo};
      
      interp_plan = this->m_StatePlan;
      
      //PUSH_RANGE("SL_Interpolate",2)
      interp_plan->interpolate( this->m_VecFieldGhost, 
                                this->isize_g, 
                                this->nlghost,
                                nl, 
                                wout,
                                this->m_Opt->m_Domain.mpicomm, 
                                this->m_tmpInterpol1, 
                                this->m_tmpInterpol2, 
                                this->m_texture, 
                                this->m_Opt->m_PDESolver.iporder, 
                                &(this->m_Opt->m_GPUtime), 0, flag);
    //POP_RANGE
    } else {
      // compute interpolation for all components of the input scalar field
      if (flag.compare("state") == 0) {
          ierr = Assert(this->m_Xstate != nullptr, "null pointer"); CHKERRQ(ierr);
          ierr = this->m_Xstate->GetArraysRead(xq1, xq2, xq3);
          const ScalarType* xq[3] = {xq1, xq2, xq3};
          gpuInterp3D(xi, xq, xo, this->m_tmpInterpol1, this->m_tmpInterpol2, this->m_Opt->m_Domain.nx, static_cast<long int>(nl), this->m_texture, this->m_Opt->m_PDESolver.iporder, &(this->m_Opt->m_GPUtime));
          ierr = this->m_Xstate->RestoreArrays(); CHKERRQ(ierr);
      } else if (flag.compare("adjoint") == 0) {
          ierr = Assert(this->m_Xadjoint != nullptr, "null pointer"); CHKERRQ(ierr);
          ierr = this->m_Xadjoint->GetArraysRead(xq1, xq2, xq3);
          const ScalarType* xq[3] = {xq1, xq2, xq3};
          gpuInterp3D(xi, xq, xo, this->m_tmpInterpol1, this->m_tmpInterpol2, this->m_Opt->m_Domain.nx, static_cast<long int>(nl), this->m_texture, this->m_Opt->m_PDESolver.iporder, &(this->m_Opt->m_GPUtime));
          ierr = this->m_Xadjoint->RestoreArrays(); CHKERRQ(ierr);
      } else {
          ierr = ThrowError("flag wrong"); CHKERRQ(ierr);
      }
    }

    ierr = this->m_Opt->StopTimer(IPSELFEXEC); CHKERRQ(ierr);
    ZeitGeist_tock(SL_INTERPOL);
    
    this->m_Opt->IncreaseInterpTimers(timers);
    this->m_Opt->IncrementCounter(IP);

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief interpolate vector field
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::Interpolate(VecField* vo, VecField* vi, std::string flag) {
    PetscErrorCode ierr = 0;
    ScalarType *p_vix1 = nullptr, *p_vix2 = nullptr, *p_vix3 = nullptr;
    ScalarType *p_vox1 = nullptr, *p_vox2 = nullptr, *p_vox3 = nullptr;
    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    ierr = Assert(vi != nullptr, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(vo != nullptr, "null pointer"); CHKERRQ(ierr);
    
    ierr = vi->GetArraysReadWrite(p_vix1, p_vix2, p_vix3); CHKERRQ(ierr);
    ierr = vo->GetArraysReadWrite(p_vox1, p_vox2, p_vox3); CHKERRQ(ierr);

    //PUSH_RANGE("SL_Interpolate",2)
    ierr = this->Interpolate(p_vox1, p_vox2, p_vox3, p_vix1, p_vix2, p_vix3, flag); CHKERRQ(ierr);
    //POP_RANGE

    ierr = vi->RestoreArrays(); CHKERRQ(ierr);
    ierr = vo->RestoreArrays(); CHKERRQ(ierr);
    
    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief interpolate vector field - single GPU optimised version
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::Interpolate(ScalarType* wx1, ScalarType* wx2, ScalarType* wx3,
                                                 ScalarType* vx1, ScalarType* vx2, ScalarType* vx3, std::string flag) {
    PetscErrorCode ierr = 0;
    //int nx[3];
    double timers[4] = {0, 0, 0, 0};
    std::stringstream ss;
    IntType nl;
    Interp3_Plan_GPU* interp_plan = nullptr;
    const ScalarType *xq1, *xq2, *xq3;

    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    ierr = Assert(vx1 != nullptr, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(vx2 != nullptr, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(vx3 != nullptr, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(wx1 != nullptr, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(wx2 != nullptr, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(wx3 != nullptr, "null pointer"); CHKERRQ(ierr);

    nl = this->m_Opt->m_Domain.nl;

    //for (int i = 0; i < 3; ++i) {
    //    nx[i] = static_cast<int>(this->m_Opt->m_Domain.nx[i]);
    //}
    
    ZeitGeist_define(SL_INTERPOL);
    ZeitGeist_tick(SL_INTERPOL);
    ierr = this->m_Opt->StartTimer(IPSELFEXEC); CHKERRQ(ierr);

    interp_plan = this->m_StatePlan;
    
    ScalarType* vin[3] = {vx1, vx2, vx3};
    ScalarType* wout[3] = {wx1, wx2, wx3};

    if (this->m_Opt->rank_cnt > 1) {
      ZeitGeist_define(SL_COMM);
      for (int i=0; i<3; i++) { 
        ZeitGeist_tick(SL_COMM);
        this->m_GhostPlan->share_ghost_x(vin[i], &this->m_VecFieldGhost[0*this->nlghost]);
        ZeitGeist_tock(SL_COMM);

        
        // do interpolation
        //PUSH_RANGE("SL_Interpolate",2)
        interp_plan->interpolate( this->m_VecFieldGhost, 
                                  this->isize_g, 
                                  this->nlghost,
                                  nl, 
                                  &wout[i],
                                  this->m_Opt->m_Domain.mpicomm, 
                                  this->m_tmpInterpol1, 
                                  this->m_tmpInterpol2, 
                                  this->m_texture, 
                                  this->m_Opt->m_PDESolver.iporder, 
                                  &(this->m_Opt->m_GPUtime), 0, flag);
      //POP_RANGE

      }
    

    } else {
      

      if (flag.compare("state") == 0) {

          ierr = this->m_Xstate->GetArraysRead(xq1, xq2, xq3);
          const ScalarType* xq[3] = {xq1, xq2, xq3};
          gpuInterpVec3D(vx1, vx2, vx3, xq, wx1, wx2, wx3, this->m_tmpInterpol1, this->m_tmpInterpol2, this->m_Opt->m_Domain.nx, static_cast<long int>(nl), this->m_texture, this->m_Opt->m_PDESolver.iporder, &(this->m_Opt->m_GPUtime));
          ierr = this->m_Xstate->RestoreArrays(); CHKERRQ(ierr);

      } else if (flag.compare("adjoint") == 0) {
          
          ierr = this->m_Xadjoint->GetArraysRead(xq1, xq2, xq3);
          const ScalarType* xq[3] = {xq1, xq2, xq3};
          gpuInterpVec3D(vx1, vx2, vx3, xq, wx1, wx2, wx3, this->m_tmpInterpol1, this->m_tmpInterpol2, this->m_Opt->m_Domain.nx, static_cast<long int>(nl), this->m_texture, this->m_Opt->m_PDESolver.iporder, &(this->m_Opt->m_GPUtime));
          ierr = this->m_Xadjoint->RestoreArrays(); CHKERRQ(ierr);
         
      } else {
          ierr = ThrowError("flag wrong"); CHKERRQ(ierr);
      }
    }
    
    ierr = this->m_Opt->StopTimer(IPSELFEXEC); CHKERRQ(ierr);
    ZeitGeist_tock(SL_INTERPOL);
    ZeitGeist_inc(SL_INTERPOL);
    ZeitGeist_inc(SL_INTERPOL);

    this->m_Opt->IncreaseInterpTimers(timers);
    this->m_Opt->IncrementCounter(IPVEC);

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(ierr);
}


/********************************************************************
 * @brief set coordinate vector and communicate to interpolation plan
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::SetQueryPoints(ScalarType* y1, ScalarType* y2, ScalarType* y3, std::string flag) {
    PetscErrorCode ierr = 0;
    VecField* X = nullptr;
    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);
    
    // Hack for bug related to deformation map computation
    ierr = AllocateOnce(this->m_Xstate, this->m_Opt); CHKERRQ(ierr);

    if (flag.compare("state") == 0) {
        X = this->m_Xstate;
    } else if (flag.compare("adjoint") == 0) {
        X = this->m_Xadjoint;
    } else {
        ierr = ThrowError("flag wrong"); CHKERRQ(ierr);
    }
  
    ierr = Assert(X != nullptr, "nullptr"); CHKERRQ(ierr);
    
    ierr = X->SetComponents(y1, y2, y3); 
    
    ScalarType invhx[3];
    for (int i=0; i<3; i++) invhx[i] = 1./this->m_Opt->m_Domain.hx[i];
    ierr = X->Scale(invhx);
    
    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(ierr);
}

PetscErrorCode SemiLagrangianGPUNew::GetQueryPoints(ScalarType* y1, ScalarType* y2, ScalarType* y3) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    ierr = Assert(this->m_Xstate != nullptr, "null pointer"); CHKERRQ(ierr);
    
#if defined(REG_HAS_CUDA) && !defined(REG_HAS_MPICUDA) 
    ScalarType scale[3];
    scale[0] = this->m_Opt->m_Domain.hx[0];
    scale[1] = this->m_Opt->m_Domain.hx[1];
    scale[2] = this->m_Opt->m_Domain.hx[2];
    this->m_Xstate->Scale(scale);
    ierr = this->m_Xstate->GetComponents(y1, y2, y3); CHKERRQ(ierr);
    scale[0] = 1./this->m_Opt->m_Domain.hx[0];
    scale[1] = 1./this->m_Opt->m_Domain.hx[1];
    scale[2] = 1./this->m_Opt->m_Domain.hx[2];
    this->m_Xstate->Scale(scale);
/*#pragma omp parallel for
    for (IntType i = 0; i < this->m_Opt->m_Domain.nl; ++i) {
      y1[i] *= this->m_Opt->m_Domain.hx[0];
      y2[i] *= this->m_Opt->m_Domain.hx[1];
      y3[i] *= this->m_Opt->m_Domain.hx[2];
    }
    */
#else
    ierr = this->m_Xstate->GetComponents(y1, y2, y3); CHKERRQ(ierr);
#endif
    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(ierr);
}


/********************************************************************
 * Name: GetQueryPoints
 * Description: get the query points in the pointer y
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::GetQueryPoints(ScalarType* y) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    ierr = Assert(this->m_Xstate != nullptr, "null pointer"); CHKERRQ(ierr);

#if defined(REG_HAS_CUDA) && !defined(REG_HAS_MPICUDA)
    ScalarType scale[3];
    scale[0] = this->m_Opt->m_Domain.hx[0];
    scale[1] = this->m_Opt->m_Domain.hx[1];
    scale[2] = this->m_Opt->m_Domain.hx[2];
    this->m_Xstate->Scale(scale);
    ierr = this->m_Xstate->GetComponents(y); CHKERRQ(ierr);
    scale[0] = 1./this->m_Opt->m_Domain.hx[0];
    scale[1] = 1./this->m_Opt->m_Domain.hx[1];
    scale[2] = 1./this->m_Opt->m_Domain.hx[2];
    this->m_Xstate->Scale(scale); 
/*#pragma omp parallel for
    for (IntType i = 0; i < this->m_Opt->m_Domain.nl; ++i) {
      y[3*i+0] *= this->m_Opt->m_Domain.hx[0];
      y[3*i+1] *= this->m_Opt->m_Domain.hx[1];
      y[3*i+2] *= this->m_Opt->m_Domain.hx[2];
    }
    */
#else
    ierr = this->m_Xstate->GetComponents(y, "stride"); CHKERRQ(ierr);
#endif
    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(ierr);
}


PetscErrorCode SemiLagrangianGPUNew::CommunicateCoord(std::string flag) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  PetscFunctionReturn(ierr);
}

/********************************************************************
 * Name: map coordinates
 * Description: change from lexicographical ordering to xyz
 *******************************************************************/
PetscErrorCode SemiLagrangianGPUNew::MapCoordinateVector(std::string flag) {
    PetscErrorCode ierr;
    //int nx[3], c_dims[2], isize[3], istart[3];
    int c_dims[2];
    IntType nl;
    double timers[4] = {0,0,0,0};
    ScalarType* p_X[3] = {nullptr, nullptr, nullptr};

    PetscFunctionBegin;
    
    if (this->m_Opt->m_Verbosity > 2) {
      ierr = DbgMsgCall("Mapping query points"); CHKERRQ(ierr);
    }

    /*for (int i = 0; i < 3; ++i){
        nx[i] = this->m_Opt->m_Domain.nx[i];
        isize[i] = this->m_Opt->m_Domain.isize[i];
        istart[i] = this->m_Opt->m_Domain.istart[i];
    }*/
    
    c_dims[0] = this->m_Opt->m_CartGridDims[0];
    c_dims[1] = this->m_Opt->m_CartGridDims[1];
  
    nl = this->m_Opt->m_Domain.nl;
    
    ZeitGeist_define(SL_COMM);
    ZeitGeist_tick(SL_COMM);
    
    ierr = this->m_WorkVecField1->GetArrays(p_X); CHKERRQ(ierr);
    
    this->m_StatePlan->scatter(this->m_Opt->m_Domain.nx, this->m_Opt->m_Domain.isize, this->m_Opt->m_Domain.istart, nl, this->nghost, p_X[0], p_X[1], p_X[2], c_dims, this->m_Opt->m_Domain.mpicomm, timers, flag);

    ierr = this->m_WorkVecField1->RestoreArrays(); CHKERRQ(ierr);

    ZeitGeist_tock(SL_COMM);

    this->m_Opt->IncreaseInterpTimers(timers);

    PetscFunctionReturn(ierr);
}

}  // namespace reg




#endif  // _SEMILAGRANGIAN_CPP_
