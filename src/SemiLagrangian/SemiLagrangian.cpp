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

#ifndef _SEMILAGRANGIAN_CPP_
#define _SEMILAGRANGIAN_CPP_

#include "SemiLagrangian.hpp"

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
SemiLagrangian::SemiLagrangian() {
    this->Initialize();
}




/********************************************************************
 * @brief default constructor
 *******************************************************************/
SemiLagrangian::SemiLagrangian(RegOpt* opt) {
    this->Initialize();
    this->m_Opt = opt;
}




/********************************************************************
 * @brief default destructor
 *******************************************************************/
SemiLagrangian::~SemiLagrangian() {
    this->ClearMemory();
}




/********************************************************************
 * @brief init class variables
 *******************************************************************/
PetscErrorCode SemiLagrangian::Initialize() {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    this->m_X = NULL;
    this->m_WorkVecField1 = NULL;
    this->m_WorkVecField2 = NULL;

    this->m_StatePlan = NULL;
    this->m_AdjointPlan = NULL;

    this->m_ScaFieldGhost = NULL;
    this->m_VecFieldGhost = NULL;

    this->m_Opt = NULL;
    this->m_Dofs[0] = 1;
    this->m_Dofs[1] = 3;

    PetscFunctionReturn(ierr);
}




/********************************************************************
 * @brief clears memory
 *******************************************************************/
PetscErrorCode SemiLagrangian::ClearMemory() {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    if (this->m_X != NULL) {
        accfft_free(this->m_X);
        this->m_X = NULL;
    }

    if (this->m_AdjointPlan != NULL) {
        delete this->m_AdjointPlan;
        this->m_AdjointPlan = NULL;
    }
    if (this->m_StatePlan != NULL) {
        delete this->m_StatePlan;
        this->m_StatePlan = NULL;
    }

    if (this->m_ScaFieldGhost != NULL) {
        accfft_free(this->m_ScaFieldGhost);
        this->m_ScaFieldGhost = NULL;
    }

    if (this->m_VecFieldGhost != NULL) {
        accfft_free(this->m_VecFieldGhost);
        this->m_VecFieldGhost = NULL;
    }

    if (this->m_WorkVecField2 != NULL) {
        delete this->m_WorkVecField2;
        this->m_WorkVecField2 = NULL;
    }

    PetscFunctionReturn(ierr);
}




/********************************************************************
 * @brief set work vector field to not have to allocate it locally
 *******************************************************************/
PetscErrorCode SemiLagrangian::SetWorkVecField(VecField* x) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian",3)
    ierr = Assert(x != NULL, "null pointer"); CHKERRQ(ierr);
    this->m_WorkVecField1 = x;
    //POP_RANGE
    PetscFunctionReturn(ierr);
}




/********************************************************************
 * @brief compute the trajectory from the velocity field based
 * on an rk2 scheme (todo: make the velocity field a const vector)
 *******************************************************************/
PetscErrorCode SemiLagrangian::ComputeTrajectory(VecField* v, std::string flag, ScalarType *mX) {
    PetscErrorCode ierr = 0;
    IntType nl;
    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian",3)
    this->m_Opt->Enter(__func__);
    

    nl = this->m_Opt->m_Domain.nl;

    // if trajectory has not yet been allocated, allocate
    if (this->m_X == NULL) {
        try {this->m_X = reinterpret_cast<ScalarType*>(accfft_alloc(3*nl*sizeof(ScalarType)));}
        catch (std::bad_alloc& err) {
            ierr = reg::ThrowError(err); CHKERRQ(ierr);
        }
    }

    // compute trajectory

    if (this->m_Opt->m_PDESolver.rkorder == 2) {
        ierr = this->ComputeTrajectoryRK2(v, flag, mX); CHKERRQ(ierr);
    } else if (this->m_Opt->m_PDESolver.rkorder == 4) {
        ierr = this->ComputeTrajectoryRK4(v, flag, mX); CHKERRQ(ierr);
    } else {
        ierr = ThrowError("rk order not implemented"); CHKERRQ(ierr);
    }

    this->m_Opt->Exit(__func__);
    //POP_RANGE
    PetscFunctionReturn(ierr);
}

PetscErrorCode SemiLagrangian::SetInitialTrajectory(const ScalarType* pX) {
    PetscErrorCode ierr;
    PetscFunctionBegin;
    
    /*ierr = AllocateOnce(this->m_InitialTrajectory, this->m_Opt); CHKERRQ(ierr);
    
    ierr = this->m_InitialTrajectory->SetComponents(pX, "stride"); CHKERRQ(ierr);
    ierr = VecScale(this->m_InitialTrajectory->m_X1, 1./this->m_Opt->m_Domain.hx[0]); CHKERRQ(ierr);
    ierr = VecScale(this->m_InitialTrajectory->m_X2, 1./this->m_Opt->m_Domain.hx[1]); CHKERRQ(ierr);
    ierr = VecScale(this->m_InitialTrajectory->m_X3, 1./this->m_Opt->m_Domain.hx[2]); CHKERRQ(ierr);*/

    PetscFunctionReturn(0);
}



/********************************************************************
 * @brief compute the trajectory from the velocity field based
 * on an rk2 scheme (todo: make the velocity field a const vector)
 *******************************************************************/
PetscErrorCode SemiLagrangian::ComputeTrajectoryRK2(VecField* v, std::string flag, ScalarType *mX) {
    PetscErrorCode ierr = 0;
    ScalarType ht, hthalf, hx[3], x1, x2, x3, scale = 0.0;
    const ScalarType *p_v1 = NULL, *p_v2 = NULL, *p_v3 = NULL;
    ScalarType *p_vX1 = NULL, *p_vX2 = NULL, *p_vX3 = NULL;
    IntType isize[3], istart[3], l, i1, i2, i3;
    std::stringstream ss;

    PetscFunctionBegin;

    //PUSH_RANGE("SemiLagrangian",3)
    this->m_Opt->Enter(__func__);

    ierr = Assert(this->m_WorkVecField1 != NULL, "null pointer"); CHKERRQ(ierr);

    ht = this->m_Opt->GetTimeStepSize();
    hthalf = 0.5*ht;
    
    if (this->m_Opt->m_Verbosity > 2) {
        std::string str = "update trajectory: ";
        str += flag;
        ierr = DbgMsg(str); CHKERRQ(ierr);
        ierr = v->DebugInfo("SL v", __LINE__, __FILE__); CHKERRQ(ierr);
    }

    // switch between state and adjoint variable
    if (strcmp(flag.c_str(), "state") == 0) {
        scale =  1.0;
    } else if (strcmp(flag.c_str(), "adjoint") == 0) {
        scale = -1.0;
    } else {
        ierr = ThrowError("flag wrong"); CHKERRQ(ierr);
    }


    for (int i = 0; i < 3; ++i) {
        hx[i]     = this->m_Opt->m_Domain.hx[i];
        isize[i]  = this->m_Opt->m_Domain.isize[i];
        istart[i] = this->m_Opt->m_Domain.istart[i];
    }


    // \tilde{X} = x - ht v
    //ierr = v->GetArraysRead(p_v1, p_v2, p_v3); CHKERRQ(ierr);
    ierr = VecGetArrayRead(v->m_X1, &p_v1); CHKERRQ(ierr);
    ierr = VecGetArrayRead(v->m_X2, &p_v2); CHKERRQ(ierr);
    ierr = VecGetArrayRead(v->m_X3, &p_v3); CHKERRQ(ierr);
    for (i1 = 0; i1 < isize[0]; ++i1) {   // x1
        for (i2 = 0; i2 < isize[1]; ++i2) {   // x2
            for (i3 = 0; i3 < isize[2]; ++i3) {   // x3
                // compute linear / flat index
                l = GetLinearIndex(i1, i2, i3, isize);
                
                // compute coordinates (nodal grid)
                if (mX) {
                  x1 = mX[l*3 + 0];
                  x2 = mX[l*3 + 1];
                  x3 = mX[l*3 + 2];
                } else {
                  x1 = hx[0]*static_cast<ScalarType>(i1 + istart[0]);
                  x2 = hx[1]*static_cast<ScalarType>(i2 + istart[1]);
                  x3 = hx[2]*static_cast<ScalarType>(i3 + istart[2]); 
                }
                
                this->m_X[l*3+0] = (x1 - scale*ht*p_v1[l])/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+1] = (x2 - scale*ht*p_v2[l])/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+2] = (x3 - scale*ht*p_v3[l])/(2.0*PETSC_PI); // normalized to [0,1]
            }  // i1
        }  // i2
    }  // i3
    //ierr = v->RestoreArraysRead(p_v1, p_v2, p_v3); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v->m_X1, &p_v1); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v->m_X2, &p_v2); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v->m_X3, &p_v3); CHKERRQ(ierr);


    // communicate the characteristic
    ierr = this->CommunicateCoord(flag); CHKERRQ(ierr);

    
    // interpolate velocity field v(X)
    PUSH_RANGE("Interpolate",2)
    ierr = this->Interpolate(this->m_WorkVecField1, v, flag); CHKERRQ(ierr);
    POP_RANGE
    

    // X = x - 0.5*ht*(v + v(x - ht v))
    //ierr = v->GetArraysRead(p_v1, p_v2, p_v3); CHKERRQ(ierr);
    ierr = VecGetArrayRead(v->m_X1, &p_v1); CHKERRQ(ierr);
    ierr = VecGetArrayRead(v->m_X2, &p_v2); CHKERRQ(ierr);
    ierr = VecGetArrayRead(v->m_X3, &p_v3); CHKERRQ(ierr);
    //ierr = this->m_WorkVecField1->GetArrays(p_vX1, p_vX2, p_vX3); CHKERRQ(ierr);
    ierr = VecGetArray(this->m_WorkVecField1->m_X1, &p_vX1); CHKERRQ(ierr);
    ierr = VecGetArray(this->m_WorkVecField1->m_X2, &p_vX2); CHKERRQ(ierr);
    ierr = VecGetArray(this->m_WorkVecField1->m_X3, &p_vX3); CHKERRQ(ierr);
    for (i1 = 0; i1 < isize[0]; ++i1) {  // x1
        for (i2 = 0; i2 < isize[1]; ++i2) {  // x2
            for (i3 = 0; i3 < isize[2]; ++i3) {  // x3
                // compute linear / flat index
                l = GetLinearIndex(i1, i2, i3, isize);
                
                // compute coordinates (nodal grid)
                if (mX) {
                  x1 = mX[l*3 + 0];
                  x2 = mX[l*3 + 1];
                  x3 = mX[l*3 + 2];
                } else {
                  x1 = hx[0]*static_cast<ScalarType>(i1 + istart[0]);
                  x2 = hx[1]*static_cast<ScalarType>(i2 + istart[1]);
                  x3 = hx[2]*static_cast<ScalarType>(i3 + istart[2]); 
                }
                
                this->m_X[l*3+0] = (x1 - scale*hthalf*(p_vX1[l] + p_v1[l]))/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+1] = (x2 - scale*hthalf*(p_vX2[l] + p_v2[l]))/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+2] = (x3 - scale*hthalf*(p_vX3[l] + p_v3[l]))/(2.0*PETSC_PI); // normalized to [0,1]
            }  // i1
        }  // i2
    }  // i3
    //ierr = this->m_WorkVecField1->RestoreArrays(p_vX1, p_vX2, p_vX3); CHKERRQ(ierr);
    ierr = VecRestoreArray(this->m_WorkVecField1->m_X1, &p_vX1); CHKERRQ(ierr);
    ierr = VecRestoreArray(this->m_WorkVecField1->m_X2, &p_vX2); CHKERRQ(ierr);
    ierr = VecRestoreArray(this->m_WorkVecField1->m_X3, &p_vX3); CHKERRQ(ierr);
    //ierr = v->RestoreArraysRead(p_v1, p_v2, p_v3); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v->m_X1, &p_v1); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v->m_X2, &p_v2); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v->m_X3, &p_v3); CHKERRQ(ierr);

    // communicate the characteristic
    ierr = this->CommunicateCoord(flag); CHKERRQ(ierr);

    this->m_Opt->Exit(__func__);
    //POP_RANGE
    PetscFunctionReturn(ierr);
}




/********************************************************************
 * @brief compute the trajectory from the velocity field based
 * on an rk2 scheme (todo: make the velocity field a const vector)
 *******************************************************************/
PetscErrorCode SemiLagrangian::ComputeTrajectoryRK4(VecField* v, std::string flag, ScalarType *mX) {
    PetscErrorCode ierr = 0;
    ScalarType ht, hthalf, hx[3], x1, x2, x3, scale = 0.0;
    const ScalarType *p_v1 = NULL, *p_v2 = NULL, *p_v3 = NULL;
    ScalarType *p_vX1 = NULL, *p_vX2 = NULL, *p_vX3 = NULL,
               *p_f1 = NULL, *p_f2 = NULL, *p_f3 = NULL;
    IntType isize[3], istart[3], l, i1, i2, i3;
    std::stringstream ss;

    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian",3)
    this->m_Opt->Enter(__func__);

    ierr = Assert(this->m_WorkVecField1 != NULL, "null pointer"); CHKERRQ(ierr);
    if (this->m_WorkVecField2 == NULL) {
        try{this->m_WorkVecField2 = new VecField(this->m_Opt);}
        catch (std::bad_alloc&) {
            ierr = reg::ThrowError("allocation failed"); CHKERRQ(ierr);
        }
    }

    ht = this->m_Opt->GetTimeStepSize();
    hthalf = 0.5*ht;

    // switch between state and adjoint variable
    if (strcmp(flag.c_str(), "state") == 0) {
        scale =  1.0;
    } else if (strcmp(flag.c_str(), "adjoint") == 0) {
        scale = -1.0;
    } else {
        ierr = ThrowError("flag wrong"); CHKERRQ(ierr);
    }


    for (int i = 0; i < 3; ++i) {
        hx[i] = this->m_Opt->m_Domain.hx[i];
        isize[i] = this->m_Opt->m_Domain.isize[i];
        istart[i] = this->m_Opt->m_Domain.istart[i];
    }

    ierr = this->m_WorkVecField2->GetArrays(p_f1, p_f2, p_f3); CHKERRQ(ierr);

    // first stage of rk4
    ierr = v->GetArraysRead(p_v1, p_v2, p_v3); CHKERRQ(ierr);
    for (i1 = 0; i1 < isize[0]; ++i1) {   // x1
        for (i2 = 0; i2 < isize[1]; ++i2) {   // x2
            for (i3 = 0; i3 < isize[2]; ++i3) {   // x3
                // compute coordinates (nodal grid)
                x1 = hx[0]*static_cast<ScalarType>(i1 + istart[0]);
                x2 = hx[1]*static_cast<ScalarType>(i2 + istart[1]);
                x3 = hx[2]*static_cast<ScalarType>(i3 + istart[2]);

                // compute linear / flat index
                l = GetLinearIndex(i1, i2, i3, isize);

                p_f1[l] = p_v1[l];
                p_f2[l] = p_v2[l];
                p_f3[l] = p_v3[l];

                this->m_X[l*3+0] = (x1 - scale*hthalf*p_v1[l])/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+1] = (x2 - scale*hthalf*p_v2[l])/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+2] = (x3 - scale*hthalf*p_v3[l])/(2.0*PETSC_PI); // normalized to [0,1]
            }  // i1
        }  // i2
    }  // i3
    ierr = v->RestoreArraysRead(p_v1, p_v2, p_v3); CHKERRQ(ierr);

    // evaluate right hand side
    ierr = this->CommunicateCoord(flag); CHKERRQ(ierr);
    PUSH_RANGE("Interpolate",2)
    ierr = this->Interpolate(this->m_WorkVecField1, v, flag); CHKERRQ(ierr);
    POP_RANGE

    // second stage of rk4
    ierr = this->m_WorkVecField1->GetArrays(p_vX1, p_vX2, p_vX3); CHKERRQ(ierr);
    for (i1 = 0; i1 < isize[0]; ++i1) {  // x1
        for (i2 = 0; i2 < isize[1]; ++i2) {  // x2
            for (i3 = 0; i3 < isize[2]; ++i3) {  // x3
                // compute coordinates (nodal grid)
                x1 = hx[0]*static_cast<ScalarType>(i1 + istart[0]);
                x2 = hx[1]*static_cast<ScalarType>(i2 + istart[1]);
                x3 = hx[2]*static_cast<ScalarType>(i3 + istart[2]);

                // compute linear / flat index
                l = GetLinearIndex(i1, i2, i3, isize);

                p_f1[l] += 2.0*p_vX1[l];
                p_f2[l] += 2.0*p_vX2[l];
                p_f3[l] += 2.0*p_vX3[l];

                this->m_X[l*3+0] = (x1 - scale*hthalf*p_vX1[l])/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+1] = (x2 - scale*hthalf*p_vX2[l])/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+2] = (x3 - scale*hthalf*p_vX3[l])/(2.0*PETSC_PI); // normalized to [0,1]
            }  // i1
        }  // i2
    }  // i3
    ierr = this->m_WorkVecField1->RestoreArrays(p_vX1, p_vX2, p_vX3); CHKERRQ(ierr);

    // evaluate right hand side
    ierr = this->CommunicateCoord(flag); CHKERRQ(ierr);
    PUSH_RANGE("Interpolate",2)
    ierr = this->Interpolate(this->m_WorkVecField1, v, flag); CHKERRQ(ierr);
    POP_RANGE

    // third stage of rk4
    ierr = this->m_WorkVecField1->GetArrays(p_vX1, p_vX2, p_vX3); CHKERRQ(ierr);
    for (i1 = 0; i1 < isize[0]; ++i1) {  // x1
        for (i2 = 0; i2 < isize[1]; ++i2) {  // x2
            for (i3 = 0; i3 < isize[2]; ++i3) {  // x3
                // compute coordinates (nodal grid)
                x1 = hx[0]*static_cast<ScalarType>(i1 + istart[0]);
                x2 = hx[1]*static_cast<ScalarType>(i2 + istart[1]);
                x3 = hx[2]*static_cast<ScalarType>(i3 + istart[2]);

                // compute linear / flat index
                l = GetLinearIndex(i1, i2, i3, isize);

                p_f1[l] += 2.0*p_vX1[l];
                p_f2[l] += 2.0*p_vX2[l];
                p_f3[l] += 2.0*p_vX3[l];

                this->m_X[l*3+0] = (x1 - scale*ht*p_vX1[l])/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+1] = (x2 - scale*ht*p_vX2[l])/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+2] = (x3 - scale*ht*p_vX3[l])/(2.0*PETSC_PI); // normalized to [0,1]
            }  // i1
        }  // i2
    }  // i3
    ierr = this->m_WorkVecField1->RestoreArrays(p_vX1, p_vX2, p_vX3); CHKERRQ(ierr);

    // evaluate right hand side
    ierr = this->CommunicateCoord(flag); CHKERRQ(ierr);
    PUSH_RANGE("Interpolate",2)
    ierr = this->Interpolate(this->m_WorkVecField1, v, flag); CHKERRQ(ierr);
    POP_RANGE

    // fourth stage of rk4
    ierr = this->m_WorkVecField1->GetArrays(p_vX1, p_vX2, p_vX3); CHKERRQ(ierr);
    for (i1 = 0; i1 < isize[0]; ++i1) {  // x1
        for (i2 = 0; i2 < isize[1]; ++i2) {  // x2
            for (i3 = 0; i3 < isize[2]; ++i3) {  // x3
                // compute coordinates (nodal grid)
                x1 = hx[0]*static_cast<ScalarType>(i1 + istart[0]);
                x2 = hx[1]*static_cast<ScalarType>(i2 + istart[1]);
                x3 = hx[2]*static_cast<ScalarType>(i3 + istart[2]);

                // compute linear / flat index
                l = GetLinearIndex(i1, i2, i3, isize);

                p_f1[l] += p_vX1[l];
                p_f2[l] += p_vX2[l];
                p_f3[l] += p_vX3[l];

                this->m_X[l*3+0] = (x1 - scale*(ht/6.0)*p_f1[l])/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+1] = (x2 - scale*(ht/6.0)*p_f2[l])/(2.0*PETSC_PI); // normalized to [0,1]
                this->m_X[l*3+2] = (x3 - scale*(ht/6.0)*p_f3[l])/(2.0*PETSC_PI); // normalized to [0,1]
            }  // i1
        }  // i2
    }  // i3
    ierr = this->m_WorkVecField1->RestoreArrays(p_vX1, p_vX2, p_vX3); CHKERRQ(ierr);

    ierr = this->m_WorkVecField2->RestoreArrays(p_f1, p_f2, p_f3); CHKERRQ(ierr);

    // communicate the final characteristic
    ierr = this->CommunicateCoord(flag); CHKERRQ(ierr);

    this->m_Opt->Exit(__func__);
    //POP_RANGE
    PetscFunctionReturn(ierr);
}




/********************************************************************
 * @brief set coordinate vector and communicate to interpolation plan
 *******************************************************************/
PetscErrorCode SemiLagrangian::SetQueryPoints(ScalarType* y1, ScalarType* y2, ScalarType* y3, std::string flag) {
    PetscErrorCode ierr = 0;
    IntType nl;
    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian",3)
    this->m_Opt->Enter(__func__);

    nl = this->m_Opt->m_Domain.nl;

    // if query points have not yet been allocated
    if (this->m_X == NULL) {
        try {this->m_X = new ScalarType[3*nl];}
        catch (std::bad_alloc& err) {
            ierr = reg::ThrowError(err); CHKERRQ(ierr);
        }
    }

    // copy data to a flat vector
    for (IntType i = 0; i < nl; ++i) {
        this->m_X[3*i+0] = y1[i]/(2.0*PETSC_PI);
        this->m_X[3*i+1] = y2[i]/(2.0*PETSC_PI);
        this->m_X[3*i+2] = y3[i]/(2.0*PETSC_PI);
    }

    // evaluate right hand side
    ierr = this->CommunicateCoord(flag); CHKERRQ(ierr);

    this->m_Opt->Exit(__func__);
    //POP_RANGE
    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief set coordinate vector and communicate to interpolation plan
 *******************************************************************/
PetscErrorCode SemiLagrangian::GetQueryPoints(ScalarType* y1, ScalarType* y2, ScalarType* y3) {
    PetscErrorCode ierr = 0;
    IntType nl;
    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian",3)
    this->m_Opt->Enter(__func__);

    nl = this->m_Opt->m_Domain.nl;

    ierr = Assert(this->m_X != NULL, "null pointer"); CHKERRQ(ierr);

    // copy data to a flat vector
    for (IntType i = 0; i < nl; ++i) {
        y1[i] = this->m_X[3*i+0]*(2.0*PETSC_PI);
        y2[i] = this->m_X[3*i+1]*(2.0*PETSC_PI);
        y3[i] = this->m_X[3*i+2]*(2.0*PETSC_PI);
    }

    this->m_Opt->Exit(__func__);
    //POP_RANGE
    PetscFunctionReturn(ierr);
}

PetscErrorCode SemiLagrangian::GetQueryPoints(ScalarType* y) {
    PetscErrorCode ierr = 0;
    IntType nl;
    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian",3)
    this->m_Opt->Enter(__func__);

    nl = this->m_Opt->m_Domain.nl;

    ierr = Assert(this->m_X != NULL, "null pointer"); CHKERRQ(ierr);

    // copy data to a flat vector
    for (IntType i = 0; i < nl*3; ++i) {
        y[i] = this->m_X[i]*(2.0*PETSC_PI);
    }

    this->m_Opt->Exit(__func__);
    //POP_RANGE
    PetscFunctionReturn(ierr);
}



/********************************************************************
 * @brief interpolate scalar field
 *******************************************************************/
PetscErrorCode SemiLagrangian::Interpolate(Vec* xo, Vec xi, std::string flag) {
    PetscErrorCode ierr = 0;
    ScalarType *p_xo = NULL, *p_xi = NULL;

    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian",3)
    this->m_Opt->Enter(__func__);

    ierr = Assert(*xo != NULL, "null pointer"); CHKERRQ(ierr);
    ierr = Assert( xi != NULL, "null pointer"); CHKERRQ(ierr);

    ierr = GetRawPointer( xi, &p_xi); CHKERRQ(ierr);
    ierr = GetRawPointer(*xo, &p_xo); CHKERRQ(ierr);
    
    //PUSH_RANGE("SL_Interpolate",2)
    ierr = this->Interpolate(p_xo, p_xi, flag); CHKERRQ(ierr);
    //POP_RANGE

    ierr = RestoreRawPointer(*xo, &p_xo); CHKERRQ(ierr);
    ierr = RestoreRawPointer( xi, &p_xi); CHKERRQ(ierr);

    this->m_Opt->Exit(__func__);
    //POP_RANGE
    PetscFunctionReturn(ierr);
}




/********************************************************************
 * @brief interpolate scalar field
 *******************************************************************/
PetscErrorCode SemiLagrangian::Interpolate(ScalarType* xo, ScalarType* xi, std::string flag) {
    PetscErrorCode ierr = 0;
    int nx[3], isize_g[3], isize[3], istart_g[3], istart[3], c_dims[2], neval, order, nghost;
    IntType nl, nalloc;
    std::stringstream ss;
    double timers[4] = {0, 0, 0, 0};

    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian",3)

    this->m_Opt->Enter(__func__);
    
    ZeitGeist_define(SL_INTERPOL);
    ZeitGeist_tick(SL_INTERPOL);
    
    ierr = Assert(xi != NULL, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(xo != NULL, "null pointer"); CHKERRQ(ierr);

    ierr = this->m_Opt->StartTimer(IPSELFEXEC); CHKERRQ(ierr);

    nl     = this->m_Opt->m_Domain.nl;
    order  = this->m_Opt->m_PDESolver.iporder;
    nghost = order;
    neval  = static_cast<int>(nl);
    
#ifdef REG_HAS_CUDA
    ScalarType *xi_d = xi;
    ScalarType *xo_d = xo;
    
    if (this->m_X == NULL) {
      try {this->m_X = reinterpret_cast<ScalarType*>(accfft_alloc(3*nl*sizeof(ScalarType)));}
      catch (std::bad_alloc& err) {
          ierr = reg::ThrowError(err); CHKERRQ(ierr);
      }
    }
    
    xi = &m_X[0*nl];
    xo = &m_X[1*nl];
    
    cudaMemcpy(xi, xi_d, nl*sizeof(ScalarType), cudaMemcpyDeviceToHost);
#endif

    for (int i = 0; i < 3; ++i) {
        nx[i]     = static_cast<int>(this->m_Opt->m_Domain.nx[i]);
        isize[i]  = static_cast<int>(this->m_Opt->m_Domain.isize[i]);
        istart[i] = static_cast<int>(this->m_Opt->m_Domain.istart[i]);
    }

    c_dims[0] = this->m_Opt->m_CartGridDims[0];
    c_dims[1] = this->m_Opt->m_CartGridDims[1];

    // deal with ghost points
    nalloc = accfft_ghost_xyz_local_size_dft_r2c(this->m_Opt, nghost, isize_g, istart_g);
    
    // if scalar field with ghost points has not been allocated
    if (this->m_ScaFieldGhost == NULL) {
        this->m_ScaFieldGhost = reinterpret_cast<ScalarType*>(accfft_alloc(nalloc));
    }

    // assign ghost points based on input scalar field
    accfft_get_ghost_xyz(this->m_Opt, nghost, isize_g, xi, this->m_ScaFieldGhost);
    
    // compute interpolation for all components of the input scalar field
    if (strcmp(flag.c_str(), "state") == 0) {
        ierr = Assert(this->m_StatePlan != NULL, "null pointer"); CHKERRQ(ierr);
        //PUSH_RANGE("SL_Interpolate",2)
        this->m_StatePlan->interpolate(this->m_ScaFieldGhost, nx, isize, istart,
                                       neval, nghost, xo, c_dims, this->m_Opt->m_FFT.mpicomm, timers, 0);
        //POP_RANGE
    } else if (strcmp(flag.c_str(), "adjoint") == 0) {
        ierr = Assert(this->m_AdjointPlan != NULL, "null pointer"); CHKERRQ(ierr);
        //PUSH_RANGE("SL_Interpolate",2)
        this->m_AdjointPlan->interpolate(this->m_ScaFieldGhost, nx, isize, istart,
                                       neval, nghost, xo, c_dims, this->m_Opt->m_FFT.mpicomm, timers, 0);
        //POP_RANGE
    } else {
        ierr = ThrowError("flag wrong"); CHKERRQ(ierr);
    }
    
    ierr = this->m_Opt->StopTimer(IPSELFEXEC); CHKERRQ(ierr);
    this->m_Opt->IncreaseInterpTimers(timers);
    this->m_Opt->IncrementCounter(IP);
    
    ZeitGeist_tock(SL_INTERPOL);
    
#ifdef REG_HAS_CUDA
    cudaMemcpy(xo_d, xo, nl*sizeof(ScalarType), cudaMemcpyHostToDevice);
#endif

    this->m_Opt->Exit(__func__);
    //POP_RANGE
    PetscFunctionReturn(0);
}




/********************************************************************
 * @brief interpolate vector field
 *******************************************************************/
PetscErrorCode SemiLagrangian::Interpolate(VecField* vo, VecField* vi, std::string flag) {
    PetscErrorCode ierr = 0;
    ScalarType *p_vix1 = NULL, *p_vix2 = NULL, *p_vix3 = NULL,
               *p_vox1 = NULL, *p_vox2 = NULL, *p_vox3 = NULL;
    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian",3)
    this->m_Opt->Enter(__func__);

    ierr = Assert(vi != NULL, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(vo != NULL, "null pointer"); CHKERRQ(ierr);

    ierr = vi->GetArrays(p_vix1, p_vix2, p_vix3); CHKERRQ(ierr);
    ierr = vo->GetArrays(p_vox1, p_vox2, p_vox3); CHKERRQ(ierr);

    //PUSH_RANGE("SL_Interpolate",2)
    ierr = this->Interpolate(p_vox1, p_vox2, p_vox3, p_vix1, p_vix2, p_vix3, flag); CHKERRQ(ierr);
    //POP_RANGE

    ierr = vo->RestoreArrays(p_vox1, p_vox2, p_vox3); CHKERRQ(ierr);
    ierr = vi->RestoreArrays(p_vix1, p_vix2, p_vix3); CHKERRQ(ierr);

    this->m_Opt->Exit(__func__);
    //POP_RANGE
    PetscFunctionReturn(ierr);
}




/********************************************************************
 * @brief interpolate vector field
 *******************************************************************/
PetscErrorCode SemiLagrangian::Interpolate(ScalarType* wx1, ScalarType* wx2, ScalarType* wx3,
                                           ScalarType* vx1, ScalarType* vx2, ScalarType* vx3, std::string flag) {
    PetscErrorCode ierr = 0;
    int nx[3], isize_g[3], isize[3], istart_g[3], istart[3], c_dims[2], nghost, order;
    double timers[4] = {0, 0, 0, 0};
    std::stringstream ss;
    IntType nl, nlghost, nalloc;
    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian_Interpolate",3)
    

    this->m_Opt->Enter(__func__);
    
    ZeitGeist_define(SL_INTERPOL);
    ZeitGeist_tick(SL_INTERPOL);

    ierr = Assert(vx1 != NULL, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(vx2 != NULL, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(vx3 != NULL, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(wx1 != NULL, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(wx2 != NULL, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(wx3 != NULL, "null pointer"); CHKERRQ(ierr);

    nl = this->m_Opt->m_Domain.nl;
    order = this->m_Opt->m_PDESolver.iporder;
    nghost = order;


    for (int i = 0; i < 3; ++i) {
        nx[i] = static_cast<int>(this->m_Opt->m_Domain.nx[i]);
        isize[i] = static_cast<int>(this->m_Opt->m_Domain.isize[i]);
        istart[i] = static_cast<int>(this->m_Opt->m_Domain.istart[i]);
    }

    // get network dimensions
    c_dims[0] = this->m_Opt->m_CartGridDims[0];
    c_dims[1] = this->m_Opt->m_CartGridDims[1];
    

    if (this->m_X == NULL) {
        try {this->m_X = reinterpret_cast<ScalarType*>(accfft_alloc(3*nl*sizeof(ScalarType)));}
        catch (std::bad_alloc& err) {
            ierr = reg::ThrowError(err); CHKERRQ(ierr);
        }
    }
    
    
#ifdef REG_HAS_CUDA
 //PUSH_RANGE("SemiLagrangian_Interpolate_memcpy",3)
    cudaMemcpy(&this->m_X[0*nl], vx1, nl*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    cudaMemcpy(&this->m_X[1*nl], vx2, nl*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    cudaMemcpy(&this->m_X[2*nl], vx3, nl*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    cudaCheckLastError();
    //POP_RANGE
#else
    // copy data to a flat vector
    for (IntType i = 0; i < nl; ++i) {
        this->m_X[0*nl+i] = vx1[i];
        this->m_X[1*nl+i] = vx2[i];
        this->m_X[2*nl+i] = vx3[i];
    }
#endif
    

    ierr = this->m_Opt->StartTimer(IPSELFEXEC); CHKERRQ(ierr);
    
      
    // get ghost sizes
    nalloc = accfft_ghost_xyz_local_size_dft_r2c(this->m_Opt, nghost, isize_g, istart_g);

    // get nl for ghosts
    nlghost = 1;
    for (int i = 0; i < 3; ++i) {
        nlghost *= static_cast<IntType>(isize_g[i]);
    }

    // deal with ghost points
    if (this->m_VecFieldGhost == NULL) {
        this->m_VecFieldGhost = reinterpret_cast<ScalarType*>(accfft_alloc(3*nalloc));
    }

      
    // do the communication for the ghost points
    for (int i = 0; i < 3; i++) {
        accfft_get_ghost_xyz(this->m_Opt, nghost, isize_g, &this->m_X[i*nl],
                             &this->m_VecFieldGhost[i*nlghost]);
    }
    

    if (strcmp(flag.c_str(),"state") == 0) {
        ierr = Assert(this->m_StatePlan != NULL, "null pointer"); CHKERRQ(ierr);
        //PUSH_RANGE("SL_Interpolate",2)
        this->m_StatePlan->interpolate(this->m_VecFieldGhost, nx, isize, istart,
                                       nl, nghost, this->m_X, c_dims, this->m_Opt->m_FFT.mpicomm, timers, 1);
        //POP_RANGE
    } else if (strcmp(flag.c_str(),"adjoint") == 0) {
        ierr = Assert(this->m_AdjointPlan != NULL, "null pointer"); CHKERRQ(ierr);
        //PUSH_RANGE("SL_Interpolate",2)
        this->m_AdjointPlan->interpolate(this->m_VecFieldGhost, nx, isize, istart,
                                         nl, nghost, this->m_X, c_dims, this->m_Opt->m_FFT.mpicomm, timers, 1);
        //POP_RANGE
    } else {
        ierr = ThrowError("flag wrong"); CHKERRQ(ierr);
    }

    ierr = this->m_Opt->StopTimer(IPSELFEXEC); CHKERRQ(ierr);

#ifdef REG_HAS_CUDA
 //PUSH_RANGE("SemiLagrangian_Interpolate_memcpy",3)
    cudaMemcpy(wx1, &this->m_X[0*nl], nl*sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(wx2, &this->m_X[1*nl], nl*sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(wx3, &this->m_X[2*nl], nl*sizeof(ScalarType), cudaMemcpyHostToDevice);
    //POP_RANGE
#else
    for (IntType i = 0; i < nl; ++i) {
        wx1[i] = this->m_X[0*nl+i];
        wx2[i] = this->m_X[1*nl+i];
        wx3[i] = this->m_X[2*nl+i];
    }
#endif
    

    this->m_Opt->IncreaseInterpTimers(timers);
    this->m_Opt->IncrementCounter(IPVEC);
    
    ZeitGeist_tock(SL_INTERPOL);
    ZeitGeist_inc(SL_INTERPOL);
    ZeitGeist_inc(SL_INTERPOL);
    
    //POP_RANGE
    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(0);
}




/********************************************************************
 * @brief communicate the coordinate vector (query points)
 * @param flag to switch between forward and adjoint solves
 *******************************************************************/
PetscErrorCode SemiLagrangian::CommunicateCoord(std::string flag) {
    PetscErrorCode ierr;
    int nx[3], nl, isize[3], istart[3], nghost;
    int c_dims[2];
    double timers[4] = {0, 0, 0, 0};
    std::stringstream ss;
    PetscFunctionBegin;
    //PUSH_RANGE("SemiLagrangian",3)

    this->m_Opt->Enter(__func__);

    ierr = Assert(this->m_Opt->m_FFT.mpicomm != NULL, "null pointer"); CHKERRQ(ierr);

    // get sizes
    nl     = static_cast<int>(this->m_Opt->m_Domain.nl);
    nghost = this->m_Opt->m_PDESolver.iporder;
    for (int i = 0; i < 3; ++i) {
        nx[i] = static_cast<int>(this->m_Opt->m_Domain.nx[i]);
        isize[i] = static_cast<int>(this->m_Opt->m_Domain.isize[i]);
        istart[i] = static_cast<int>(this->m_Opt->m_Domain.istart[i]);
    }

    // get network dimensions
    c_dims[0] = this->m_Opt->m_CartGridDims[0];
    c_dims[1] = this->m_Opt->m_CartGridDims[1];

    ierr = this->m_Opt->StartTimer(IPSELFEXEC); CHKERRQ(ierr);

    if (strcmp(flag.c_str(), "state") == 0) {
        // characteristic for state equation should have been computed already
        ierr = Assert(this->m_X != NULL, "null pointer"); CHKERRQ(ierr);
        // create planer
        if (this->m_StatePlan == NULL) {
            if (this->m_Opt->m_Verbosity > 2) {
                ierr = DbgMsg("allocating state plan"); CHKERRQ(ierr);
            }
            try {this->m_StatePlan = new Interp3_Plan();}
            catch (std::bad_alloc& err) {
                ierr = reg::ThrowError(err); CHKERRQ(ierr);
            }
            this->m_StatePlan->allocate(nl, this->m_Dofs, 2);
        }

        // scatter
        this->m_StatePlan->scatter(nx, isize, istart, nl, nghost, this->m_X,
                                   c_dims, this->m_Opt->m_FFT.mpicomm, timers);
    } else if (strcmp(flag.c_str(), "adjoint") == 0) {
        // characteristic for adjoint equation should have been computed already
        ierr = Assert(this->m_X != NULL, "null pointer"); CHKERRQ(ierr);
        // create planer
        if (this->m_AdjointPlan == NULL) {
            if (this->m_Opt->m_Verbosity > 2) {
                ierr = DbgMsg("allocating adjoint plan"); CHKERRQ(ierr);
            }
            try {this->m_AdjointPlan = new Interp3_Plan();}
            catch (std::bad_alloc& err) {
                ierr = reg::ThrowError(err); CHKERRQ(ierr);
            }
            this->m_AdjointPlan->allocate(nl, this->m_Dofs, 2);
        }

        // communicate coordinates
        this->m_AdjointPlan->scatter(nx, isize, istart, nl, nghost, this->m_X,
                                     c_dims, this->m_Opt->m_FFT.mpicomm, timers);
    } else {
        ierr = ThrowError("flag wrong"); CHKERRQ(ierr);
    }

    ierr = this->m_Opt->StopTimer(IPSELFEXEC); CHKERRQ(ierr);

    this->m_Opt->IncreaseInterpTimers(timers);

    this->m_Opt->Exit(__func__);
    //POP_RANGE
    PetscFunctionReturn(0);
}




}  // namespace reg




#endif  // _SEMILAGRANGIAN_CPP_
