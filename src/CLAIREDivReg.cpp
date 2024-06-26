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

#ifndef _OPTIMALCONTROLREGISTRATIONRELAXEDIC_CPP_
#define _OPTIMALCONTROLREGISTRATIONRELAXEDIC_CPP_

#include "CLAIREDivReg.hpp"

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
CLAIREDivReg::CLAIREDivReg() : SuperClass() {
    this->Initialize();
}

/********************************************************************
 * @brief default destructor
 *******************************************************************/
CLAIREDivReg::~CLAIREDivReg() {
    this->ClearMemory();
}

/********************************************************************
 * @brief constructor
 *******************************************************************/
CLAIREDivReg::CLAIREDivReg(RegOpt* opt) : SuperClass(opt) {
    this->Initialize();
}

/********************************************************************
 * @brief init variables
 *******************************************************************/
PetscErrorCode CLAIREDivReg::Initialize() {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief clean up
 *******************************************************************/
PetscErrorCode CLAIREDivReg::ClearMemory() {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    PetscFunctionReturn(ierr);
}

PetscErrorCode CLAIREDivReg::CreateCoarseReg() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  ierr = Assert(this->m_CoarseRegOpt != nullptr, "coarse grid RegOpt not initialized"); CHKERRQ(ierr);
  
  ierr = AllocateOnce<CLAIREDivReg>(this->m_CoarseReg, this->m_CoarseRegOpt); CHKERRQ(ierr);
  
  PetscFunctionReturn(ierr);
}

/********************************************************************
 * @brief evaluates the objective value
 *******************************************************************/
PetscErrorCode CLAIREDivReg::EvaluateObjective(ScalarType* J, Vec v) {
    PetscErrorCode ierr = 0;
    ScalarType D = 0.0, Rv = 0.0, Rw = 0.0;
    std::stringstream ss;
    PetscFunctionBegin;
    
    this->m_Opt->Enter(__func__);

    // allocate velocity field
    //if (v) {
    //  ierr = Free(this->m_VelocityField); CHKERRQ(ierr);
    //  ierr = AllocateOnce(this->m_VelocityField, this->m_Opt, v); CHKERRQ(ierr);
    //} else {
      ierr = AllocateOnce(this->m_VelocityField, this->m_Opt); CHKERRQ(ierr);
      ierr = this->m_VelocityField->SetComponents(v); CHKERRQ(ierr);
    //}

    // allocate regularization model
    if (this->m_Regularization == NULL) {
        ierr = this->SetupRegularization(); CHKERRQ(ierr);
    }

    if (this->m_Opt->m_Verbosity > 2) {
        ierr = DbgMsg2("evaluating objective functional"); CHKERRQ(ierr);
    }

    ZeitGeist_define(EVAL_OBJ);
    ZeitGeist_tick(EVAL_OBJ);
    ierr = this->m_Opt->StartTimer(OBJEXEC); CHKERRQ(ierr);

    // set components of velocity field
    //ierr = this->m_VelocityField->SetComponents(v); CHKERRQ(ierr);

    // evaluate the L2 distance
    ierr = this->EvaluateDistanceMeasure(&D); CHKERRQ(ierr);
    
    // evaluate the regularization model
    ierr = this->IsVelocityZero(); CHKERRQ(ierr);
    if (!this->m_VelocityIsZero) {
        // evaluate the regularization model for v
        ierr = AllocateOnce(this->m_WorkVecField1, this->m_Opt); CHKERRQ(ierr);
        ierr = this->m_Regularization->SetWorkVecField(this->m_WorkVecField1); CHKERRQ(ierr);
        ierr = AllocateOnce(this->m_WorkVecField4, this->m_Opt); CHKERRQ(ierr);
        ierr = AllocateOnce(this->m_WorkScaField1, this->m_Opt, this->m_WorkVecField4->m_X1); CHKERRQ(ierr);
        ierr = AllocateOnce(this->m_WorkScaField2, this->m_Opt, this->m_WorkVecField4->m_X2); CHKERRQ(ierr);
        ierr = AllocateOnce(this->m_WorkScaField3, this->m_Opt, this->m_WorkVecField4->m_X3); CHKERRQ(ierr);
        ierr = this->m_Regularization->SetWorkScaField(this->m_WorkScaField1); CHKERRQ(ierr);
        if (this->m_Opt->m_Diff.diffPDE == FINITE && 
            (this->m_Opt->m_RegNorm.type == H1 || this->m_Opt->m_RegNorm.type == H1SN)) {
          ierr = AllocateOnce(this->m_DifferentiationFD, this->m_Opt); CHKERRQ(ierr);
          ierr = this->m_Regularization->SetDifferentiation(this->m_DifferentiationFD); CHKERRQ(ierr);
        }
        ierr = this->m_Regularization->EvaluateFunctional(&Rv, this->m_VelocityField); CHKERRQ(ierr);
        if (this->m_Opt->m_Diff.diffPDE == FINITE && 
            (this->m_Opt->m_RegNorm.type == H1 || this->m_Opt->m_RegNorm.type == H1SN)) {
          ierr = AllocateOnce<DifferentiationSM>(this->m_Differentiation, this->m_Opt); CHKERRQ(ierr);
          ierr = this->m_Regularization->SetDifferentiation(this->m_Differentiation); CHKERRQ(ierr);
        }
        // evaluate the regularization model for w = div(v)
        ierr = this->EvaluteRegularizationDIV(&Rw); CHKERRQ(ierr); CHKERRQ(ierr);
    }

    // add up the contributions
    *J = D + Rv + Rw;

    // store for access
    this->m_Opt->m_Monitor.jval = *J;
    this->m_Opt->m_Monitor.dval = D;
    this->m_Opt->m_Monitor.rval = Rv + Rw;

    if (this->m_Opt->m_Verbosity > 1) {
        ss << "J(v) = D(v) + R1(v) + R2(div(v)) = " << std::scientific
           << D << " + " << Rv << " + " << Rw;
        ierr = DbgMsg1(ss.str()); CHKERRQ(ierr);
    }

    ierr = this->m_Opt->StopTimer(OBJEXEC); CHKERRQ(ierr);
    ZeitGeist_tock(EVAL_OBJ);

    this->m_Opt->IncrementCounter(OBJEVAL);

    this->m_Opt->Exit(__func__);
    PetscFunctionReturn(0);
}

/********************************************************************
 * @brief compute the body force
 * b = K[\int_0^1 \igrad m \lambda d t],
 * where K is an operator that projects v onto the manifold of
 * divergence free velocity fields
 *******************************************************************/
PetscErrorCode CLAIREDivReg::EvaluteRegularizationDIV(ScalarType* Rw) {
    PetscErrorCode ierr = 0;
    ScalarType value, regvalue, betaw, hd;

    PetscFunctionBegin;
    this->m_Opt->Enter(__func__);

    hd  = this->m_Opt->GetLebesgueMeasure();   

    ierr = AllocateOnce(this->m_WorkVecField1, this->m_Opt); CHKERRQ(ierr);
    ierr = AllocateOnce(this->m_WorkVecField4, this->m_Opt); CHKERRQ(ierr);
    ierr = AllocateOnce(this->m_WorkScaField1, this->m_Opt, this->m_WorkVecField4->m_X1); CHKERRQ(ierr);
    ierr = AllocateOnce(this->m_WorkScaField2, this->m_Opt, this->m_WorkVecField4->m_X2); CHKERRQ(ierr);
    ierr = AllocateOnce(this->m_WorkScaField3, this->m_Opt, this->m_WorkVecField4->m_X3); CHKERRQ(ierr);

    // get regularization weight
    betaw = this->m_Opt->m_RegNorm.beta[2];

    //ierr = AllocateOnce(this->m_DifferentiationFD, this->m_Opt); CHKERRQ(ierr);
    Differentiation* diff = this->m_Differentiation;
    if (this->m_Opt->m_Diff.diffPDE == FINITE) {
      ierr = AllocateOnce(this->m_DifferentiationFD, this->m_Opt); CHKERRQ(ierr);
      diff = this->m_DifferentiationFD;
    }
    
    // compute \div(\vect{v})
    PUSH_RANGE("Differentiation",5)
    ierr = diff->Divergence(*this->m_WorkScaField1, this->m_VelocityField); CHKERRQ(ierr);
    POP_RANGE

    // compute gradient of div(v)
    PUSH_RANGE("Differentiation",5)
    ierr = diff->Gradient(this->m_WorkVecField1, *this->m_WorkScaField1); CHKERRQ(ierr);
    POP_RANGE

    // compute inner products ||\igrad w||_L2 + ||w||_L2
    regvalue = 0.0;
    ierr = VecTDot(this->m_WorkVecField1->m_X1, this->m_WorkVecField1->m_X1, &value); CHKERRQ(ierr); regvalue += value;
    ierr = VecTDot(this->m_WorkVecField1->m_X2, this->m_WorkVecField1->m_X2, &value); CHKERRQ(ierr); regvalue += value;
    ierr = VecTDot(this->m_WorkVecField1->m_X3, this->m_WorkVecField1->m_X3, &value); CHKERRQ(ierr); regvalue += value;
    ierr = VecTDot(*this->m_WorkScaField1, *this->m_WorkScaField1, &value); CHKERRQ(ierr); regvalue += value;

    // add up contributions
    *Rw = 0.5*hd*betaw*regvalue;

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(0);
}

/********************************************************************
 * @brief apply projection to map \tilde{v} onto the manifold
 * of divergence free velocity fields
 *******************************************************************/
PetscErrorCode CLAIREDivReg::ApplyProjection() {
    PetscErrorCode ierr = 0;
    ScalarType beta[3];

    PetscFunctionBegin;
    this->m_Opt->Enter(__func__);

    // allocate spectral data
    //ierr = this->SetupSpectralData(); CHKERRQ(ierr);
    
    beta[0] = this->m_Opt->m_RegNorm.beta[0];
    beta[2] = this->m_Opt->m_RegNorm.beta[2];
    
    ierr = AllocateOnce<DifferentiationSM>(this->m_Differentiation, this->m_Opt); CHKERRQ(ierr);
    
    if (this->m_Opt->m_Verbosity > 2) {
        ierr = DbgMsg2("starting Leray projection"); CHKERRQ(ierr);
        ierr = Assert(this->m_Differentiation != NULL, "null pointer"); CHKERRQ(ierr);
        ierr = Assert(this->m_WorkVecField1 != NULL, "null pointer"); CHKERRQ(ierr);
        ierr = Assert(this->m_WorkVecField2 != NULL, "null pointer"); CHKERRQ(ierr);
    }
    
    ierr = this->m_Differentiation->LerayOperator(this->m_WorkVecField2, this->m_WorkVecField2, beta[0], beta[2]); CHKERRQ(ierr);
        
    //ierr = this->m_WorkVecField2->AXPY(1.0, this->m_WorkVecField1); CHKERRQ(ierr);

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(0);
}

}  // namespace reg




#endif  // _OPTIMALCONTROLREGISTRATIONRELAXEDIC_CPP_
