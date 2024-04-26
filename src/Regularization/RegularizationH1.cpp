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

#ifndef _REGULARIZATIONH1_CPP_
#define _REGULARIZATIONH1_CPP_

#include "RegularizationH1.hpp"

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
RegularizationH1::RegularizationH1() : SuperClass() {
}

/********************************************************************
 * @brief default destructor
 *******************************************************************/
RegularizationH1::~RegularizationH1(void) {
    this->ClearMemory();
}

/********************************************************************
 * @brief constructor
 *******************************************************************/
RegularizationH1::RegularizationH1(RegOpt* opt) : SuperClass(opt) {
}

/********************************************************************
 * @brief evaluates the functional
 *******************************************************************/
PetscErrorCode RegularizationH1::EvaluateFunctional(ScalarType* R, VecField* v) {
    PetscErrorCode ierr;
    ScalarType value, beta[2], H1v, L2v, hd;
    
    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    beta[0] = this->m_Opt->m_RegNorm.beta[0];
    beta[1] = this->m_Opt->m_RegNorm.beta[1];
    hd  = this->m_Opt->GetLebesgueMeasure();   

    *R= 0.0;

    //if ((beta[0] != 0.0)  && (beta[1] != 0.0)) {
    if (beta[0] != 0.0) {
        ierr = Assert(v != NULL, "null pointer"); CHKERRQ(ierr);
        ierr = Assert(this->m_WorkVecField != NULL, "null pointer"); CHKERRQ(ierr);

        H1v = 0.0;

        // X1 gradient
        PUSH_RANGE("Differentiation",5)
        ierr = this->m_Differentiation->Gradient(this->m_WorkVecField, v->m_X1); 
        POP_RANGE
        CHKERRQ(ierr);
       
        // compute inner products
        ierr = VecTDot(this->m_WorkVecField->m_X1, this->m_WorkVecField->m_X1, &value); CHKERRQ(ierr); H1v +=value;
        ierr = VecTDot(this->m_WorkVecField->m_X2, this->m_WorkVecField->m_X2, &value); CHKERRQ(ierr); H1v +=value;
        ierr = VecTDot(this->m_WorkVecField->m_X3, this->m_WorkVecField->m_X3, &value); CHKERRQ(ierr); H1v +=value;

        // X2 gradient
        PUSH_RANGE("Differentiation",5)
        ierr = this->m_Differentiation->Gradient(this->m_WorkVecField, v->m_X2); 
        POP_RANGE
        CHKERRQ(ierr);
   
        // compute inner products
        ierr = VecTDot(this->m_WorkVecField->m_X1, this->m_WorkVecField->m_X1, &value); CHKERRQ(ierr); H1v +=value;
        ierr = VecTDot(this->m_WorkVecField->m_X2, this->m_WorkVecField->m_X2, &value); CHKERRQ(ierr); H1v +=value;
        ierr = VecTDot(this->m_WorkVecField->m_X3, this->m_WorkVecField->m_X3, &value); CHKERRQ(ierr); H1v +=value;

        // X3 gradient
        PUSH_RANGE("Differentiation",5)
        ierr = this->m_Differentiation->Gradient(this->m_WorkVecField, v->m_X3); 
        POP_RANGE
        CHKERRQ(ierr);
       
                // compute inner products
        ierr = VecTDot(this->m_WorkVecField->m_X1, this->m_WorkVecField->m_X1, &value); CHKERRQ(ierr); H1v +=value;
        ierr = VecTDot(this->m_WorkVecField->m_X2, this->m_WorkVecField->m_X2, &value); CHKERRQ(ierr); H1v +=value;
        ierr = VecTDot(this->m_WorkVecField->m_X3, this->m_WorkVecField->m_X3, &value); CHKERRQ(ierr); H1v +=value;

        L2v = 0.0;
        ierr = VecTDot(v->m_X1, v->m_X1, &value); CHKERRQ(ierr); L2v +=value;
        ierr = VecTDot(v->m_X2, v->m_X2, &value); CHKERRQ(ierr); L2v +=value;
        ierr = VecTDot(v->m_X3, v->m_X3, &value); CHKERRQ(ierr); L2v +=value;

        // add up contributions
        //*R = 0.5*(beta[0]*H1v + beta[1]*L2v);
        *R = 0.5*hd*beta[0]*(H1v + beta[1]*L2v);
    }

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(0);
}

/********************************************************************
 * @brief evaluates first variation of regularization norm
 *******************************************************************/
PetscErrorCode RegularizationH1::EvaluateGradient(VecField* dvR, VecField* v) {
    PetscErrorCode ierr = 0;
    ScalarType beta[2], hd;

    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    ierr = Assert(v != NULL, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(dvR != NULL, "null pointer"); CHKERRQ(ierr);

    beta[0] = this->m_Opt->m_RegNorm.beta[0];
    beta[1] = this->m_Opt->m_RegNorm.beta[1];
    hd  = this->m_Opt->GetLebesgueMeasure();   

    // if regularization weight is zero, do noting
    //if ( (beta[0] == 0.0)  && (beta[1] == 0.0) ) {
    if (beta[0] == 0.0) {
        ierr = dvR->SetValue(0.0); CHKERRQ(ierr);
    } else {
        //PUSH_RANGE("Differentiation->RegLapOp",5)
        ierr = this->m_Differentiation->RegLapOp(dvR, v, hd*beta[0], beta[1]); CHKERRQ(ierr);
        //POP_RANGE
    }

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(0);
}

/********************************************************************
 * @brief applies second variation of regularization norm to
 * a vector (note: the first and second variation are identical)
 *******************************************************************/
PetscErrorCode RegularizationH1::HessianMatVec(VecField* dvvR, VecField* vtilde) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    ierr = this->EvaluateGradient(dvvR, vtilde); CHKERRQ(ierr);

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(0);
}

/********************************************************************
 * @brief apply the inverse of the regularization operator; we
 * can invert this operator analytically due to the spectral
 * 
 *******************************************************************/
PetscErrorCode RegularizationH1::ApplyInverse(VecField* Ainvx, VecField* x, bool applysqrt) {
    PetscErrorCode ierr = 0;
    ScalarType beta[2];

    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    ierr = Assert(x != NULL, "null pointer"); CHKERRQ(ierr);
    ierr = Assert(Ainvx != NULL, "null pointer"); CHKERRQ(ierr);

    beta[0] = this->m_Opt->m_RegNorm.beta[0];
    beta[1] = this->m_Opt->m_RegNorm.beta[1];

    // if regularization weight is zero, do noting
//    if ((beta[0] == 0.0)  && (beta[1] == 0.0)) {
    if (beta[0] == 0.0) {
        ierr = VecCopy(x->m_X1, Ainvx->m_X1); CHKERRQ(ierr);
        ierr = VecCopy(x->m_X2, Ainvx->m_X2); CHKERRQ(ierr);
        ierr = VecCopy(x->m_X3, Ainvx->m_X3); CHKERRQ(ierr);
    } else {
        
        //PUSH_RANGE("Differentiation->InvRegLapOp",5)
        ierr = this->m_Differentiation->InvRegLapOp(Ainvx, x, applysqrt, beta[0], beta[1]); CHKERRQ(ierr);
        //POP_RANGE
    }

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(0);
}

/********************************************************************
 * @brief computes the largest and smallest eigenvalue of
 * the inverse regularization operator
 *******************************************************************/
PetscErrorCode RegularizationH1::GetExtremeEigValsInvOp(ScalarType& emin, ScalarType& emax) {
    PetscErrorCode ierr = 0;
    ScalarType w[3], beta1, beta2, regop;

    PetscFunctionBegin;

    this->m_Opt->Enter(__func__);

    beta1 = this->m_Opt->m_RegNorm.beta[0];
    beta2 = this->m_Opt->m_RegNorm.beta[1];

    // get max value
    w[0] = static_cast<ScalarType>(this->m_Opt->m_Domain.nx[0])/2.0;
    w[1] = static_cast<ScalarType>(this->m_Opt->m_Domain.nx[1])/2.0;
    w[2] = static_cast<ScalarType>(this->m_Opt->m_Domain.nx[2])/2.0;

    // compute largest value for operator
    regop = -(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]); // laplacian
    //regop = -beta1*regop + beta2; // -beta_1 * lap + beta_2
    regop = beta1*(-regop + beta2); // -beta_1 *(lap + beta_2)
    emin = 1.0/regop;
    emax = 1.0/beta2;  // 1.0/(\beta_1*0 + \beta_2)

    this->m_Opt->Exit(__func__);

    PetscFunctionReturn(ierr);
}

}  // namespace reg

#endif  // _REGULARIZATIONREGISTRATIONH1_CPP_
