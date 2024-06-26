/*************************************************************************
 *  Copyright (c) 2018.
 *  All rights reserved.
 *  This file is part of the CLAIRE library.
 *
 *  CLAIRE is free software: you c1n redistribute it and/or modify
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



#ifndef _TEXTUREDIFFERENTIATIONKERNEL_CPP_
#define _TEXTUREDIFFERENTIATIONKERNEL_CPP_

#include "TextureDifferentiationKernel.hpp"
#include "cuda_helper.hpp"
#include "cuda_profiler_api.h"
#define HALO 4 
#define spencil 4
#define lpencil 32
#define sharedrows 32
#define perthreadcomp 8


//const float h_c[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};
//const float h2_c[HALO+1] = {-205.f/72.f, 8.f / 5.f , -1.f / 5.f , 8.f / 315.f, -1.f / 560.f};


// device constants
/*__constant__ int nl.x, nl.y, nl.z;
__constant__ int nl.x, nl.y, nl.z;
__constant__ int ng.x, ng.y, ng.z;
__constant__ int halo.x, halo.y, halo.z;

__constant__ float inx.x, inx.y, inx.z;
__constant__ float d_invhx, d_invhy, d_invhz;
__constant__ float d_cx[HALO], d_cy[HALO], d_cz[HALO];
__constant__ float d_cxx[HALO+1], d_cyy[HALO+1], d_czz[HALO+1];*/

const int sx = spencil;
const int sy = sharedrows;
const int sxx = lpencil;
const int syy = sharedrows;


inline int3 make_int3(IntType x, IntType y, IntType z) {
  int3 r;
  r.x = x;
  r.y = y;
  r.z = z;
  return r;
}

__device__ inline int getLinearIdx(int i, int j, int k, int3 nl) {
    return i*nl.y*nl.z + j*nl.z + k;
}

/**********************************************************************************
 * @brief compute z-gradient using 8th order finite differencing
 * @param[in] f = input scalar field with ghost padding
 * @param[out] dfz = z-component of gradient
**********************************************************************************/
__global__ void mgpu_gradient_z(ScalarType* dfz, const ScalarType* f, int3 nl, int3 ng, int3 halo, const float ih) {
  __shared__ float s_f[sx][sy+2*HALO]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int i = blockIdx.z;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x + HALO;       // local k for shared memory ac3ess + halo offset
  int sj = threadIdx.y; // local j for shared memory ac3ess
  int zblock_width;
  int id,lid,rid;
  
  if (blockIdx.x < gridDim.x - 1) {
    zblock_width = blockDim.x;
  }
  else {
    zblock_width = nl.z - blockIdx.x*blockDim.x;
  }
  
  bool internal = (j < nl.y) && (threadIdx.x < zblock_width);
  
  const float cz[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};
  
  if (internal) {
    id = getLinearIdx(i+halo.x,j+halo.y,k+halo.z, ng);
    s_f[sj][sk] = f[id];
  }
  
  __syncthreads();
    
  // fill in periodic images in shared memory array 
  if (threadIdx.x < HALO) {
    if (halo.z == 0) {
      lid = k%nl.z-HALO;
      if (lid<0) lid+=nl.z;
      id = getLinearIdx(i+halo.x, j+halo.y, lid, ng);
      s_f[sj][sk-HALO] = f[id];
      rid = (k+zblock_width)%nl.z;
      id = getLinearIdx(i+halo.x, j+halo.y, rid, ng);
      s_f[sj][zblock_width+sk] = f[id];
    } else {
      id = getLinearIdx(i+halo.x, j+halo.y, k, ng);
      s_f[sj][sk-HALO] = f[id];
      id = getLinearIdx(i+halo.x, j+halo.y, k+zblock_width+halo.z, ng);
      s_f[sj][zblock_width+sk] = f[id];
    }
  }
  
  __syncthreads();
  
  ScalarType result = 0;
  if (internal) {
    id = getLinearIdx(i,j,k, nl);
    for(int l=0; l<HALO; l++) {
      result += cz[l] * (s_f[sj][sk+1+l] - s_f[sj][sk-1-l]);
    }
    dfz[id] += result*ih;
  }
}



/**********************************************************************************
 * @brief compute y-component of gradient using 8th order finite differencing
 * @param[in]    f = scalar field with ghost layer padding
 * @param[out] dfy = y-component of gradient
**********************************************************************************/
__global__ void mgpu_gradient_y(ScalarType* dfy, const ScalarType* f, int3 nl, int3 ng, int3 halo, const float ih) {
  __shared__ float s_f[syy+2*HALO][sxx]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int i  = blockIdx.z;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x;       // local k for shared memory ac3ess, fixed
  int yblock_width, id, lid, rid, sj;
  bool internal;
  
  if ( blockIdx.y < gridDim.y - 1) {
    yblock_width = syy;
  }
  else {
    yblock_width = nl.y - syy*blockIdx.y;
  }
    
  for(int j = threadIdx.y; j < yblock_width; j += blockDim.y) {
    internal = ((blockIdx.y*syy+j) < nl.y) && (k < nl.z);
    if (internal) {
      id = getLinearIdx(i+halo.x, blockIdx.y*syy + j + halo.y, k+halo.z, ng);
      sj = j + HALO;
      s_f[sj][sk] = f[id];
    }
  }
  
  const float cz[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};
  
  __syncthreads();

  
  sj = threadIdx.y + HALO;
  int y = syy*blockIdx.y + threadIdx.y;
  // fill in periodic images in shared memory array 
  if (threadIdx.y < HALO) {
    if (halo.y == 0) {
      lid = y%nl.y-HALO;
      if (lid<0) lid+=nl.y;
      id = getLinearIdx(i+halo.x, lid, k+halo.z, ng);
      s_f[sj-HALO][sk] = f[id];
      rid = (y+yblock_width)%nl.y;
      id = getLinearIdx(i+halo.x, rid, k+halo.z, ng);
      s_f[sj+yblock_width][sk] = f[id];
    } else {
      id = getLinearIdx(i+halo.x, y, k+halo.z, ng);
      s_f[sj-HALO][sk] = f[id];
      id = getLinearIdx(i+halo.x, y+yblock_width+halo.y, k+halo.z, ng);
      s_f[sj+yblock_width][sk] = f[id];
    }
  }

  __syncthreads();
  
  ScalarType result;
  for(int j = threadIdx.y; j < yblock_width; j += blockDim.y) {
    internal = ((blockIdx.y*syy+j) < nl.y) && (k < nl.z);
    result = 0;
    if (internal) {
      int id = getLinearIdx(i, blockIdx.y*syy + j ,k, nl);
      int sj = j + HALO;
      for( int l=0; l<HALO; l++) {
        result += cz[l] * ( s_f[sj+1+l][sk] - s_f[sj-1-l][sk]);
      }
      dfy[id] += result*ih;
    }
  }
}



/**********************************************************************************
 * @brief compute x-component of gradient using 8th order finite differencing
 * @param[in]    f = scalar field with ghost layer padding
 * @param[out] dfx = x-component of gradient
**********************************************************************************/
__global__ void mgpu_gradient_x(ScalarType* dfx, const ScalarType* f, int3 nl, int3 ng, int3 halo, const float ih) {
  __shared__ float s_f[syy+2*HALO][sxx]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int j  = blockIdx.z;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x;       // local k for shared memory ac3ess, fixed
    
  int id, lid, rid, si;
  int xblock_width;
  bool internal;
    
  if ( blockIdx.y < gridDim.y - 1) {
    xblock_width = syy;
  }
  else {
    xblock_width = nl.x - syy*blockIdx.y;
  }
  
  for(int i = threadIdx.y; i < xblock_width; i += blockDim.y) {
    internal = ((blockIdx.y*syy + i) < nl.x) && (k < nl.z);
    if (internal) {
      id = getLinearIdx(blockIdx.y*syy + i + halo.x, j + halo.y, k + halo.z, ng);
      si = i + HALO;
      s_f[si][sk] = f[id];
    }
  }
  
  const float cz[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};

  __syncthreads();

  
  si = threadIdx.y + HALO;
  int x = syy*blockIdx.y + threadIdx.y;
  // fill in periodic images in shared memory array 
  if (threadIdx.y < HALO) {
    if (halo.x == 0) {
      lid = x%nl.x-HALO;
      if (lid<0) lid+=nl.x;
      id = getLinearIdx(lid, j+halo.y, k+halo.z, ng);
      s_f[si-HALO][sk] = f[id];
      rid = (x+xblock_width)%nl.x;
      id = getLinearIdx(rid, j+halo.y, k+halo.z, ng);
      s_f[si+xblock_width][sk] = f[id];
    } else {
      id = getLinearIdx(x, j+halo.y, k+halo.z, ng);
      s_f[si-HALO][sk] = f[id];
      id = getLinearIdx(x+xblock_width+halo.x, j+halo.y, k+halo.z, ng);
      s_f[si+xblock_width][sk] = f[id];
    }
  }

  __syncthreads();

  ScalarType result;
  for(int i = threadIdx.y; i < xblock_width; i += blockDim.y) {
    internal = ((blockIdx.y*syy + i) < nl.x) && (k < nl.z);
    result = 0;
    if (internal) {
      id = getLinearIdx(blockIdx.y*syy + i , j, k, nl);
      si = i + HALO;
      for( int l=0; l<HALO; l++) {
          result +=  cz[l] * ( s_f[si+1+l][sk] - s_f[si-1-l][sk]);
      }
      dfx[id] += result*ih;
    }
  }
}

/**********************************************************************************
 * @brief compute laplacian using 8th order finite differencing
 * @param[out]   ddf  = laplacian of scalar field f 
 * @param[in]    f    = scalar field f with ghost padding
 * @param[in]    beta = some constant which needs to be defined TODO
**********************************************************************************/
__global__ void mgpu_d_zz(ScalarType* ddf, const ScalarType* f, const ScalarType beta, int3 nl, int3 ng, int3 halo, const float ih2) {
  __shared__ float s_f[sx][sy+2*HALO]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int i = blockIdx.z;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x + HALO;       // local k for shared memory ac3ess + halo offset
  int sj = threadIdx.y; // local j for shared memory ac3ess
  int zblock_width;
  int id,lid,rid;
  
  if (blockIdx.x < gridDim.x - 1) {
    zblock_width = blockDim.x;
  }
  else {
    zblock_width = nl.z - blockIdx.x*blockDim.x;
  }
  
  bool internal = (j < nl.y) && (threadIdx.x < zblock_width);
  
  if (internal) {
    id = getLinearIdx(i+halo.x,j+halo.y,k+halo.z, ng);
    s_f[sj][sk] = f[id];
  }
  
  const float cxx[HALO+1] = {-205.f/72.f, 8.f / 5.f , -1.f / 5.f , 8.f / 315.f, -1.f / 560.f};
  
  __syncthreads();
    
  // fill in periodic images in shared memory array 
  if (threadIdx.x < HALO) {
    if (halo.z == 0) {
      lid = k%nl.z-HALO;
      if (lid<0) lid+=nl.z;
      id = getLinearIdx(i+halo.x, j+halo.y, lid, ng);
      s_f[sj][sk-HALO] = f[id];
      rid = (k+zblock_width)%nl.z;
      id = getLinearIdx(i+halo.x, j+halo.y, rid, ng);
      s_f[sj][zblock_width+sk] = f[id];
    } else {
      id = getLinearIdx(i+halo.x, j+halo.y, k, ng);
      s_f[sj][sk-HALO] = f[id];
      id = getLinearIdx(i+halo.x, j+halo.y, k+zblock_width+halo.z, ng);
      s_f[sj][zblock_width+sk] = f[id];
    }
  }
  
  __syncthreads();
  
  ScalarType lval = cxx[0]*s_f[sj][sk];
  if (internal) {
    id = getLinearIdx(i,j,k, nl);
    for(int l=0; l<HALO; l++) {
      lval += cxx[l] * (s_f[sj][sk+l] + s_f[sj][sk-l]);
    }
    ddf[id] = lval*ih2;
  }
}

/**********************************************************************************
 * @brief compute laplacian using 8th order finite differencing
 * @param[inout] ddf  = partial laplacian of f
 * @param[in]    f    = scalar field f with ghost padding
 * @param[in]    beta = some constant which needs to be defined TODO
**********************************************************************************/
__global__ void mgpu_d_yy(ScalarType* ddf, const ScalarType* f, const ScalarType beta, int3 nl, int3 ng, int3 halo, const float ih2) {
  __shared__ float s_f[syy+2*HALO][sxx]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int i  = blockIdx.z;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x;       // local k for shared memory ac3ess, fixed
  int yblock_width, id, lid, rid, sj;
  bool internal;
  
  if ( blockIdx.y < gridDim.y - 1) {
    yblock_width = syy;
  }
  else {
    yblock_width = nl.y - syy*blockIdx.y;
  }
    
  for(int j = threadIdx.y; j < yblock_width; j += blockDim.y) {
    internal = ((blockIdx.y*syy+j) < nl.y) && (k < nl.z);
    if (internal) {
      id = getLinearIdx(i+halo.x, blockIdx.y*syy + j + halo.y, k+halo.z, ng);
      sj = j + HALO;
      s_f[sj][sk] = f[id];
    }
  }
  
   const float cxx[HALO+1] = {-205.f/72.f, 8.f / 5.f , -1.f / 5.f , 8.f / 315.f, -1.f / 560.f};
  
  __syncthreads();

  
  sj = threadIdx.y + HALO;
  int y = syy*blockIdx.y + threadIdx.y;
  // fill in periodic images in shared memory array 
  if (threadIdx.y < HALO) {
    if (halo.y == 0) {
      lid = y%nl.y-HALO;
      if (lid<0) lid+=nl.y;
      id = getLinearIdx(i+halo.x, lid, k+halo.z, ng);
      s_f[sj-HALO][sk] = f[id];
      rid = (y+yblock_width)%nl.y;
      id = getLinearIdx(i+halo.x, rid, k+halo.z, ng);
      s_f[sj+yblock_width][sk] = f[id];
    } else {
      id = getLinearIdx(i+halo.x, y, k+halo.z, ng);
      s_f[sj-HALO][sk] = f[id];
      id = getLinearIdx(i+halo.x, y+yblock_width+halo.y, k+halo.z, ng);
      s_f[sj+yblock_width][sk] = f[id];
    }
  }

  __syncthreads();
  
  for(int j = threadIdx.y; j < yblock_width; j += blockDim.y) {
    internal = ((blockIdx.y*syy+j) < nl.y) && (k < nl.z);
    if (internal) {
      ScalarType lval = cxx[0]*s_f[sj][sk];
      int id = getLinearIdx(i, blockIdx.y*syy + j ,k, nl);
      int sj = j + HALO;
      for( int l=0; l<HALO; l++) {
        lval += cxx[l] * ( s_f[sj+l][sk] + s_f[sj-l][sk]);
      }
      ddf[id] += lval*ih2;
    }
  }
}

/**********************************************************************************
 * @brief compute laplacian using 8th order finite differencing
 * @param[inout] ddf  = parital laplacian of scalar field f
 * @param[in]    f    = scalar field f with ghost padding
 * @param[in]    beta = some constant which needs to be defined TODO
**********************************************************************************/
__global__ void mgpu_d_xx(ScalarType* ddf, const ScalarType* f, const ScalarType beta, int3 nl, int3 ng, int3 halo, const float ih2) {
  __shared__ float s_f[syy+2*HALO][sxx]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int j  = blockIdx.z;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x;       // local k for shared memory ac3ess, fixed
    
  int id, lid, rid, si;
  int xblock_width;
  bool internal;
    
  if ( blockIdx.y < gridDim.y - 1) {
    xblock_width = syy;
  }
  else {
    xblock_width = nl.x - syy*blockIdx.y;
  }
  
  for(int i = threadIdx.y; i < xblock_width; i += blockDim.y) {
    internal = ((blockIdx.y*syy + i) < nl.x) && (k < nl.z);
    if (internal) {
      id = getLinearIdx(blockIdx.y*syy + i + halo.x, j + halo.y, k + halo.z, ng);
      si = i + HALO;
      s_f[si][sk] = f[id];
    }
  }
  
   const float cxx[HALO+1] = {-205.f/72.f, 8.f / 5.f , -1.f / 5.f , 8.f / 315.f, -1.f / 560.f};

  __syncthreads();

  
  si = threadIdx.y + HALO;
  int x = syy*blockIdx.y + threadIdx.y;
  // fill in periodic images in shared memory array 
  if (threadIdx.y < HALO) {
    if (halo.x == 0) {
      lid = x%nl.x-HALO;
      if (lid<0) lid+=nl.x;
      id = getLinearIdx(lid, j+halo.y, k+halo.z, ng);
      s_f[si-HALO][sk] = f[id];
      rid = (x+xblock_width)%nl.x;
      id = getLinearIdx(rid, j+halo.y, k+halo.z, ng);
      s_f[si+xblock_width][sk] = f[id];
    } else {
      id = getLinearIdx(x, j+halo.y, k+halo.z, ng);
      s_f[si-HALO][sk] = f[id];
      id = getLinearIdx(x+xblock_width+halo.x, j+halo.y, k+halo.z, ng);
      s_f[si+xblock_width][sk] = f[id];
    }
  }

  __syncthreads();

  for(int i = threadIdx.y; i < xblock_width; i += blockDim.y) {
    internal = ((blockIdx.y*syy + i) < nl.x) && (k < nl.z);
    if (internal) {
      id = getLinearIdx(blockIdx.y*syy + i , j, k, nl);
      si = i + HALO;
      ScalarType lval = cxx[0]*s_f[si][sk];
      for( int l=0; l<HALO; l++) {
          lval += cxx[l] * ( s_f[si+l][sk] + s_f[si-l][sk]);
      }
      ddf[id] += lval*ih2;
      ddf[id] *= beta;
    }
  }
}

inline __device__ void add_op (ScalarType& a, ScalarType b) { a += b; }
inline __device__ void replace_op (ScalarType& a, ScalarType b) { a = b; }

/**************************************************************************************************
 * @brief compute z-component of gradient using 8th order finite differencing (single GPU version)
 * @param[in]    f = scalar field with no ghost layer padding
 * @param[out] dfz = z-component of gradient
**************************************************************************************************/
template<void(*Op)(ScalarType&,ScalarType)=add_op>
__global__ void gradient_z(ScalarType* dfz, const ScalarType* f, const int3 nl, const float ih) {
  __shared__ float s_f[sx][sy+2*HALO]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int i = blockIdx.z;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x + HALO;       // local k for shared memory ac3ess + halo offset
  int sj = threadIdx.y; // local j for shared memory ac3ess
  int zblock_width, id;
  
  const float cz[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};
  
  if (blockIdx.x < gridDim.x - 1) {
    zblock_width = blockDim.x;
  }
  else {
    zblock_width = nl.z - blockIdx.x*blockDim.x;
  }
  
  bool internal = (j < nl.y) && (threadIdx.x < zblock_width);
  
  if (internal) {
    id = getLinearIdx(i,j,k,nl);
    s_f[sj][sk] = f[id];
  }

  __syncthreads();
    
  int lid,rid;
  // fill in periodic images in shared memory array 
  if (threadIdx.x < HALO) {
    lid = k%nl.z-HALO;
    if (lid<0) lid+=nl.z;
    s_f[sj][sk-HALO] = f[i*nl.y*nl.z + j*nl.z + lid];
    rid = (k+zblock_width)%nl.z;
    s_f[sj][zblock_width+sk] = f[i*nl.y*nl.z + j*nl.z + rid];
  }

  __syncthreads();
  
  ScalarType result = 0;
  if (internal) {
    for(int l=0; l<HALO; l++) {
        result += cz[l] * (s_f[sj][sk+1+l] - s_f[sj][sk-1-l]);
    }
    Op(dfz[id], result*ih);
  }
}


/**************************************************************************************************
 * @brief compute y-component of gradient using 8th order finite differencing (single GPU version)
 * @param[in]    f = scalar field with no ghost layer padding
 * @param[out] dfy = y-component of gradient
**************************************************************************************************/
template<void(*Op)(ScalarType&,ScalarType)=add_op>
__global__ void gradient_y(ScalarType* dfy, const ScalarType* f, const int3 nl, const float ih) {
  __shared__ float s_f[syy+2*HALO][sxx]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int i  = blockIdx.z;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x;       // local k for shared memory ac3ess, fixed
    
  int yblock_width, globalIdx, sj;
  bool internal;
  
  const float cy[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};
  
  if ( blockIdx.y < gridDim.y - 1) {
    yblock_width = syy;
  }
  else {
    yblock_width = nl.y - syy*blockIdx.y;
  }
  
    
  for(int j = threadIdx.y; j < yblock_width; j += blockDim.y) {
    internal = ((blockIdx.y*syy+j) < nl.y) && (k < nl.z);
    if (internal) {
        globalIdx = getLinearIdx(i, blockIdx.y*syy + j ,k, nl);
        sj = j + HALO;
        s_f[sj][sk] = f[globalIdx];
    }
  }

  __syncthreads();

  
  int lid,rid;
  sj = threadIdx.y + HALO;
  int y = syy*blockIdx.y + threadIdx.y;
  // fill in periodic images in shared memory array 
  if (threadIdx.y < HALO) {
    lid = y%nl.y-HALO;
    if (lid<0) lid+=nl.y;
    s_f[sj-HALO][sk] = f[i*nl.y*nl.z + lid*nl.z + k];
    rid = (y+yblock_width)%nl.y;
    s_f[sj+yblock_width][sk] = f[i*nl.y*nl.z + rid*nl.z + k];
  }

  __syncthreads();
    
  
  ScalarType result;
  for(int j = threadIdx.y; j < yblock_width; j += blockDim.y) {
    result = 0;
    internal = ((blockIdx.y*syy+j) < nl.y) && (k < nl.z);
    if (internal) {
      globalIdx = getLinearIdx(i, blockIdx.y*syy + j ,k, nl);
      sj = j + HALO;
      for( int l=0; l<HALO; l++) {
          result +=  cy[l] * ( s_f[sj+1+l][sk] - s_f[sj-1-l][sk]);
      }
      Op(dfy[globalIdx], result*ih);
    }
  }
}

/**************************************************************************************************
 * @brief compute x-component of gradient using 8th order finite differencing (single GPU version)
 * @param[in]    f = scalar field with no ghost layer padding
 * @param[out] dfz = x-component of gradient
**************************************************************************************************/
template<void(*Op)(ScalarType&,ScalarType)=add_op>
__global__ void gradient_x(ScalarType* dfx, const ScalarType* f, const int3 nl, const float ih) {
  __shared__ float s_f[syy+2*HALO][sxx]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int j  = blockIdx.z;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x;       // local k for shared memory ac3ess, fixed
    
  int xblock_width, globalIdx, si;
  bool internal;
  
  const float cx[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};
  
  if ( blockIdx.y < gridDim.y - 1) {
    xblock_width = syy;
  }
  else {
    xblock_width = nl.x - syy*blockIdx.y;
  }
    
  for(int i = threadIdx.y; i < xblock_width; i += blockDim.y) {
    internal = ((blockIdx.y*syy + i) < nl.x) && (k < nl.z);
    if (internal) {
        globalIdx = getLinearIdx(blockIdx.y*syy + i, j ,k, nl);
        si = i + HALO;
        s_f[si][sk] = f[globalIdx];
    }
  }

  __syncthreads();

  
  int lid,rid;
  si = threadIdx.y + HALO;
  int x = syy*blockIdx.y + threadIdx.y;
  // fill in periodic images in shared memory array 
  if (threadIdx.y < HALO) {
    lid = x%nl.x-HALO;
    if (lid<0) lid+=nl.x;
    s_f[si-HALO][sk] = f[lid*nl.y*nl.z + j*nl.z + k];
    rid = (x+xblock_width)%nl.x;
    s_f[si+xblock_width][sk] = f[rid*nl.y*nl.z + j*nl.z + k];
  }

  __syncthreads();
    
  
  for(int i = threadIdx.y; i < syy; i += blockDim.y) {
    ScalarType result = 0;
    internal = ((blockIdx.y*syy + i) < nl.x) && (k < nl.z);
    if (internal) {
      int globalIdx = getLinearIdx(blockIdx.y*syy + i , j, k, nl);
      int si = i + HALO;
      for( int l=0; l<HALO; l++) {
        result += cx[l] * ( s_f[si+1+l][sk] - s_f[si-1-l][sk]);
      }
      Op(dfx[globalIdx],result*ih);
    } 
  }
}

/**********************************************************************************
 * @brief compute laplacian using 8th order finite differencing (single GPU code)
 * @param[inout] dfz  = parital laplacian of scalar field f
 * @param[in]    f    = scalar field f
 * @param[in]    beta = some constant which needs to be defined TODO
**********************************************************************************/
__global__ void d_zz(ScalarType* ddf, const ScalarType* f, const ScalarType beta, int3 nl, const float ih2) {
  __shared__ float s_f[sx][sy+2*HALO]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int i   = blockIdx.z;
  int j   = blockIdx.y*blockDim.y + threadIdx.y;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x + HALO;       // local k for shared memory ac3ess + halo offset
  int sj = threadIdx.y; // local j for shared memory ac3ess

  int globalIdx = getLinearIdx(i,j,k, nl);

  s_f[sj][sk] = f[globalIdx];
  
   const float cxx[HALO+1] = {-205.f/72.f, 8.f / 5.f , -1.f / 5.f , 8.f / 315.f, -1.f / 560.f};

  __syncthreads();
    
  int lid,rid;
  // fill in periodic images in shared memory array 
  if (threadIdx.x < HALO) {
    lid = k%nl.z-HALO;
    if (lid<0) lid+=nl.z;
    s_f[sj][sk-HALO] = f[i*nl.y*nl.z + j*nl.z + lid];
    rid = (k+sy)%nl.z;
    s_f[sj][sy+sk] = f[i*nl.y*nl.z + j*nl.z + rid];
  }

  __syncthreads();

    ScalarType lval = cxx[0]*s_f[sj][sk];
    for(int l=1; l<=HALO; l++) {
      lval += cxx[l] * (s_f[sj][sk+l] + s_f[sj][sk-l]);
    }
    ddf[globalIdx] = lval*ih2;
}

/**********************************************************************************
 * @brief compute laplacian using 8th order finite differencing (single GPU code)
 * @param[inout] dfy  = parital laplacian of scheme field f 
 * @param[in]    f    = scalar field f
 * @param[in]    beta = some constant which needs to be defined TODO
**********************************************************************************/
__global__ void d_yy(ScalarType* ddf, const ScalarType* f, const ScalarType beta, int3 nl, const float ih2) {
  __shared__ float s_f[syy+2*HALO][sxx]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int i  = blockIdx.z;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x;       // local k for shared memory ac3ess, fixed
    
    
  for(int j = threadIdx.y; j < syy; j += blockDim.y) {
    int globalIdx = getLinearIdx(i, blockIdx.y*syy + j ,k,nl);
    int sj = j + HALO;
    s_f[sj][sk] = f[globalIdx];
  }
  
   const float cxx[HALO+1] = {-205.f/72.f, 8.f / 5.f , -1.f / 5.f , 8.f / 315.f, -1.f / 560.f};

  __syncthreads();

  
  int lid,rid, sj = threadIdx.y + HALO;
  int y = syy*blockIdx.y + threadIdx.y;
  // fill in periodic images in shared memory array 
  if (threadIdx.y < HALO) {
    lid = y%nl.y-HALO;
    if (lid<0) lid+=nl.y;
    s_f[sj-HALO][sk] = f[i*nl.y*nl.z + lid*nl.z + k];
    rid = (y+syy)%nl.y;
    s_f[sj+syy][sk] = f[i*nl.y*nl.z + rid*nl.z + k];
  }

  __syncthreads();
  
  for(int j = threadIdx.y; j < syy; j += blockDim.y) {
    int globalIdx = getLinearIdx(i, blockIdx.y*syy + j ,k, nl);
    int sj = j + HALO;
    ScalarType lval =cxx[0]*s_f[sj][sk];
    for( int l=1; l<=HALO; l++) {
        lval += cxx[l] * ( s_f[sj+l][sk] + s_f[sj-l][sk]);
    }
    ddf[globalIdx] += lval*ih2;
  }

}

/**********************************************************************************
 * @brief compute laplacian using 8th order finite differencing (single GPU code)
 * @param[inout] ddf  = partial laplacian of scalar field f
 * @param[in]    f    = scalar field f
 * @param[in]    beta = some constant which needs to be defined TODO
**********************************************************************************/
__global__ void d_xx(ScalarType* ddf, const ScalarType* f, const ScalarType beta, int3 nl, const float ih2) {
  __shared__ float s_f[syy+2*HALO][sxx]; // HALO-wide halo for central diferencing scheme
    
  // note i and k have been exchanged to ac3ount for k being the fastest changing index
  int j  = blockIdx.z;
  int k  = blockIdx.x*blockDim.x + threadIdx.x;
  int sk = threadIdx.x;       // local k for shared memory ac3ess, fixed
    
    
  for(int i = threadIdx.y; i < syy; i += blockDim.y) {
    int globalIdx = getLinearIdx(blockIdx.y*syy + i, j ,k, nl);
    int si = i + HALO;
    s_f[si][sk] = f[globalIdx];
  }
  
   const float cxx[HALO+1] = {-205.f/72.f, 8.f / 5.f , -1.f / 5.f , 8.f / 315.f, -1.f / 560.f};

  __syncthreads();

  
  int lid,rid, si = threadIdx.y + HALO;
  int x = syy*blockIdx.y + threadIdx.y;
  // fill in periodic images in shared memory array 
  if (threadIdx.y < HALO) {
    lid = x%nl.x-HALO;
    if (lid<0) lid+=nl.x;
    s_f[si-HALO][sk] = f[lid*nl.y*nl.z + j*nl.z + k];
    rid = (x+syy)%nl.x;
    s_f[si+syy][sk] = f[rid*nl.y*nl.z + j*nl.z + k];
  }

  __syncthreads();
  
  for(int i = threadIdx.y; i < syy; i += blockDim.y) {
    int globalIdx = getLinearIdx(blockIdx.y*syy + i , j, k, nl);
    int si = i + HALO;
    ScalarType lval = cxx[0]*s_f[si][sk];
    for( int l=1; l<=HALO; l++) {
        lval += cxx[l] * ( s_f[si+l][sk] + s_f[si-l][sk]);
    }
    ddf[globalIdx] += lval*ih2;
    ddf[globalIdx] *= beta;
  }
}

__global__ void TextureDivXComputeKernel(cudaTextureObject_t tex, ScalarType* div, int3 nl, const float3 inx, const float ih) {
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const int tidz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tidx < nl.x && tidy < nl.y && tidz < nl.z) {    
      // global index
      const int gid = tidz + tidy*nl.z + tidx*nl.y*nl.z;
      float3 id = make_float3( tidz*inx.z, tidy*inx.y, tidx*inx.x);
      
      const float cx[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};
      
      float dfx=0;
      for(int l=1; l<HALO+1; l++) {
          dfx += (tex3D<float>(tex, id.x, id.y, id.z + l*inx.x) - tex3D<float>(tex, id.x, id.y, id.z - l*inx.x))*cx[l-1];
      }
      div[gid] = dfx*ih;
    }
}

__global__ void TextureDivYComputeKernel(cudaTextureObject_t tex, ScalarType* div, int3 nl, const float3 inx, const float ih) {
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const int tidz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tidx < nl.x && tidy < nl.y && tidz < nl.z) {    
      // global index
      const int gid = tidz + tidy*nl.z + tidx*nl.y*nl.z;
      float3 id = make_float3( tidz*inx.z, tidy*inx.y, tidx*inx.x);
      
      const float cx[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};
      
      float dfy=0;
      for(int l=1; l<HALO+1; l++) {
          dfy += (tex3D<float>(tex, id.x, id.y + l*inx.y, id.z) - tex3D<float>(tex, id.x, id.y - l*inx.y, id.z))*cx[l-1];
      }
      div[gid] += dfy*ih;
    }
}

__global__ void TextureDivZComputeKernel(cudaTextureObject_t tex, ScalarType* div, int3 nl, const float3 inx, const float ih) {
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const int tidz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tidx < nl.x && tidy < nl.y && tidz < nl.z) {    
      // global index
      const int gid = tidz + tidy*nl.z + tidx*nl.y*nl.z;
      float3 id = make_float3( tidz*inx.z, tidy*inx.y, tidx*inx.x);
      
      const float cx[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};
      
      float dfz=0;
      for(int l=1; l<HALO+1; l++) {
          dfz += (tex3D<float>(tex, id.x + l*inx.z, id.y, id.z) - tex3D<float>(tex, id.x - l*inx.z, id.y, id.z))*cx[l-1];
      }
      div[gid] += dfz*ih;
    }
}

__global__ void TextureGradientComputeKernel(cudaTextureObject_t tex, ScalarType* dmx, ScalarType* dmy, ScalarType* dmz, int3 nl, const float3 inx, const float3 ih) {
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (tidx < nl.x && tidy < nl.y && tidz < nl.z) {
      // global index
      const int gid = tidz + tidy*nl.z + tidx*nl.y*nl.z;
      float3 id = make_float3( tidz*inx.z, tidy*inx.y, tidx*inx.x);
      
      const float cx[HALO] = {4.f / 5.f , -1.f / 5.f , 4.f / 105.f, -1.f / 280.f};
    
      float dfx=0,dfy=0,dfz=0;
      for(int l=1; l<HALO+1; l++) {
          dfz += (tex3D<float>(tex, id.x + l*inx.z, id.y, id.z) - tex3D<float>(tex, id.x - l*inx.z, id.y, id.z))*cx[l-1];
          dfy += (tex3D<float>(tex, id.x, id.y + l*inx.y, id.z) - tex3D<float>(tex, id.x, id.y - l*inx.y, id.z))*cx[l-1];
          dfx += (tex3D<float>(tex, id.x, id.y, id.z + l*inx.x) - tex3D<float>(tex, id.x, id.y, id.z - l*inx.x))*cx[l-1];
      }
      dmz[gid] = dfz*ih.x;
      dmy[gid] = dfy*ih.y;
      dmx[gid] = dfx*ih.z;
    }
}

void printFloat3(float3 a){
    printf("x = %f\t y = %f\t z = %f\n",a.x,a.y,a.z);
}

namespace reg {

cudaTextureObject_t gpuInitEmptyGradientTexture(IntType *nx) {
   cudaTextureObject_t texObj = 0;
#if defined(USE_TEXTURE_GRADIENT)
   cudaError_t err = cudaSuccess;
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
   cudaExtent extent = make_cudaExtent(nx[2], nx[1], nx[0]);
   cudaArray* cuArray;
   err = cudaMalloc3DArray(&cuArray, &channelDesc, extent, 0);
   if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate 3D cudaArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
   }
    
    /* create cuda resource description */
    struct cudaResourceDesc resDesc;
    memset( &resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[2] = cudaAddressModeWrap;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 1;

    err = cudaCreateTextureObject( &texObj, &resDesc, &texDesc, NULL);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to create texture (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif
    return texObj;
}


/********************************************************************************
 * @brief update texture object by copying volume data to 3D cudaArray container
 *******************************************************************************/
void updateTextureFromVolume(cudaPitchedPtr volume, cudaExtent extent, cudaTextureObject_t texObj) {
    cudaError_t err = cudaSuccess;

    /* create cuda resource description */
    struct cudaResourceDesc resDesc;
    memset( &resDesc, 0, sizeof(resDesc));
    cudaGetTextureObjectResourceDesc( &resDesc, texObj);

    cudaMemcpy3DParms p = {0};
    p.srcPtr = volume;
    p.dstArray = resDesc.res.array.array;
    p.extent = extent;
    p.kind = cudaMemcpyDeviceToDevice;
    err = cudaMemcpy3D(&p);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy 3D memory to cudaArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

PetscErrorCode initConstants(const IntType* iisize, const IntType* iisize_g, const ScalarType* hx, const IntType* ihalo) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  /*int halo[3], isize[3], isize_g[3];
  halo[0] = ihalo[0]; halo[1] = ihalo[1]; halo[2] = ihalo[2];
  isize[0] = iisize[0]; isize[1] = iisize[1]; isize[2] = iisize[2];
  isize_g[0] = iisize_g[0]; isize_g[1] = iisize_g[1]; isize_g[2] = iisize_g[2];
  
  float3 inv_nx = make_float3(  1.0f/static_cast<float>(isize[0]),
                                1.0f/static_cast<float>(isize[1]), 
                                1.0f/static_cast<float>(isize[2]));
  float3 inv_hx = make_float3(0.5f/hx[0], 0.5f/hx[1], 0.5f/hx[2]);

  //cudaMemcpyToSymbol(halo.x, &halo[0], sizeof(int), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(halo.y, &halo[1], sizeof(int), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(halo.z, &halo[2], sizeof(int), 0, cudaMemcpyHostToDevice);

  //cudaMemcpyToSymbol(nl.x, &isize[0], sizeof(int), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(nl.y, &isize[1], sizeof(int), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(nl.z, &isize[2], sizeof(int), 0, cudaMemcpyHostToDevice);
  
  //cudaMemcpyToSymbol(nl.x, &isize[0], sizeof(int), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(nl.y, &isize[1], sizeof(int), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(nl.z, &isize[2], sizeof(int), 0, cudaMemcpyHostToDevice);
  
  //cudaMemcpyToSymbol(ng.x, &isize_g[0], sizeof(int), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(ng.y, &isize_g[1], sizeof(int), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(ng.z, &isize_g[2], sizeof(int), 0, cudaMemcpyHostToDevice);
  
  //cudaMemcpyToSymbol(inx.x, &inv_nx.x, sizeof(float), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(inx.y, &inv_nx.y, sizeof(float), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(inx.z, &inv_nx.z, sizeof(float), 0, cudaMemcpyHostToDevice);

  //cudaMemcpyToSymbol(d_invhx, &inv_hx.x, sizeof(float), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(d_invhy, &inv_hx.y, sizeof(float), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(d_invhz, &inv_hx.z, sizeof(float), 0, cudaMemcpyHostToDevice);
  
  float h_ct[HALO+1];
  for(int l=0; l<HALO; l++) h_ct[l] = h_c[l]/hx[0];
  //cudaMemcpyToSymbol(d_cx, h_ct, sizeof(float)*HALO, 0, cudaMemcpyHostToDevice);
  for(int l=0; l<HALO; l++) h_ct[l] = h_c[l]/hx[1];
  //cudaMemcpyToSymbol(d_cy, h_ct, sizeof(float)*HALO, 0, cudaMemcpyHostToDevice);
  for(int l=0; l<HALO; l++) h_ct[l] = h_c[l]/hx[2];
  //cudaMemcpyToSymbol(d_cz, h_ct, sizeof(float)*HALO, 0, cudaMemcpyHostToDevice);
  
  for(int l=0; l<=HALO; l++) h_ct[l] = h2_c[l]/(hx[0]*hx[0]);
  //cudaMemcpyToSymbol(d_cxx, h_ct, sizeof(float)*(HALO+1), 0, cudaMemcpyHostToDevice);
  for(int l=0; l<=HALO; l++) h_ct[l] = h2_c[l]/(hx[1]*hx[1]);
  //cudaMemcpyToSymbol(d_cyy, h_ct, sizeof(float)*(HALO+1), 0, cudaMemcpyHostToDevice);
  for(int l=0; l<=HALO; l++) h_ct[l] = h2_c[l]/(hx[2]*hx[2]);
  //cudaMemcpyToSymbol(d_czz, h_ct, sizeof(float)*(HALO+1), 0, cudaMemcpyHostToDevice);
  */
  PetscFunctionReturn(ierr);
}

void getThreadBlockDimensionsX(dim3& threads, dim3& blocks, IntType* nx) {
  threads.x = sxx;
  threads.y = syy/perthreadcomp;
  threads.z = 1;
  blocks.x = (nx[2]+sxx-1)/sxx;
  blocks.y = (nx[0]+syy-1)/syy;
  blocks.z = nx[1];
}

void getThreadBlockDimensionsY(dim3& threads, dim3& blocks, IntType* nx) {
  threads.x = sxx;
  threads.y = syy/perthreadcomp;
  threads.z = 1;
  blocks.x = (nx[2]+sxx-1)/sxx;
  blocks.y = (nx[1]+syy-1)/syy;
  blocks.z = nx[0];
}

void getThreadBlockDimensionsZ(dim3& threads, dim3& blocks, IntType* nx) {
  threads.x = sy;
  threads.y = sx;
  threads.z = 1;
  blocks.x = (nx[2]+sy-1)/sy;
  blocks.y = (nx[1]+sx-1)/sx;
  blocks.z = nx[0];
}

PetscErrorCode computeDivergence(ScalarType* l, const ScalarType* g1, const ScalarType* g2, const ScalarType* g3, cudaTextureObject_t mtex, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, bool mgpu) {
 
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  //PUSH_RANGE("CU_Differentiation_ComputeDivergence",2)
  size_t count = sizeof(ScalarType)*nx[0]*nx[1]*nx[2];
  if (mgpu) {
    ierr = cudaMemset((void*)l, 0, count); CHKERRCUDA(ierr);
  }

#if defined(USE_TEXTURE_GRADIENT)
  // create a cudaExtent for input resolution
  cudaExtent extent = make_cudaExtent(nx[2], nx[1], nx[0]);
  
  // Texture gradient
  dim3 threadsPerBlock(1,8,32);
  dim3 numBlocks(nx[0]/threadsPerBlock.x, (nx[1]+7)/threadsPerBlock.y, (nx[2]+31)/threadsPerBlock.z);
  
  cudaPitchedPtr m_cudaPitchedPtr;
  
  // make input image a cudaPitchedPtr for m
  m_cudaPitchedPtr = make_cudaPitchedPtr((void*)(g1), nx[2]*sizeof(ScalarType), nx[2], nx[1]);
  // update texture object
  updateTextureFromVolume(m_cudaPitchedPtr, extent, mtex);

  TextureDivXComputeKernel<<<numBlocks, threadsPerBlock>>>(mtex, l);
  cudaCheckKernelError();
  cudaDeviceSynchronize();
  
  // make input image a cudaPitchedPtr for m
  m_cudaPitchedPtr = make_cudaPitchedPtr((void*)(g2), nx[2]*sizeof(ScalarType), nx[2], nx[1]);
  // update texture object
  updateTextureFromVolume(m_cudaPitchedPtr, extent, mtex);

  TextureDivYComputeKernel<<<numBlocks, threadsPerBlock>>>(mtex, l);
  cudaCheckKernelError();
  cudaDeviceSynchronize();
  
  // make input image a cudaPitchedPtr for m
  m_cudaPitchedPtr = make_cudaPitchedPtr((void*)(g3), nx[2]*sizeof(ScalarType), nx[2], nx[1]);
  // update texture object
  updateTextureFromVolume(m_cudaPitchedPtr, extent, mtex);

  TextureDivZComputeKernel<<<numBlocks, threadsPerBlock>>>(mtex, l);
  cudaCheckKernelError();
  cudaDeviceSynchronize();
#else
  const int3 nl = make_int3(nx[0], nx[1], nx[2]);
  const int3 ng = make_int3(nghost[0], nghost[1], nghost[2]);
  const int3 halo = make_int3(nhalo[0], nhalo[1], nhalo[2]);
  // Shared memory implementation
  // Z-Gradient
  dim3 threadsPerBlock_z, numBlocks_z;
  getThreadBlockDimensionsZ(threadsPerBlock_z, numBlocks_z, nx);
  if (mgpu)
    mgpu_gradient_z<<<numBlocks_z, threadsPerBlock_z>>>(l,g3,nl,ng,halo, 1./hx[2]);
  else
    gradient_z<replace_op><<<numBlocks_z, threadsPerBlock_z>>>(l,g3,nl,1./hx[2]);
  cudaCheckKernelError();
    
  // Y-Gradient 
  dim3 threadsPerBlock_y, numBlocks_y;
  getThreadBlockDimensionsY(threadsPerBlock_y, numBlocks_y, nx);
  if (mgpu)
    mgpu_gradient_y<<<numBlocks_y, threadsPerBlock_y>>>(l, g2,nl,ng,halo, 1./hx[1]);
  else
    gradient_y<<<numBlocks_y, threadsPerBlock_y>>>(l, g2,nl,1./hx[1]);
  cudaCheckKernelError();
    
  // X-Gradient
  dim3 threadsPerBlock_x, numBlocks_x;
  getThreadBlockDimensionsX(threadsPerBlock_x, numBlocks_x, nx);
  if (mgpu)
    mgpu_gradient_x<<<numBlocks_x, threadsPerBlock_x>>>(l, g1,nl,ng,halo, 1./hx[0]);
  else
    gradient_x<<<numBlocks_x, threadsPerBlock_x>>>(l, g1,nl,1./hx[0]);
  cudaCheckKernelError();
  cudaDeviceSynchronize();
#endif
  //POP_RANGE
  PetscFunctionReturn(ierr);

}


PetscErrorCode computeDivergenceZ(ScalarType* l, ScalarType* gz, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, bool mgpu) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  //PUSH_RANGE("CU_Differentiation_ComputeDivergence_Z",2)
  const int3 nl = make_int3(nx[0], nx[1], nx[2]);
  const int3 ng = make_int3(nghost[0], nghost[1], nghost[2]);
  const int3 halo = make_int3(nhalo[0], nhalo[1], nhalo[2]);
  
  dim3 threadsPerBlock_z, numBlocks_z;
  getThreadBlockDimensionsZ(threadsPerBlock_z, numBlocks_z, nx);
  if (mgpu)
    mgpu_gradient_z<<<numBlocks_z, threadsPerBlock_z>>>(l,gz,nl,ng,halo, 1./hx[2]);
  else
    gradient_z<<<numBlocks_z, threadsPerBlock_z>>>(l,gz,nl,1./hx[2]);
  cudaCheckKernelError();
  cudaDeviceSynchronize();
  POP_RANGE    
  PetscFunctionReturn(ierr);
 
}

PetscErrorCode computeDivergenceY(ScalarType* l, ScalarType* gy, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, bool mgpu) {
 
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  //PUSH_RANGE("CU_Differentiation_ComputeDivergence_Y",2)
  const int3 nl = make_int3(nx[0], nx[1], nx[2]);
  const int3 ng = make_int3(nghost[0], nghost[1], nghost[2]);
  const int3 halo = make_int3(nhalo[0], nhalo[1], nhalo[2]);
  
  dim3 threadsPerBlock_y, numBlocks_y;
  getThreadBlockDimensionsY(threadsPerBlock_y, numBlocks_y, nx);
  if (mgpu)
    mgpu_gradient_y<<<numBlocks_y, threadsPerBlock_y>>>(l, gy,nl,ng,halo, 1./hx[1]);
  else
    gradient_y<<<numBlocks_y, threadsPerBlock_y>>>(l, gy, nl, 1./hx[1]);
  cudaCheckKernelError();
  cudaDeviceSynchronize();
  //POP_RANGE
  PetscFunctionReturn(ierr);
  
}


PetscErrorCode computeDivergenceX(ScalarType* l, ScalarType* gx, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, bool mgpu) {
 
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  //PUSH_RANGE("CU_Differentiation_ComputeDivergence_X",2)
  const int3 nl = make_int3(nx[0], nx[1], nx[2]);
  const int3 ng = make_int3(nghost[0], nghost[1], nghost[2]);
  const int3 halo = make_int3(nhalo[0], nhalo[1], nhalo[2]);
  
  dim3 threadsPerBlock_x, numBlocks_x;
  getThreadBlockDimensionsX(threadsPerBlock_x, numBlocks_x, nx);
  if (mgpu)
    mgpu_gradient_x<<<numBlocks_x, threadsPerBlock_x>>>(l, gx,nl,ng,halo, 1./hx[0]);
  else
    gradient_x<<<numBlocks_x, threadsPerBlock_x>>>(l, gx, nl, 1./hx[0]);
  cudaCheckKernelError();
  cudaDeviceSynchronize();
  //POP_RANGE
  PetscFunctionReturn(ierr);
  
}

PetscErrorCode computeGradient(ScalarType* gx, ScalarType* gy, ScalarType* gz, const ScalarType* m, cudaTextureObject_t mtex, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, bool mgpu) {
   
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    //PUSH_RANGE("CU_Differentiation_ComputeGradient",2)
    // set all values to zero first
    size_t count = sizeof(ScalarType)*nx[0]*nx[1]*nx[2];
    if (mgpu) {
      ierr = cudaMemset((void*)gz, 0, count); CHKERRCUDA(ierr);
      ierr = cudaMemset((void*)gy, 0, count); CHKERRCUDA(ierr);
      ierr = cudaMemset((void*)gx, 0, count); CHKERRCUDA(ierr);
    }

#if defined(USE_TEXTURE_GRADIENT)
    // make input image a cudaPitchedPtr for m
    cudaPitchedPtr m_cudaPitchedPtr = make_cudaPitchedPtr((void*)(m), nx[2]*sizeof(ScalarType), nx[2], nx[1]);
    
    // create a cudaExtent for input resolution
    cudaExtent extent = make_cudaExtent(nx[2], nx[1], nx[0]);
    
    // update texture object
    updateTextureFromVolume(m_cudaPitchedPtr, extent, mtex);
   
    // Texture gradient
    dim3 threadsPerBlock(1,8,32);
    dim3 numBlocks(nx[0]/threadsPerBlock.x, (nx[1]+7)/threadsPerBlock.y, (nx[2]+31)/threadsPerBlock.z);
    TextureGradientComputeKernel<<<numBlocks, threadsPerBlock>>>(mtex, gx, gy, gz);
    cudaCheckKernelError();
    cudaDeviceSynchronize();
#else
    const int3 nl = make_int3(nx[0], nx[1], nx[2]);
  const int3 ng = make_int3(nghost[0], nghost[1], nghost[2]);
  const int3 halo = make_int3(nhalo[0], nhalo[1], nhalo[2]);
    
    // Shared Memory implementation of Gradient Kernel
    // Z-Gradient
    dim3 threadsPerBlock_z, numBlocks_z;
    getThreadBlockDimensionsZ(threadsPerBlock_z, numBlocks_z, nx);
    if (mgpu)
      mgpu_gradient_z<<<numBlocks_z, threadsPerBlock_z>>>(gz,m,nl,ng,halo, 1./hx[2]);
    else
      gradient_z<replace_op><<<numBlocks_z, threadsPerBlock_z>>>(gz,m, nl, 1./hx[2]);
    cudaCheckKernelError();
    
    // Y-Gradient 
    dim3 threadsPerBlock_y, numBlocks_y;
    getThreadBlockDimensionsY(threadsPerBlock_y, numBlocks_y, nx);
    if (mgpu)
      mgpu_gradient_y<<<numBlocks_y, threadsPerBlock_y>>>(gy, m,nl,ng,halo, 1./hx[1]);
    else
      gradient_y<replace_op><<<numBlocks_y, threadsPerBlock_y>>>(gy, m, nl, 1./hx[1]);
    cudaCheckKernelError();
    
    // X-Gradient
    dim3 threadsPerBlock_x, numBlocks_x;
    getThreadBlockDimensionsX(threadsPerBlock_x, numBlocks_x, nx);
    if (mgpu)
      mgpu_gradient_x<<<numBlocks_x, threadsPerBlock_x>>>(gx, m,nl,ng,halo, 1./hx[0]);
    else
      gradient_x<replace_op><<<numBlocks_x, threadsPerBlock_x>>>(gx, m, nl, 1./hx[0]);
    cudaCheckKernelError();
    cudaDeviceSynchronize();
#endif
    //POP_RANGE
    PetscFunctionReturn(ierr);
    
}


PetscErrorCode computeLaplacian(ScalarType* ddm, const ScalarType* m, cudaTextureObject_t mtex, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, ScalarType beta, bool mgpu) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    //PUSH_RANGE("CU_Differentiation_ComputeLaplasian",2)
    const int3 nl = make_int3(nx[0], nx[1], nx[2]);
    const int3 ng = make_int3(nghost[0], nghost[1], nghost[2]);
    const int3 halo = make_int3(nhalo[0], nhalo[1], nhalo[2]);
    
    // Z-component
    dim3 threadsPerBlock_z, numBlocks_z;
    getThreadBlockDimensionsZ(threadsPerBlock_z, numBlocks_z, nx);
    if (mgpu)
      mgpu_d_zz<<<numBlocks_z, threadsPerBlock_z>>>(ddm, m, beta,nl,ng,halo, 1./(hx[2]*hx[2]));
    else
      d_zz<<<numBlocks_z, threadsPerBlock_z>>>(ddm, m, beta,nl,1./(hx[2]*hx[2]));
    cudaCheckKernelError();
    
    // Y-component
    dim3 threadsPerBlock_y, numBlocks_y;
    getThreadBlockDimensionsY(threadsPerBlock_y, numBlocks_y, nx);
    if (mgpu)
      mgpu_d_yy<<<numBlocks_y, threadsPerBlock_y>>>(ddm, m, beta,nl,ng,halo, 1./(hx[1]*hx[1]));
    else
      d_yy<<<numBlocks_y, threadsPerBlock_y>>>(ddm, m, beta,nl,1./(hx[1]*hx[1]));
    cudaCheckKernelError();
    
    // X-component
    dim3 threadsPerBlock_x, numBlocks_x;
    getThreadBlockDimensionsX(threadsPerBlock_x, numBlocks_x, nx);
    if (mgpu)
      mgpu_d_xx<<<numBlocks_x, threadsPerBlock_x>>>(ddm, m, beta,nl,ng,halo, 1./(hx[0]*hx[0]));
    else
      d_xx<<<numBlocks_x, threadsPerBlock_x>>>(ddm, m, beta,nl,1./(hx[0]*hx[0]));
    cudaCheckKernelError();
    cudaDeviceSynchronize();
    //POP_RANGE
    PetscFunctionReturn(ierr);

}

}


#endif
