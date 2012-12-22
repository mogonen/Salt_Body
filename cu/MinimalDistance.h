#ifndef _MINIMALDISTANCE_H
#define _MINIMALDISTANCE_H

#include "CudaKDTree.h"

typedef struct MinDistanceContext_c
{
  float3
    m_cOrigo;

  float
    m_maxDistance;

  float
    m_distanceScale,
    m_distanceOffset;

  float3
    m_cExtent0,
    m_cExtent1,
    m_cExtent2;

  int
    m_anMin[3],
    m_anMax[3],
    m_anTotalSize[3],
    m_anOutputPitch[3];

  float
    *m_pOutput;

  int3
    *m_pnTriangles;

  int2
    *m_pnLines;

 CudaBoundingVolumeHierarchy
    m_BVH;

 CudaBoundingBox
   *m_pcBoundingBox;

 float4
   *m_pcFloatV;

} * MindDistanceContext_pc;

#endif