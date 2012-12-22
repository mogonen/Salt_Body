#include "MinimalDistance.h"

using namespace Hue::Util;

texture<int,1,cudaReadModeElementType> texTriangles;
texture<int2,1,cudaReadModeElementType> texLines;
texture<int,1,cudaReadModeElementType> texPrimitives;
texture<float4,1,cudaReadModeElementType> texVertices;


__constant__ MinDistanceContext_c cMC;

__device__ float
distSquared(float3 P0, float3 P1)
{
  float3
    P;
  
  P.x = P1.x - P0.x;
  P.y = P1.y - P0.y;
  P.z = P1.z - P0.z;

  return P.x * P.x + P.y * P.y + P.z * P.z;
}

__device__ 
float3 dsub(float3 cA, float3 cB)
{
  float3
    cR;

  cR.x = cA.x - cB.x;
  cR.y = cA.y - cB.y;
  cR.z = cA.z - cB.z;

  return cR;
}

__device__ 
float3 dadd(float3 cA, float3 cB)
{
  float3
    cR;

  cR.x = cA.x + cB.x;
  cR.y = cA.y + cB.y;
  cR.z = cA.z + cB.z;

  return cR;
}

__device__
float3 dscale(float3 cA, float rS)
{
  cA.x *= rS;
  cA.y *= rS;
  cA.z *= rS;

  return cA;
}

__device__
float3 dnorm(float3 cA)
{
  float
    rLength = sqrtf(cA.x * cA.x + 
                    cA.y * cA.y + 
                    cA.z * cA.z);

  if (rLength > 0)
  {
    rLength = 1.0f / rLength;
  }

  cA = dscale(cA,rLength);
  
  return cA;
}

__device__
float ddot(float3 cA, float3 cB)
{
  return cA.x * cB.x + cA.y * cB.y + cA.z * cB.z;
}

__device__ float
distanceTriangle(const float3 &cP, float3 &cNearestPos, const float4 &c4A, const float4 &c4B, const float4 &c4C)
{
  float3
    cA,
    cB,
    cC;

  cA.x = c4A.x;
  cA.y = c4A.y;
  cA.z = c4A.z;
  cB.x = c4B.x;
  cB.y = c4B.y;
  cB.z = c4B.z;
  cC.x = c4C.x;
  cC.y = c4C.y;
  cC.z = c4C.z;

  float3
    cAB = dsub(cB,cA),
    cAC = dsub(cC,cA),
    cAP = dsub(cP,cA);

  float
    rD1 = ddot(cAB,cAP),
    rD2 = ddot(cAC,cAP);

  if ((rD1 <= 0) && (rD2 <= 0))
  {
    cNearestPos = cA;
    return distSquared(cP, cNearestPos);
  }

  float3
    cBP = dsub(cP,cB);

  float
    rD3 = ddot(cAB,cBP),
    rD4 = ddot(cAC,cBP);


  if ((rD3 >= 0) && (rD4 <= rD3))
  {
    cNearestPos = cB;
    return distSquared(cP, cNearestPos);
  }

  float
    rVC = rD1 * rD4 - rD3 * rD2;

  if ((rVC <= 0) && (rD1 >= 0) && (rD3 <= 0))
  {
    float
      rV = 0;

    if (rD1 != rD3)
    {
      rV = rD1 / (rD1 - rD3);
    }

    cNearestPos = dadd(cA,dscale(cAB,rV));
    return distSquared(cP, cNearestPos);
  }

  float3
    cCP = dsub(cP,cC);

  float
    rD5 = ddot(cAB,cCP),
    rD6 = ddot(cAC,cCP);

  if ((rD6 >= 0) && (rD5 <= rD6))
  {
    cNearestPos = cC;
    return distSquared(cP, cNearestPos);
  }

  float
    rVB = rD5*rD2 - rD1*rD6;

  if ((rVB <= 0) && (rD2 >= 0) && (rD6 <= 0))
  {
    float
      rW = 0;

    if (rD2 != rD6)
    {
      rW = rD2 / (rD2 - rD6);
    }

    cNearestPos = dadd(cA,dscale(cAC,rW));
    return distSquared(cP, cNearestPos);
  }

  float
    rVA = rD3 * rD6 - rD5 * rD4;

  if ((rVA <= 0) && ((rD4 - rD3) >= 0) && ((rD5 - rD6) >= 0))
  {
    float
      rBelow = ((rD4 - rD3) + (rD5 - rD6)),
      rW = 0;

    if (rBelow)
    {
      rW = (rD4 - rD3) / rBelow;
    }

    float3
      cBC = dsub(cC,cB);

    cNearestPos = dadd(cB,dscale(cBC,rW));
    return distSquared(cP, cNearestPos);
  }

  float
    rDenom  = 1.0f / (rVA + rVB + rVC),
    rV      = rVB * rDenom,
    rW      = rVC * rDenom;
  
  cNearestPos = dadd(dadd(cA,dscale(cAB,rV)),dscale(cAC,rW));
  return distSquared(cP, cNearestPos);
}

__device__ float
distanceToLine(float3 P, float4 P40, float4 P41)
{
  float3
    P0,
    P1;

  P0.x = P40.x;
  P0.y = P40.y;
  P0.z = P40.z;
  P1.x = P41.x;
  P1.y = P41.y;
  P1.z = P41.z;

  float3 v = dsub(P1,P0);
  float3 w = dsub(P,P0);

  float c1 = ddot(w,v);

  if (c1 <= 0)
  {
    return distSquared(P, P0);
  }

  float c2 = ddot(v,v);

  if (c2 <= c1)
  {
    return distSquared(P, P1);
  }
  
  float b = c1 / c2;

  float3 Pb = dadd(P0,dscale(v,b));
  return distSquared(P, Pb);
}


__device__ void MinMaxDistanceFromBBToPoint( const float3 &cPos, const CudaBoundingBox &cBB, float &rMinDist0, float &rMaxDist0) 
{
  float
    rDistX = 0.0f,
    rDistY = 0.0f,
    rDistZ = 0.0f;

  if      (cPos.x < cBB.Min.x) rDistX = cBB.Min.x - cPos.x;
  else if (cPos.x > cBB.Max.x) rDistX = cPos.x - cBB.Max.x;
  if      (cPos.y < cBB.Min.y) rDistY = cBB.Min.y - cPos.y;
  else if (cPos.y > cBB.Max.y) rDistY = cPos.y - cBB.Max.y;
  if      (cPos.z < cBB.Min.z) rDistZ = cBB.Min.z - cPos.z;
  else if (cPos.z > cBB.Max.z) rDistZ = cPos.z - cBB.Max.z;

  rMinDist0 = rDistX * rDistX + rDistY * rDistY + rDistZ * rDistZ;

  rDistX = max(cBB.Max.x - cPos.x, cPos.x - cBB.Min.x);
  rDistY = max(cBB.Max.y - cPos.y, cPos.y - cBB.Min.y);
  rDistZ = max(cBB.Max.z - cPos.z, cPos.z - cBB.Min.z);

  rMaxDist0 = rDistX * rDistX + rDistY * rDistY + rDistZ * rDistZ;
}

template <bool isLine>
__device__ void FindClosestDistanceAndPrimitiveID(const float3 &cPos, float &rFinalClosestDist) 
{
  const CudaBoundingVolumeHierarchy::NodeEncoding * __restrict__
    pcNodes = (CudaBoundingVolumeHierarchy::NodeEncoding *)cMC.m_BVH.m_pcNodes;
  
  int
    anNode[32];
  
  const CudaBoundingBox
    *pcBB = cMC.m_pcBoundingBox;

 CudaBoundingVolumeHierarchy::NodeEncoding
    cNode = pcNodes[0];

  int 
    nNode = 0;

  bool
    isCurrentNodeSet = true;

  float
    rClosestDist = rFinalClosestDist * rFinalClosestDist;
  
  while (nNode || isCurrentNodeSet)
  {
    if (!isCurrentNodeSet)
    {
      nNode--;
      
      int
        iNode = anNode[nNode];

      cNode = pcNodes[iNode];
    }

    if (cNode.IsLeafNode())
    {
      isCurrentNodeSet = false;

      int
        nPrimitives = cNode.GetPrimitiveCount();

      for(int iPrimitive = 0; iPrimitive < nPrimitives; iPrimitive++)
      {
        int
          iPrimitiveIndex = tex1Dfetch(texPrimitives, cNode.GetPrimitiveOffet() + iPrimitive);
        
        if (isLine)
        {
          int2
            cLine = tex1Dfetch(texLines, iPrimitiveIndex);


          float4
            cA = tex1Dfetch(texVertices, cLine.x),
            cB = tex1Dfetch(texVertices, cLine.y);

          float rLength = distanceToLine(cPos, cA, cB);

          if (rLength < rClosestDist)
          { 
            rClosestDist = rLength;
            rFinalClosestDist = sqrtf(rLength);
          }
        }
        else
        {
          int3
            cTriangle;
          
          cTriangle.x = tex1Dfetch(texTriangles, iPrimitiveIndex * 3 + 0);
          cTriangle.y = tex1Dfetch(texTriangles, iPrimitiveIndex * 3 + 1);
          cTriangle.z = tex1Dfetch(texTriangles, iPrimitiveIndex * 3 + 2);

          float4
            cA = tex1Dfetch(texVertices, cTriangle.x),
            cB = tex1Dfetch(texVertices, cTriangle.y),
            cC = tex1Dfetch(texVertices, cTriangle.z);

          float3
            cNearestPos;

          float 
            rLength = distanceTriangle(cPos, cNearestPos, cA, cB, cC);

          if (rLength <= rClosestDist)
          { 
            cB.x -= cA.x;
            cB.y -= cA.y;
            cB.z -= cA.z;
            cC.x -= cA.x;
            cC.y -= cA.y;
            cC.z -= cA.z;
            
            float3
              cNormal;

            cNormal.x = cB.y * cC.z - cB.z * cC.y;
            cNormal.y = cB.z * cC.x - cB.x * cC.z;
            cNormal.z = cB.x * cC.y - cB.y * cC.x;

            cNormal = dnorm(cNormal);
            cNearestPos = dnorm(dsub(cPos,cNearestPos));
            
            float
              rAngle = ddot(cNormal,cNearestPos);

            // displace front/back unceartainty by 1/1000th (so that it finds the closest face in a deterministic way)
            float
              rStoreLength = rLength;

            rLength *= (-rAngle + 1.0) * 0.001f + 1.0f;

            if (rLength <= rClosestDist)
            {
              rClosestDist = rLength;
              rFinalClosestDist = sqrtf(rStoreLength);
              
              //if (rAngle < 0) rFinalClosestDist *= -1;
              rFinalClosestDist *= rAngle; 
            }
          }
        }
      }
    }
    else
    {
      int
        iNode0 = cNode.GetFirstChild(),
        iNode1 = iNode0 + 1;

      // find dist to cNode0 & cNode1
      CudaBoundingBox
        cBB0 = pcBB[iNode0],
        cBB1 = pcBB[iNode1];

      float
        rMinDist0,
        rMaxDist0,
        rMinDist1,
        rMaxDist1;

      MinMaxDistanceFromBBToPoint(cPos, cBB0, rMinDist0, rMaxDist0);
      MinMaxDistanceFromBBToPoint(cPos, cBB1, rMinDist1, rMaxDist1);

      bool 
        isR0 = (rMinDist0 * 1.00f < rClosestDist),
        isR1 = (rMinDist1 * 1.00f < rClosestDist);

      int
        nValid = isR0 + isR1;
   
      if (!nValid)
      {
        isCurrentNodeSet = false;
        continue;
      }

      isCurrentNodeSet = true;

      if (nValid == 1)
      {
        if (isR0)
        {
          cNode = pcNodes[iNode0];
        }
        else
        {
          cNode = pcNodes[iNode1];
        }
      }
      else
      {
        bool
          isSwap = false;
        
        // swap?
        if (rMinDist1 < rMinDist0)
        {
          isSwap = true;
        }
        else if (rMinDist0 == rMinDist1)
        {
          if (rMaxDist1 < rMaxDist0)
          {
            isSwap = true;
          }
        }

        if (isSwap)
        {
          anNode[nNode++] = iNode0;
          cNode = pcNodes[iNode1];
        }
        else
        {
          anNode[nNode++] = iNode1;
          cNode = pcNodes[iNode0];
        }
      }
    }
  }
}

__global__ void FindMinimalDistance(const int nDim2Modulo, int isLine)
{
  int iDim0 = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;                                                                                    
  int iDim2 = blockIdx.y / nDim2Modulo;                                                                                                       
  int nBlockIdY = blockIdx.y - (iDim2 * nDim2Modulo);                                                                                         
  int iDim1 = __mul24(nBlockIdY,blockDim.y) + threadIdx.y;                                                                                    

  iDim0 += cMC.m_anMin[0];
  iDim1 += cMC.m_anMin[1];
  iDim2 += cMC.m_anMin[2];

  if (iDim0 < cMC.m_anMax[0] &&
      iDim1 < cMC.m_anMax[1] &&
      iDim2 < cMC.m_anMax[2])
  {
      // calculate where voxel is in mesh
    float
      rRelativePos0 = 0,
      rRelativePos1 = 0,
      rRelativePos2 = 0;

    if (cMC.m_anTotalSize[0] > 1) rRelativePos0 = (float)iDim0 / (float)(cMC.m_anTotalSize[0] - 1);
    if (cMC.m_anTotalSize[1] > 1) rRelativePos1 = (float)iDim1 / (float)(cMC.m_anTotalSize[1] - 1);
    if (cMC.m_anTotalSize[2] > 1) rRelativePos2 = (float)iDim2 / (float)(cMC.m_anTotalSize[2] - 1);

    // Trace up/down against grid
    float3
      cPos;

    cPos.x = cMC.m_cOrigo.x;
    cPos.y = cMC.m_cOrigo.y;
    cPos.z = cMC.m_cOrigo.z;

    cPos = dadd(cPos, dscale(cMC.m_cExtent0, rRelativePos0));
    cPos = dadd(cPos, dscale(cMC.m_cExtent1, rRelativePos1));
    cPos = dadd(cPos, dscale(cMC.m_cExtent2, rRelativePos2));

    // Find closest
    float
      rClosestDist = cMC.m_maxDistance;
    
    if (isLine)
    {
      FindClosestDistanceAndPrimitiveID<true>(cPos, rClosestDist);
    }
    else
    {
      FindClosestDistanceAndPrimitiveID<false>(cPos, rClosestDist);
    }
  
    rClosestDist *= cMC.m_distanceScale;
    rClosestDist += cMC.m_distanceOffset;

    int
      iOffset0 = (iDim0 - cMC.m_anMin[0]) * cMC.m_anOutputPitch[0] + 
                 (iDim1 - cMC.m_anMin[1]) * cMC.m_anOutputPitch[1] + 
                 (iDim2 - cMC.m_anMin[2]) * cMC.m_anOutputPitch[2];

    
    cMC.m_pOutput[iOffset0] = rClosestDist;
  }
}

extern "C" void
MinDistCuda(MinDistanceContext_c cContext, bool isLine)
{
  dim3
    cGridSize,
    cBlockSize;
  
  cBlockSize.x = 8;
  cBlockSize.y = 16;
  cBlockSize.z = 1;

  cGridSize.x = cContext.m_anMax[0] - cContext.m_anMin[0];
  cGridSize.y = cContext.m_anMax[1] - cContext.m_anMin[1];
  cGridSize.x = (cGridSize.x + cBlockSize.x - 1) / cBlockSize.x;
  cGridSize.y = (cGridSize.y + cBlockSize.y - 1) / cBlockSize.y;
  cGridSize.z = 1;

  //// trick for doing 3d thread setup, when grids only can be 2d...
  int nDim2Modulo = cGridSize.y;
  cGridSize.y *= cContext.m_anMax[2] - cContext.m_anMin[2];

  cudaMemcpyToSymbol(cMC, &cContext, sizeof(MinDistanceContext_c));

  cudaChannelFormatDesc 
    channelDescInt1,
    channelDescInt2,
    channelDescFloat4;

  channelDescInt1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
  channelDescInt2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
  channelDescFloat4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
 
  if (isLine)
  {
    cudaBindTexture(0, texLines, cContext.m_pnLines, channelDescInt2);
  }
  else
  {
    cudaBindTexture(0, texTriangles, cContext.m_pnTriangles, channelDescInt1);
  }

  cudaBindTexture(0, texPrimitives, cContext.m_BVH.m_pnPrimitives, channelDescInt1);
  cudaBindTexture(0, texVertices, cContext.m_pcFloatV, channelDescFloat4);
  
  FindMinimalDistance<<<cGridSize, cBlockSize, 0, 0>>>(nDim2Modulo, isLine);

  if (isLine)
  {
    cudaUnbindTexture(texLines);
  }
  else
  {
    cudaUnbindTexture(texTriangles);
  }
  cudaUnbindTexture(texPrimitives);
  cudaUnbindTexture(texVertices);
}



