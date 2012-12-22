#ifndef _CUDAKDTREE_H
#define _CUDAKDTREE_H

#include <cuda_runtime_api.h>
#include "HueComponentInterfaces.h"
#include "PluginInterface.h"
//#include "HueSpaceLib.h"

class CudaBoundingBox
{
public:
  float3
    Min,
    Max;
};

class CudaBoundingBoxDouble
{
public:
  double3
    Min,
    Max;
};

class CudaBoundingVolumeHierarchy
{
  friend class Node;

public:

  class NodeEncoding
  {
  public:
    unsigned int
      m_bData; // 2 bits defining axis split, if it is 2 x 15 bits define max left and min right bb  - if leaf node, then number of primitives

    int
      m_iLeft; // if negative, index to next node, if 0 or postive, leafnode and index to primitives -

    __host__ __device__ bool IsLeafNode()
    {
      return m_iLeft >= 0; 
    }

    __host__ __device__ int GetPrimitiveCount()
    {
      return (int)m_bData;
    }

    __host__ __device__ int GetPrimitiveOffet()
    {
      return m_iLeft;
    }

    __host__ __device__ int GetFirstChild()
    {
      return -m_iLeft;
    }

    __host__ __device__ CudaBoundingBox GetBBLeft(const CudaBoundingBox &cBB) const
    {
      CudaBoundingBox cChildBB;

      unsigned int
        bData = m_bData;

      int
        iAxis = (bData >> 30) & 0x3;

      bData = (bData >> 15) & 0x7fff;

      float
        rFactorMin = (float)((bData >> 0) & 0x7f) / 127.0f,
        rFactorMax = (float)((bData >> 7) & 0x7f) / 127.0f;

      cChildBB.Min = cBB.Min;
      cChildBB.Max = cBB.Max;

      switch(iAxis)
      {
      case 0:
        cChildBB.Min.x = (cBB.Max.x - cBB.Min.x) * rFactorMin + cBB.Min.x;
        cChildBB.Max.x = (cBB.Max.x - cBB.Min.x) * rFactorMax + cBB.Min.x;
        break;
      case 1:
        cChildBB.Min.y = (cBB.Max.y - cBB.Min.y) * rFactorMin + cBB.Min.y;
        cChildBB.Max.y = (cBB.Max.y - cBB.Min.y) * rFactorMax + cBB.Min.y;
        break;
      case 2:
        cChildBB.Min.z = (cBB.Max.z - cBB.Min.z) * rFactorMin + cBB.Min.z;
        cChildBB.Max.z = (cBB.Max.z - cBB.Min.z) * rFactorMax + cBB.Min.z;
        break;
      }

      return cChildBB;
    }

    __host__ __device__ CudaBoundingBox GetBBRight(const CudaBoundingBox &cBB) const
    {
      CudaBoundingBox cChildBB;

      unsigned int
        bData = m_bData;

      int
        iAxis = (bData >> 30) & 0x3;

      bData = (bData >> 0) & 0x7fff;

      float
        rFactorMin = (float)((bData >> 0) & 0x7f) / 127.0f,
        rFactorMax = (float)((bData >> 7) & 0x7f) / 127.0f;

      cChildBB.Min = cBB.Min;
      cChildBB.Max = cBB.Max;

      switch(iAxis)
      {
      case 0:
        cChildBB.Min.x = (cBB.Max.x - cBB.Min.x) * rFactorMin + cBB.Min.x;
        cChildBB.Max.x = (cBB.Max.x - cBB.Min.x) * rFactorMax + cBB.Min.x;
        break;
      case 1:
        cChildBB.Min.y = (cBB.Max.y - cBB.Min.y) * rFactorMin + cBB.Min.y;
        cChildBB.Max.y = (cBB.Max.y - cBB.Min.y) * rFactorMax + cBB.Min.y;
        break;
      case 2:
        cChildBB.Min.z = (cBB.Max.z - cBB.Min.z) * rFactorMin + cBB.Min.z;
        cChildBB.Max.z = (cBB.Max.z - cBB.Min.z) * rFactorMax + cBB.Min.z;
        break;
      }

      return cChildBB;
    }


  };

  CudaBoundingBoxDouble
    m_cTopBB;

  const NodeEncoding *
    m_pcNodes;

  const int
    *m_pnRecursiveCount;

  const int
    *m_pnPrimitives;

  int
    m_nNodes,
    m_nRecursiveCount,
    m_nPrimitives;

  class Node
  {
    friend class CudaBoundingVolumeHierarchy;

    const CudaBoundingVolumeHierarchy
      *m_pcCudaBoundingVolumeHierarchy;

    int
      m_iNodeIndex;

    CudaBoundingBox
      m_cCudaBoundingBox;

    __host__ __device__ NodeEncoding const &GetNodeEncoding() const
    {
      return m_pcCudaBoundingVolumeHierarchy->m_pcNodes[m_iNodeIndex];
    }

    __host__ __device__ Node(CudaBoundingVolumeHierarchy const &cCudaBoundingVolumeHierarchy, int iNodeIndex, CudaBoundingBox const &cCudaBoundingBox) : m_pcCudaBoundingVolumeHierarchy(&cCudaBoundingVolumeHierarchy), m_iNodeIndex(iNodeIndex), m_cCudaBoundingBox(cCudaBoundingBox)
    {
    }

    __host__ __device__ Node GetChildNode(int iChild) const
    {
      int
        iChildNode = -GetNodeEncoding().m_iLeft + iChild;

      CudaBoundingBox cChildBB;

      unsigned int
        bData = GetNodeEncoding().m_bData;

      int
        iAxis = (bData >> 30) & 0x3;

      if(iChild == 0)
      {
        bData = (bData >> 15) & 0x7fff;
      }
      else
      {
        bData = (bData >> 0) & 0x7fff;
      }

      if (iAxis == 3)
      {
        // find min & max
        for (iAxis = 0; iAxis < 3; iAxis++)
        {
          float
            rFactorMin = (float)(bData & 7) / 8.0f,
            rFactorMax = (float)((bData >> 3) & 3) / 4.0f;

          switch(iAxis)
          {
          case 0:
            cChildBB.Min.x = (m_cCudaBoundingBox.Max.x - m_cCudaBoundingBox.Min.x) * rFactorMin + m_cCudaBoundingBox.Min.x;
            cChildBB.Max.x = (cChildBB.Min.x - m_cCudaBoundingBox.Max.x) * rFactorMax + m_cCudaBoundingBox.Max.x;
            break;
          case 1:
            cChildBB.Min.y = (m_cCudaBoundingBox.Max.y - m_cCudaBoundingBox.Min.y) * rFactorMin + m_cCudaBoundingBox.Min.y;
            cChildBB.Max.y = (cChildBB.Min.y - m_cCudaBoundingBox.Max.y) * rFactorMax + m_cCudaBoundingBox.Max.y;
            break;
          case 2:
            cChildBB.Min.z = (m_cCudaBoundingBox.Max.z - m_cCudaBoundingBox.Min.z) * rFactorMin + m_cCudaBoundingBox.Min.z;
            cChildBB.Max.z = (cChildBB.Min.z - m_cCudaBoundingBox.Max.z) * rFactorMax + m_cCudaBoundingBox.Max.z;
            break;
          }

          bData >>= 5;
        }
      }
      else
      {
        float
          rFactorMin = (float)((bData >> 0) & 0x7f) / 127.0f,
          rFactorMax = (float)((bData >> 7) & 0x7f) / 127.0f;

        cChildBB.Min = m_cCudaBoundingBox.Min;
        cChildBB.Max = m_cCudaBoundingBox.Max;

        switch(iAxis)
        {
        case 0:
          cChildBB.Min.x = (m_cCudaBoundingBox.Max.x - m_cCudaBoundingBox.Min.x) * rFactorMin + m_cCudaBoundingBox.Min.x;
          cChildBB.Max.x = (m_cCudaBoundingBox.Max.x - m_cCudaBoundingBox.Min.x) * rFactorMax + m_cCudaBoundingBox.Min.x;
          break;
        case 1:
          cChildBB.Min.y = (m_cCudaBoundingBox.Max.y - m_cCudaBoundingBox.Min.y) * rFactorMin + m_cCudaBoundingBox.Min.y;
          cChildBB.Max.y = (m_cCudaBoundingBox.Max.y - m_cCudaBoundingBox.Min.y) * rFactorMax + m_cCudaBoundingBox.Min.y;
          break;
        case 2:
          cChildBB.Min.z = (m_cCudaBoundingBox.Max.z - m_cCudaBoundingBox.Min.z) * rFactorMin + m_cCudaBoundingBox.Min.z;
          cChildBB.Max.z = (m_cCudaBoundingBox.Max.z - m_cCudaBoundingBox.Min.z) * rFactorMax + m_cCudaBoundingBox.Min.z;
          break;
        }
      }

      return Node(*m_pcCudaBoundingVolumeHierarchy, iChildNode, cChildBB);
    }

  public:
    __host__ __device__ Node() : m_pcCudaBoundingVolumeHierarchy(NULL), m_iNodeIndex(-1), m_cCudaBoundingBox()
    {
    }

    __host__ __device__ CudaBoundingBox const &
      GetCudaBoundingBox() const
    {
      return m_cCudaBoundingBox;
    }

    __host__ __device__ bool IsLeafNode() const
    {
      return GetNodeEncoding().m_iLeft >= 0; 
    }


    __host__ __device__ Node GetLeftChild() const
    {
      return GetChildNode(0);
    }

    __host__ __device__ Node GetRightChild() const
    {
      return GetChildNode(1);
    }

    __host__ __device__ int GetPrimitiveCount() const
    {
      return (int)GetNodeEncoding().m_bData;
    }

    __host__ __device__ int GetPrimitiveIndex(int primitive) const
    {
      return m_pcCudaBoundingVolumeHierarchy->m_pnPrimitives[GetNodeEncoding().m_iLeft + primitive];
    }

    __host__ __device__ int GetRecursiveCount(int iNode) const
    {
      if (m_pcCudaBoundingVolumeHierarchy->m_pnRecursiveCount)
      {
        return m_pcCudaBoundingVolumeHierarchy->m_pnRecursiveCount[iNode];
      }
      else
      {
        return 0;
      }
    }
  };

public:


 // __host__ __device__ CudaBoundingVolumeHierarchy() : m_pcNodes(NULL), m_pnRecursiveCount(NULL), m_pnPrimitives(NULL) {}
};

#endif