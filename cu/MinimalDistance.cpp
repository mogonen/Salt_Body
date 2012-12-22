/////////////////////////////////////////////////////////////////////////////
// Filter test
// (c) Hue AS 2007

// FILTER CODE

#include <assert.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime_api.h>

#ifdef WIN32
  #define WIN32_LEAN_AND_MEAN		
  #include <windows.h>
  #define DLLEXPORT __declspec(dllexport)
  BOOL APIENTRY DllMain(HANDLE, DWORD, LPVOID) { return TRUE; }
#else
  #include <string.h>
  #define DLLEXPORT
#endif

// INCLUDES /////////////////////////////////////////////////////////////////
#ifndef HUEFILTERHELPER_DISCARD_COMPILE_CPP
#include "HueComponentInterfaces.h"
#include "PluginInterface.h"
#include "HueSpaceLib.h"
#endif

#define HUEFILTERHELPER_CUDACOMPILE
// DEFINES //////////////////////////////////////////////////////////////////
#include "../Common/HueFilterHelper.h"
#include "../Common/HueFilterHelper_1Output1Input_Setup.h"

#include "MinimalDistance.h"

using namespace Hue::Util;

class MinimalDistance : public Hue::HueSpaceLib::VDSFilterPlugin
{
private:
  bool          m_isNeedCudaUpdate;
  bool          m_isCudaAllocated;

  int           m_size[3];

  double        m_maxDistance;

  double        m_distanceScale,
                m_distanceOffset;

  Hue::Util::DoubleVector3
                m_origin;

  Hue::Util::DoubleVector3
                m_Dim0Extent,
                m_Dim1Extent,
                m_Dim2Extent;

  Hue::Util::DoubleVector3
                m_vertexScale;

  Hue::HueSpaceLib::ProxyProperty<Hue::Util::DoubleVector3>
                m_vertexProperty;

  Hue::HueSpaceLib::ProxyProperty<Hue::HueSpaceLib::LineSegment>
                m_lineProperty;

  Hue::HueSpaceLib::ProxyProperty<Hue::HueSpaceLib::Triangle>
                m_triangleProperty;

  Hue::HueSpaceLib::BoundingVolumeHierarchy
                m_lineBoundingVolumeHierarchy,
                m_triangleBoundingVolumeHierarchy;

  MinDistanceContext_c
                cContext;

public:
  MinimalDistance(Hue::HueSpaceLib::DataBlockFactory *pcDataBlockFactory)
    : Hue::HueSpaceLib::VDSFilterPlugin(pcDataBlockFactory)
  {
    m_isCudaAllocated = false;
    m_isNeedCudaUpdate = false;
  }

  ~MinimalDistance() 
  {
    m_isCudaAllocated = false;
    m_isNeedCudaUpdate = false;
  }

  void DeAllocateCuda()
  {
    if (!m_isCudaAllocated) return;
    m_isCudaAllocated = false;

    cudaFree((void *)cContext.m_BVH.m_pnPrimitives);
    cudaFree((void *)cContext.m_BVH.m_pcNodes);
    
    if (cContext.m_pnLines)
    {
      cudaFree((void *)cContext.m_pnLines);
    }
    else
    {
      cudaFree((void *)cContext.m_pnTriangles);
    }

    cudaFree((void *)cContext.m_pcBoundingBox);
    cudaFree((void *)cContext.m_pcFloatV);
  }


  void AllocateCuda()
  {
    if (m_isCudaAllocated) return;
    m_isCudaAllocated = true;
    // Relativize top TopBB and scale with ve
    CudaBoundingVolumeHierarchy *
      pcCurrentBVH;

    bool
      isLine = false;

    if (m_lineProperty.size())
    {   
      isLine = true;
      pcCurrentBVH = (CudaBoundingVolumeHierarchy *)&m_lineBoundingVolumeHierarchy;
    }
    else
    {
      pcCurrentBVH = (CudaBoundingVolumeHierarchy *)&m_triangleBoundingVolumeHierarchy;
    }

    cContext.m_BVH = *pcCurrentBVH;

    // Scale TopBB to vertexscale
    cContext.m_BVH.m_cTopBB.Min.x *= m_vertexScale.X;
    cContext.m_BVH.m_cTopBB.Min.y *= m_vertexScale.Y;
    cContext.m_BVH.m_cTopBB.Min.z *= m_vertexScale.Z;
    cContext.m_BVH.m_cTopBB.Max.x *= m_vertexScale.X;
    cContext.m_BVH.m_cTopBB.Max.y *= m_vertexScale.Y;
    cContext.m_BVH.m_cTopBB.Max.z *= m_vertexScale.Z;

    // Apply vertexscale and reltivise origo to TopBB Min
    cContext.m_cOrigo.x = (float)(m_origin.X * m_vertexScale.X - cContext.m_BVH.m_cTopBB.Min.x);
    cContext.m_cOrigo.y = (float)(m_origin.Y * m_vertexScale.Y - cContext.m_BVH.m_cTopBB.Min.y);
    cContext.m_cOrigo.z = (float)(m_origin.Z * m_vertexScale.Z - cContext.m_BVH.m_cTopBB.Min.z);

    for (int i=0; i<3; i++)
    {
      cContext.m_anTotalSize[i] = m_size[i];
    }

    cContext.m_cExtent0.x = (float)(m_Dim0Extent.X * m_vertexScale.X);
    cContext.m_cExtent0.y = (float)(m_Dim0Extent.Y * m_vertexScale.Y);
    cContext.m_cExtent0.z = (float)(m_Dim0Extent.Z * m_vertexScale.Z);
    cContext.m_cExtent1.x = (float)(m_Dim1Extent.X * m_vertexScale.X);
    cContext.m_cExtent1.y = (float)(m_Dim1Extent.Y * m_vertexScale.Y);
    cContext.m_cExtent1.z = (float)(m_Dim1Extent.Z * m_vertexScale.Z);
    cContext.m_cExtent2.x = (float)(m_Dim2Extent.X * m_vertexScale.X);
    cContext.m_cExtent2.y = (float)(m_Dim2Extent.Y * m_vertexScale.Y);
    cContext.m_cExtent2.z = (float)(m_Dim2Extent.Z * m_vertexScale.Z);

    cContext.m_distanceScale = (float)m_distanceScale;
    cContext.m_distanceOffset = (float)m_distanceOffset;

    float4
      *pcFloatV = (float4 *)malloc(sizeof(float4) * m_vertexProperty.size());

    // Create float relative vertices(from top bb min) and scale to vertexscale
    for (unsigned int i=0; i< m_vertexProperty.size(); i++)
    {
      pcFloatV[i].x = (float)(m_vertexProperty[i].X * m_vertexScale.X - cContext.m_BVH.m_cTopBB.Min.x);
      pcFloatV[i].y = (float)(m_vertexProperty[i].Y * m_vertexScale.Y - cContext.m_BVH.m_cTopBB.Min.y);
      pcFloatV[i].z = (float)(m_vertexProperty[i].Z * m_vertexScale.Z - cContext.m_BVH.m_cTopBB.Min.z);
      pcFloatV[i].w = 0;
    }

    // Relativise top bb to itself
    cContext.m_BVH.m_cTopBB.Max.x -= cContext.m_BVH.m_cTopBB.Min.x;
    cContext.m_BVH.m_cTopBB.Max.y -= cContext.m_BVH.m_cTopBB.Min.y;
    cContext.m_BVH.m_cTopBB.Max.z -= cContext.m_BVH.m_cTopBB.Min.z;
    cContext.m_BVH.m_cTopBB.Min.x = 0.0f;
    cContext.m_BVH.m_cTopBB.Min.y = 0.0f;
    cContext.m_BVH.m_cTopBB.Min.z = 0.0f;

    // create boundingboxes
    CudaBoundingBox
      *pcBB = (CudaBoundingBox *)malloc(sizeof(CudaBoundingBox) * cContext.m_BVH.m_nNodes); 

    CreateBB(pcBB, 0, cContext.m_BVH.m_pcNodes, cContext.m_BVH.m_cTopBB);

    cudaMalloc((void **)&cContext.m_BVH.m_pnPrimitives, sizeof(int) * cContext.m_BVH.m_nPrimitives);
    cudaMalloc((void **)&cContext.m_BVH.m_pcNodes, sizeof(int2) * cContext.m_BVH.m_nNodes);

    cContext.m_pnTriangles = NULL;
    cContext.m_pnLines = NULL;

    if (isLine)
    {
      cudaMalloc((void **)&cContext.m_pnLines,      sizeof(int2) * m_lineProperty.size());
    }
    else
    {
      cudaMalloc((void **)&cContext.m_pnTriangles,      sizeof(int3) * m_triangleProperty.size());
    }

    cudaMalloc((void **)&cContext.m_pcBoundingBox, sizeof(CudaBoundingBox) * cContext.m_BVH.m_nNodes);
    cudaMalloc((void **)&cContext.m_pcFloatV , sizeof(float4) * m_vertexProperty.size());

    if (isLine)
    {
      cudaMemcpy(cContext.m_pnLines, &m_lineProperty[0], sizeof(int2) * m_lineProperty.size(), cudaMemcpyHostToDevice);
    }
    else
    {
      cudaMemcpy(cContext.m_pnTriangles, &m_triangleProperty[0], sizeof(int3) * m_triangleProperty.size(), cudaMemcpyHostToDevice);
    }

    cudaMemcpy((void *)cContext.m_BVH.m_pnPrimitives, (void *)pcCurrentBVH->m_pnPrimitives, sizeof(int) * cContext.m_BVH.m_nPrimitives, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)cContext.m_BVH.m_pcNodes, (void *)pcCurrentBVH->m_pcNodes, sizeof(int2) * cContext.m_BVH.m_nNodes, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)cContext.m_pcBoundingBox, pcBB, sizeof(CudaBoundingBox) * cContext.m_BVH.m_nNodes, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)cContext.m_pcFloatV, pcFloatV, sizeof(float4) * m_vertexProperty.size(), cudaMemcpyHostToDevice);

    free(pcBB);
    free(pcFloatV);
  }

  virtual void ProcessFilter(Hue::HueSpaceLib::VolumeDataCacheItem** papcOutputVolumeDataCacheItem, int nOutputVolumeDataCacheItem, const Hue::HueSpaceLib::VolumeDataCacheItem** papcInputVolumeDataCacheItem, int nInputVolumeDataCacheItem, Context *pcContext);

  virtual Hue::HueSpaceLib::VolumeDataAxisDescriptor
    DescribeOutputAxis(int dimension, Hue::HueSpaceLib::AccessPluginParameterInterface& parameterInterface)
  {
    const char *names[3] = { "X", "Y", "Z"};

    return Hue::HueSpaceLib::VolumeDataAxisDescriptor(m_size[dimension], names[dimension], "", 1.0f, 1.0f + m_size[dimension]);
  }

  virtual void 
    OnPluginParametersCreated(Hue::HueSpaceLib::AccessPluginParameterInterface& parameterInterface)
  {
    m_maxDistance = parameterInterface.GetValueDouble("MaxDistance");
    m_size[0] = parameterInterface.GetValueInt("SizeDim0");
    m_size[1] = parameterInterface.GetValueInt("SizeDim1");
    m_size[2] = parameterInterface.GetValueInt("SizeDim2");
    m_origin = parameterInterface.GetValueDoubleVector3("Origin");
    m_Dim0Extent = parameterInterface.GetValueDoubleVector3("Dim0Extent");
    m_Dim1Extent = parameterInterface.GetValueDoubleVector3("Dim1Extent");
    m_Dim2Extent = parameterInterface.GetValueDoubleVector3("Dim2Extent");
    m_vertexScale = parameterInterface.GetValueDoubleVector3("VertexScale");
    m_vertexProperty = parameterInterface.GetValueShapeVertexProperty("Shape");
    m_lineProperty = parameterInterface.GetValueShapeLineSegmentPrimitives("Shape");
    m_triangleProperty = parameterInterface.GetValueShapeTrianglePrimitives("Shape");
    m_triangleBoundingVolumeHierarchy = parameterInterface.GetValueShapeTriangleBoundingVolumeHierarchy("Shape");
    m_lineBoundingVolumeHierarchy = parameterInterface.GetValueShapeLineSegmentBoundingVolumeHierarchy("Shape");
    m_distanceScale = parameterInterface.GetValueDouble("DistanceScale");
    m_distanceOffset = parameterInterface.GetValueDouble("DistanceOffset");

    m_isNeedCudaUpdate = true;
  }

  virtual ParametersChangedAction
    OnPluginParametersChanged(Hue::HueSpaceLib::AccessPluginParameterInterface& parameterInterface) 
  {
    OnPluginParametersCreated(parameterInterface);
    return Hue::HueSpaceLib::VDSPlugin::Restart_Processing; 
  }

  virtual Hue::HueSpaceLib::IndexingDescriptor
    DescribeInputIndexingMethod(int iInputVDS, int iDimension)
  {
    return IndexingDescriptorEnd();
  }

  static Hue::HueSpaceLib::PluginParameterDescriptor
    DescribePluginParameter(int iParameter)
  {
    static Hue::HueSpaceLib::PluginParameterDescriptor parameters[] =
    {
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterDouble("MaxDistance", "Max Distance", "Max Distance we are to find", 1000000000.02),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterInt("SizeDim0", "Size Dim 0", "Size along dimension 0 (in voxels)", 200, 1, 10000),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterInt("SizeDim1", "Size Dim 1", "Size along dimension 1 (in voxels)", 200, 1, 10000),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterInt("SizeDim2", "Size Dim 2", "Size along dimension 2 (in voxels)", 200, 1, 10000),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterDoubleVector3("Origin", "Origin", "The XYZ origin of the IJK grid (i.e. the XYZ position of IJK coordinate 0,0,0)", Hue::HueSpaceLib::DoubleVector3(-1, -1, -1)),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterDoubleVector3("Dim0Extent", "Dim 0 Extent", "The XYZ spacing of the whole volume in the Dim 0 direction", Hue::HueSpaceLib::DoubleVector3(2, 0, 0)),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterDoubleVector3("Dim1Extent", "Dim 1 Extent", "The XYZ spacing of the whole volume in the Dim 1 direction", Hue::HueSpaceLib::DoubleVector3(0, 2, 0)),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterDoubleVector3("Dim2Extent", "Dim 2 Extent", "The XYZ spacing of the whole volume in the Dim 2 direction", Hue::HueSpaceLib::DoubleVector3(0, 0, 2)),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterDoubleVector3("VertexScale", "VertexScale", "The XYZ scaling of the vertices", Hue::HueSpaceLib::DoubleVector3(1, 1, 1)),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterShape("Shape", "Shape", "Shape"),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterDouble("DistanceScale", "Distance Scale", "The resulting distance is scaled with this value", 1.0),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterDouble("DistanceOffset", "Distance Offset", "The resulting distance is offsetted with this value", 0.0),
      Hue::HueSpaceLib::PluginParameterDescriptor::PluginParameterEnd()
    };

    return parameters[iParameter];
  }

  virtual int GetOutputDimensionality(Hue::HueSpaceLib::AccessPluginParameterInterface& parameterInterface)
  {
    return 3;
  }

  virtual int
  GetOutputChannelCount(Hue::HueSpaceLib::AccessPluginParameterInterface& parameterInterface)
  {
    return 1;
  }

  virtual Hue::HueSpaceLib::VolumeDataChannelDescriptor
    DescribeOutputValue(Hue::HueSpaceLib::AccessPluginParameterInterface& parameterInterface, int iChannel)
  {

    //Value range is ditance of extent

    DoubleVector3
      cLength(m_Dim0Extent);

    cLength.Add(m_Dim1Extent);
    cLength.Add(m_Dim2Extent);
    
    double
      rValueMax = cLength.Length();

    return Hue::HueSpaceLib::VolumeDataChannelDescriptor(Hue::HueSpaceLib::DataBlock::FORMAT_R32,
                                                           Hue::HueSpaceLib::DataBlock::COMPONENTS_1,
                                                           "Distance", "",
                                                           (float)-rValueMax, (float)rValueMax);
  }
 
  virtual Hue::HueSpaceLib::InputChannelDescriptor
  DescribeInputChannel(int iChannel)
  {
    return Hue::HueSpaceLib::InputChannelDescriptor::InputChannelEnd();
  }


  void CreateBB(CudaBoundingBox * pcBB, int iNode, const CudaBoundingVolumeHierarchy::NodeEncoding * pcNodes, const CudaBoundingBoxDouble &cTopBB);

};


extern "C" 
{

  // GetPluginFactoryInstance /////////////////////////////////////////////////

  Hue::HueSpaceLib::PluginFactory &
    GetPluginFactoryInstance()
  {
    static Hue::HueSpaceLib::PluginFactoryTemplate<MinimalDistance>
      instance("MinimalDistance",
               "MinimalDistance",
               "Creates the distance field from lines or triangles within a threshold");

    return instance;
  }

  ////////////////////////////////////////////////////////////////////////////
  //
  // PLUGIN INTERFACE
  //
  void
    DLLEXPORT
    HuePlugin_Init(Hue::HueSpaceLib::HueInterfaceEnumerator *pHueInterfaceEnumerator)
  {
    Hue::HueSpaceLib::PluginRegistry *
      pPluginRegistry = (Hue::HueSpaceLib::PluginRegistry *)pHueInterfaceEnumerator->FindInterface(HUE_IID_PLUGINREGISTRY);

    assert(pPluginRegistry);
    pPluginRegistry->RegisterFactory(&GetPluginFactoryInstance());
  }


  void
    DLLEXPORT
    HuePlugin_DeInit(Hue::HueSpaceLib::HueInterfaceEnumerator *pHueInterfaceEnumerator)
  {
    Hue::HueSpaceLib::PluginRegistry *
      pPluginRegistry = (Hue::HueSpaceLib::PluginRegistry *)pHueInterfaceEnumerator->FindInterface(HUE_IID_PLUGINREGISTRY);

    assert(pPluginRegistry);
    pPluginRegistry->RemoveFactory(&GetPluginFactoryInstance());
  }

}

extern "C" void
MinDistCuda(MinDistanceContext_c cContext, bool isLine);

extern "C" void
InsideCuda(MinDistanceContext_c cContext);

//////////////////////////////////////////////////////////////////////////////////////
// ProcessFilter 
void 
MinimalDistance::CreateBB(CudaBoundingBox * pcBB, int iNode, const CudaBoundingVolumeHierarchy::NodeEncoding * pcNodes, const CudaBoundingBoxDouble &cTopBB)
{
  int
    anNode[32];
  
  CudaBoundingBox
    acBB[32];

  CudaBoundingBox
    cBB;
  
  cBB.Min.x = (float)cTopBB.Min.x;
  cBB.Min.y = (float)cTopBB.Min.y;
  cBB.Min.z = (float)cTopBB.Min.z;
  cBB.Max.x = (float)cTopBB.Max.x;
  cBB.Max.y = (float)cTopBB.Max.y;
  cBB.Max.z = (float)cTopBB.Max.z;

  int 
    nNode = 1;

  anNode[0] = 0;
  acBB[0] = cBB;

  while (nNode)
  {
    nNode--;
    
    int 
      iNode = anNode[nNode];

    CudaBoundingVolumeHierarchy::NodeEncoding
      cNode = pcNodes[iNode];

    cBB = acBB[nNode];
    pcBB[iNode] = cBB;

    if (!cNode.IsLeafNode())
    {
      int
        iNode0 = cNode.GetFirstChild(),
        iNode1 = iNode0 + 1;
 
      // find dist to cNode0 & cNode1
      CudaBoundingBox
        cBB0 = cNode.GetBBLeft(cBB),
        cBB1 = cNode.GetBBRight(cBB);
        
      acBB[nNode] = cBB0;
      anNode[nNode++] = iNode0;
          
      acBB[nNode] = cBB1;
      anNode[nNode++] = iNode1;
    }
  }
}


void 
MinimalDistance::ProcessFilter(Hue::HueSpaceLib::VolumeDataCacheItem** papcOutputVolumeDataCacheItem, int nOutputVolumeDataCacheItem, const Hue::HueSpaceLib::VolumeDataCacheItem** papcInputVolumeDataCacheItem, int nInputVolumeDataCacheItem, Context *pcContext)
{
 // Init FilterHelper class, creates inputs
  HueFilterHelper
    cFilterHelper(this, (void **)papcOutputVolumeDataCacheItem, nOutputVolumeDataCacheItem, (const void **)papcInputVolumeDataCacheItem, nInputVolumeDataCacheItem, pcContext);

  if (!cFilterHelper._isCUDA) return;

  if (!m_vertexProperty.size() ||
      (!m_lineProperty.size() &&
       !m_triangleProperty.size()))
  {
    papcOutputVolumeDataCacheItem[0]->SetConstant(0);
    return;
  }

  if (m_isNeedCudaUpdate)
  {
    DeAllocateCuda();
    AllocateCuda();
    m_isNeedCudaUpdate = false;
  }
     // Get min max of voxels we are to produce
  for (int i=0;i<3;i++)
  {
    cContext.m_anMin[i] = cFilterHelper.GetMin(HUEFILTERHELPER_OUTPUT, 0)[i];
    cContext.m_anMax[i] = cFilterHelper.GetMax(HUEFILTERHELPER_OUTPUT, 0)[i];
  }

  cContext.m_pOutput = (float *)cFilterHelper._apxOutputData[0];

  for (int i=0; i<3; i++)
  {
    cContext.m_anOutputPitch[i] =  cFilterHelper._aanOutputPitch[0][i];
  }

  cContext.m_maxDistance = (float)m_maxDistance;
  
  static cudaEvent_t
    tStart,
    tStop;
  
  cudaEventCreate(&tStart);
  cudaEventCreate(&tStop);

  bool
    isLine = false;

  if (cContext.m_pnLines) isLine = true;

  cudaEventRecord(tStart, 0);

  MinDistCuda(cContext, isLine);

  cudaEventRecord(tStop, 0);
  cudaEventSynchronize(tStop);

  static float
    outerTime;

  cudaEventElapsedTime(&outerTime, tStart, tStop);    

  // circle 669
  // cow 418
  // fault 41

  cudaEventDestroy(tStart);
  cudaEventDestroy(tStop);

}
