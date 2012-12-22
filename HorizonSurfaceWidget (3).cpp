/***************************************************************************\

  NAME:         HORIZONSURFACE.CPP

  DESCRIPTION:  This sample shows how to instantiate and animate some simple
                mesh objects

  NOTE:         This file is part of the HueSpace visualization API.
                Copyright (C) 2001-2011 Hue AS. All rights reserved.

\***************************************************************************/

// INCLUDES /////////////////////////////////////////////////////////////////

#include <QtGui/QMouseEvent>

#include "HorizonSurfaceWidget.h"

#include <sys/stat.h>
#include <math.h>
#include "RootObject.h"
#include "Workspace.h"
#include "SceneManager.h"
#include "Scene.h"
#include "ProjectManager.h"
#include "Project.h"
#include "VMVolumeProgram.h"
#include "ProjectLibrary.h"

#include "StringObjectMask.h"
#include "ObjectMaskManager.h"
#include "Viewer.h"
#include "ViewerManager.h"
#include "Camera.h"
#include "MovableObjectManager.h"
#include "Horizon.h"
#include "VDSGenNoise.h"
#include "VDSManager.h"
#include "Edit3DController.h"
#include "ControllerManager.h"
#include "OrbitCameraController.h"
#include "TransferFunction.h"
#include "TransferFunctionManager.h"
#include "RenderLayerScreengrab.h"
#include "RenderLayerManager.h"
#include "RenderLayerSupersample.h"
#include "RenderLayerManagerSingle.h"
#include "RenderLayer3D.h"
#include "ViewContextContainer.h"
#include "ViewContextManager.h"
#include "ShapeInstanceViewContext.h"
#include "WorldBackgroundViewContext.h"
#include "HorizonViewContext.h"
#include "TorusShape.h"
#include "MovableObjectManager.h"
#include "ShapeInstance.h"
#include "ViewContextManager.h"
#include "ViewContextContainer.h"
#include "ShapeInstanceViewContext.h"
#include "HueQtDefaultViewerWidget.h"
#include "Workspace.h"
#include "SceneManager.h"
#include "Scene.h"
#include "ProjectManager.h"

#include "ViewerManager.h"
#include "Viewer.h"
#include "ShapeManager.h"
#include "VolumeBox.h"
#include "VolumeSlice.h"
#include "RenderStateTree.h"
#include "VolumeRSManager.h"
#include "RenderStateTreeManager.h"
#include "RSTViewContext.h"
#include "SetFrontFaceStateNode.h"
#include "SetVolumeStateNode.h"
#include "RSTNodeManagerSingle.h"
#include "DirectShape.h"
#include "ShrinkWrapShape.h"
#include "IntersectionShape.h"
#include "TransformedShape.h"
#include "MinimalDistanceShape.h"
#include "Container.h"
#include "ContainerManager.h"
#include "VDSFilter.h"
#include "Plugin.h"
#include "PluginManager.h"
#include "PluginParameterManager.h"
#include "PluginParameterDouble.h"
#include "PluginParameterDoubleVector2.h"
#include "PluginParameterDoubleVector3.h"
#include "PluginParameterFloat.h"
#include "PluginParameterInt.h"
#include "PluginParameterShape.h"
#include "FileShape.h"
#include "RenderLayerEdgeShading.h"
#include "ConfigMemoryManagement.h"
#include "BranchInsideProbeNode.h"

#include "VMVolumeProgram.h"
#include "VMVolumeProgramManager.h"
#include "VMSystemVariable.h"
#include "VMSystemVariableManager.h"
#include "VMParameterManager.h"
#include "VMParameter.h"
#include "VMConstant.h"
#include "VMTempVariableManager.h"
#include "VMTempVariable.h"
#include "VMCommand.h"
#include "VMCommandManager.h"
#include "VMCommandAssign.h"
#include "VMCommandIfGreater.h"
#include "VMConstant.h"
#include "VMConstantManager.h"
#include "VMVariable.h"
#include "VMVariableManager.h"
#include "VMTypeInVMVariable.h"
#include "VMTypeInVMVariableObjectMask.h"
#include "VMTypeInVMVariableSubObject.h"
#include "Trace2DEventContext.h"
#include "IntersectionObjectListPolicy.h"
#include "VolumeIntersection.h"

#include <QtCore/QTimer>

using namespace Hue::ProxyLib;

// METHODS //////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// HorizonSurfaceWidget constructor

HorizonSurfaceWidget::HorizonSurfaceWidget(QWidget* pParent) : HueQtDefaultViewerWidget(pParent)
{
  Hue::ProxyLib::Project*
    pProject = Hue::ProxyLib::Workspace::Instance()->Scenes()[0]->Projects()[0];

  Hue::ProxyLib::Viewer*
    pViewer = pProject->Viewers().Create();

  m_pProject = pProject;
  m_pViewer = pViewer;

  m_nWellShapes = 0;
  m_nFaults = 0;

  m_pFilterContainer = NULL;

  m_cSceneScale = DoubleVector3(1,1,5);

  CreateVolume(pProject, pViewer);
  CreateHorizons(pProject, pViewer);
  CreateFaults(pProject, pViewer);
//  CreateWells(pProject, pViewer);
  CreateCombineChain(pProject, pViewer, 3);

  // Apply global scene scale to camera
  pViewer->ActiveCameraController()->CameraTarget()->SetSceneScale(m_cSceneScale);

  // Snap camera to content of scene
  pViewer->ActiveCameraController()->SnapToContents(CameraControllerSnapToContentsAxis::XZPlus);

  pViewer->Edit3DController()->SetActive(true);
  pViewer->Edit3DController()->SetEdit3DTargetType(Edit3DTargetType::ThreeDObject);
  pViewer->Edit3DController()->SetViewContextContainerSelection(NULL);
  pViewer->Edit3DController()->SetScreenSizeManipulatorCorrection(true);
  pViewer->Edit3DController()->SetDrawCornersOnManipulator(false);

  pViewer->Edit3DController()->SetEdit3DHitObjectListPolicy(IntersectionObjectListPolicy::Include);

  SetActiveViewer(pViewer);

  pViewer->SetLODSelectionMethod(LODSelectionMethod::ForceFullResolution);
  pViewer->RenderLayer3DInsideMagnifier()->SetAllowedRenderTime(100000.0f);
  pViewer->RenderLayer3DOutsideMagnifier()->SetAllowedRenderTime(100000.0f);

  //Set edge shading post process effect
  pViewer->RenderLayerEdgeShading()->SetEdgeStrength(0.4f);
  pViewer->RenderLayerEdgeShading()->SetSSAOStrength(1.0f);
  pViewer->RenderLayerEdgeShading()->SetSSAORadius(500.0f);

  Hue::ProxyLib::ConfigMemoryManagement::Instance()->SetGPUCacheMax(384);
  Hue::ProxyLib::ConfigMemoryManagement::Instance()->SetGPUDecompressMax(32);

  // Set
//   QTimer *foo = new QTimer(this);
//   connect(foo, SIGNAL(timeout()), this, SLOT(UpdateMyShape()));
//   foo->start(50);
}


/////////////////////////////////////////////////////////////////////////////
// HorizonSurfaceWidget destructor

HorizonSurfaceWidget::~HorizonSurfaceWidget()
{

}

void
HorizonSurfaceWidget::mouseMoveEvent(QMouseEvent *pMouseEvent)
{
  HueQtDefaultViewerWidget::mouseMoveEvent(pMouseEvent);

  Trace2DEventContext
    *pcEvent = Perform2DTrace(QPoint(pMouseEvent->x(), pMouseEvent->y()));

  char
    zToolTip[1024],
    zIntersection[512];

  sprintf_s(zToolTip, "");

  for (int i=0;i<pcEvent->VolumeIntersection().Count(); i++)
  {
    sprintf_s(zIntersection, "Value %d: %.3f  LOD: %d\n", i, pcEvent->VolumeIntersection().Item(i).Value, pcEvent->VolumeIntersection().Item(i).LOD);
    strcat_s(zToolTip,zIntersection);
  }

  sprintf_s(zIntersection,"Color: %.2f,%.2f,%.2f,%.2f", pcEvent->HitRGBAColor().X, pcEvent->HitRGBAColor().Y, pcEvent->HitRGBAColor().Z, pcEvent->HitRGBAColor().T);
  strcat_s(zToolTip,zIntersection);

  this->setToolTip(zToolTip);

  HueQtDefaultViewerWidget::mouseMoveEvent(pMouseEvent);

}

void HorizonSurfaceWidget::CreateWells( Hue::ProxyLib::Project* pProject, Hue::ProxyLib::Viewer* pViewer )
{
  // Create well ViewContext Container

  Hue::ProxyLib::ViewContextContainer
    *pWellViewContextContainer = pViewer->ViewContextContainerInsideAndOutsideMagnifier()->ViewContexts().CreateViewContextContainer();

  pWellViewContextContainer->SetName("myWellViewContextContainer");

  // Create well shape container
  Hue::ProxyLib::Container
    *pWellContainer = pProject->Containers().Create();

  pWellContainer->SetName("myWellContainer"); // for easier debugging

  // create 10 wells
  int
    nNumWells = 2;

  for (int i=0; i<nNumWells;i++)
  {
    Hue::ProxyLib::DirectShape
      *pWellShape = pWellContainer->Shapes().CreateDirectShape();

    double
      rXPos = 0,
      rYPos = 0,
      rZPos = 1;

    double
      rDirX = sin(7.28 * i / nNumWells),
      rDirY = cos(7.28 * i / nNumWells);

    rXPos += rDirX * 0.03;
    rYPos += rDirY * 0.03;

    rDirX /= 10000.0;
    rDirY /= 10000.0;

    // add some points to well
    pWellShape->EnterEditMode();

    int
      nSteps = 50;

    for (int iZ=0; iZ < nSteps; iZ++)
    {
      pWellShape->Vertices().Add(DoubleVector3(-rZPos, rXPos, rYPos));

      if (iZ!=0)
      {
        pWellShape->Lines().Add(Line(iZ-1,iZ));
      }

      rXPos += rDirX;
      rYPos += rDirY;
      rZPos += -2.0 / nSteps;

      rDirX *= 1.17;
      rDirY *= 1.17;
    }

    pWellShape->ExitEditMode();

    Hue::ProxyLib::ShapeInstance
      *pSI = m_pSeismicVB->Attached3DObj().CreateShapeInstance();

    pSI->SetShape(pWellShape);

    Hue::ProxyLib::ShapeInstanceViewContext
      *pVC = pWellViewContextContainer->ViewContexts().CreateShapeInstanceViewContext();

    pVC->SetShapeInstance(pSI);

    // Create a transformed shape with ShapeInstance as position, so that we can "move it" - just for fun
    Hue::ProxyLib::TransformedShape
      *pTS = pProject->Shapes().CreateTransformedShape();

    pTS->SetInputShape(pWellShape);
    pTS->SetMovableObjectOut(pSI);

    // add transformed shape in list of transformed shapes
    m_apWellShapes[m_nWellShapes++] = pTS;

   
  }
}

void HorizonSurfaceWidget::CreateFaults( Hue::ProxyLib::Project* pProject, Hue::ProxyLib::Viewer* pViewer )
{
  // Create well ViewContext Container
  Hue::ProxyLib::ViewContextContainer
    *pFaultViewContextContainer = pViewer->ViewContextContainerInsideAndOutsideMagnifier()->ViewContexts().CreateViewContextContainer();

  pFaultViewContextContainer->SetName("myFaultViewContextContainer");

  // Create well shape container
  Hue::ProxyLib::Container
    *pFaultContainer = pProject->Containers().Create();

  pFaultContainer->SetName("myFaultContainer"); 

  // create 2 faults
  int
    nNumFaults = 3;

  DoubleVector3
    cPos = m_pSeismicVB->Position(),
    cScale = m_pSeismicVB->Scale();

  cScale.X *= 2;
  cScale.Y *= 2;
  cScale.Z *= 3;

  m_nFaultViewContext = 0;

  for (int i=0; i<nNumFaults;i++)
  {
    Hue::ProxyLib::DirectShape
      *pFaultShape = pFaultContainer->Shapes().CreateDirectShape();

    double
      rXPos = 0,
      rYPos = 0;
    
    double
      rDirX = sin((7.28 * i) / nNumFaults),
      rDirY = cos((7.28 * i)/ nNumFaults);

    rXPos += rDirY * 3;
    rYPos -= rDirX;

    int
      nSteps = 20;//1024;

    rDirX = sin((7.28 * i) / nNumFaults),
    rDirY = cos((7.28 * i)/ nNumFaults);
    rDirX /= 3.0;
    rDirY /= 3.0;

    rXPos -= rDirX * 0.5 * nSteps;
    rYPos -= rDirY * 0.5 * nSteps;

    pFaultShape->EnterEditMode();

    for (int iZ=0; iZ < nSteps; iZ++)
    {
      DoubleVector3
        cP0(rXPos, rYPos, 1),
        cP1(rXPos * 1.1, rYPos * 1.1, -1);

      cP0.X *= cScale.Y;
      cP0.Y *= cScale.Y;
      cP0.Z *= cScale.Z;
      cP1.X *= cScale.Y;
      cP1.Y *= cScale.Y;
      cP1.Z *= cScale.Z;

      cP0.Add(cPos);
      cP1.Add(cPos);

      pFaultShape->Vertices().Add(cP0);
      pFaultShape->Vertices().Add(cP1);

      int
        iVertexOffset = iZ * 2;
      
      if (iVertexOffset!=0)
      {
        pFaultShape->Triangles().Add(Triangle(iVertexOffset-2,iVertexOffset,iVertexOffset -1));
        pFaultShape->Triangles().Add(Triangle(iVertexOffset-1,iVertexOffset,iVertexOffset + 1));
      }

      double rRand = (rand() & 32767) / 32767.0 - 0.5;

      rRand *= 3.0;

      rDirX = sin((7.28 * i + rRand) / nNumFaults),
      rDirY = cos((7.28 * i + rRand)/ nNumFaults);
      rDirX /= 3.0;
      rDirY /= 3.0;

      rXPos += rDirX;
      rYPos += rDirY;
    }

    pFaultShape->ExitEditMode();

    Hue::ProxyLib::ShapeInstance
      *pSI = pFaultContainer->MovableObjects().CreateShapeInstance();

    pSI->SetShape(pFaultShape);
    pSI->SetSurfaceShadingMode(SurfaceShadingMode::Smooth);

    pViewer->Edit3DController()->Edit3DHitObjectList().Add(pSI);

    Hue::ProxyLib::ShapeInstanceViewContext
      *pVC = pFaultViewContextContainer->ViewContexts().CreateShapeInstanceViewContext();

    pVC->SetShapeInstance(pSI);
    pVC->SetForceTwoSidedTriangles(true);
    pVC->SetTriangleRGBAColor(RGBAColor(183,212,183,255));

    // Create a transformed shape with ShapeInstance as position, so that we can "move it" - just for fun
    Hue::ProxyLib::TransformedShape
      *pTS = pProject->Shapes().CreateTransformedShape();

    pTS->SetInputShape(pFaultShape);
    pTS->SetMovableObjectOut(pSI);

    // add transformed shape in list of transformed shapes
    m_paAndShapes[m_nFaults++] = pTS;

    m_apFaultViewContext[m_nFaultViewContext++] = pVC;
  }
}

void HorizonSurfaceWidget::CreateVolume( Hue::ProxyLib::Project* pProject, Hue::ProxyLib::Viewer* pViewer)
{
  Hue::ProxyLib::Container
    *pSeismicContainer = pProject->Containers().Create();

  Hue::ProxyLib::VDS
    *pSeismicVDS = pSeismicContainer->RestoreVDSFromFileName("c:/project/PQT/Data/Stack_Training");

  Hue::ProxyLib::VolumeBox
    *pSeismicVB = pSeismicContainer->MovableObjects().CreateVolumeBox(),
    *pSeismicVB2 = pSeismicVB->Attached3DObj().CreateVolumeBox();

  m_pSeismicVDS = pSeismicVDS;
  m_pSeismicVB = pSeismicVB;

  Hue::ProxyLib::VolumeSlice
    *pVolumeSlice0 = pSeismicVB->Attached3DObj().CreateVolumeSlice(),
    *pVolumeSlice1 = pSeismicVB->Attached3DObj().CreateVolumeSlice();

  pVolumeSlice0->SetPosition(DoubleVector3(0,0,1));
  pVolumeSlice1->SetPosition(DoubleVector3(0,1,0));

  Hue::ProxyLib::ShapeInstance
    *pProbe = pSeismicVB->Attached3DObj().CreateShapeInstance();

  Hue::ProxyLib::BoxShape
    *pBoxShape = pProject->Shapes().CreateBoxShape();

  pProbe->SetShape((Hue::ProxyLib::Shape *)pBoxShape);
  pProbe->SetScale(DoubleVector3(1.0,1.0,1.0));

  pSeismicVB->SetVBCacheBias(0);
  pSeismicVB2->SetVBCacheBias(0);
  pVolumeSlice0->SetCacheBias(16);
  pVolumeSlice1->SetCacheBias(16);
  pVolumeSlice0->SetPlane(AxisAlignedPlane::XY);
  pVolumeSlice1->SetPlane(AxisAlignedPlane::XZ);

  pSeismicVB->SetValueVDS(pSeismicVDS);
  pSeismicVB->GetVDSCoordinateSystem(); 

  Hue::ProxyLib::TransferFunction
    *pSeismicTFSlice = pSeismicContainer->TransferFunctions().Create(), 
    *pSeismicTF0 = pSeismicContainer->TransferFunctions().Create(), 
    *pSeismicTF1 = pSeismicContainer->TransferFunctions().Create(),
    *pDistanceFieldTF = pSeismicContainer->TransferFunctions().Create();

  CreateSeismicTransferFunctions(pSeismicTF0, pSeismicVDS, pSeismicTF1, pSeismicTFSlice, pDistanceFieldTF);

  Hue::ProxyLib::VolumeRS *
    pVolumeRS = pSeismicContainer->VolumeRenderStates().Create(); 

  StringObjectMask
    *pStringObjectMask1D1SReadout = RootObject::Instance()->ObjectMasks().CreateStringObjectMask(SubObjectEnumString::NamedObject_Name, "1D1S_Readout");

  Hue::ProxyLib::ProxyObjectList<VMVolumeProgram*>
    vmVolumePrograms1D1SReadout = VMVolumeProgram::StaticFindAll(ProjectLibrary::Instance(), true, pStringObjectMask1D1SReadout);

  pVolumeRS->SetVMVolumeProgram(vmVolumePrograms1D1SReadout[0]); 
  pVolumeRS->SetVolumeBox0(pSeismicVB); 
  pVolumeRS->SetTransferFunction0(pSeismicTFSlice); 

  // Set up distance volume and renderstate
  Hue::ProxyLib::VolumeBox
    *pDistanceVolumeBox = pSeismicVB->Attached3DObj().CreateVolumeBox();

  pDistanceVolumeBox->SetVBCacheBias(0);

  m_pDistanceVolumeBox = pDistanceVolumeBox;

  pDistanceVolumeBox->SetValueLoadVoxelFormat(LoadDataFormat::UInt16);

  // Create own VMProgram that chooses which transferfunction for dataset1 based on value in dataset 0
  Hue::ProxyLib::VolumeRS *
    pDistVolumeRS = pProject->VolumeRenderStates().Create(); // This is a volume render state which defines how volume rendering should be performed.

  m_pDistVolumeRS = pDistVolumeRS;

  pDistVolumeRS->SetVMVolumeProgram(NULL);
  pDistVolumeRS->SetVolumeBox0(pDistanceVolumeBox);
  pDistVolumeRS->SetVolumeBox1(pSeismicVB);
  pDistVolumeRS->SetVolumeBox2(pSeismicVB2);
  pDistVolumeRS->SetTransferFunction0(pSeismicTF0);
  pDistVolumeRS->SetTransferFunction1(pSeismicTF1);
  pDistVolumeRS->SetTransferFunction2(pDistanceFieldTF);
  pDistVolumeRS->SetFloat0(50.0f);

  CreateDistanceFieldVMProgramOnRenderState(m_pDistVolumeRS,false,false,false,false);
  m_lastOutsideAlpha = 0.0f;

  // Create render state tree:
  Hue::ProxyLib::RenderStateTree *
    pRenderStateTree = pSeismicContainer->RenderStateTrees().Create(); 

  Hue::ProxyLib::SetFrontfaceStateNode
    *pVolumeSlice0State = pRenderStateTree->ChildNode().CreateSetFrontfaceStateNode(),
    *pVolumeSlice1State = pVolumeSlice0State->ChildNode().CreateSetFrontfaceStateNode(); 

  pVolumeSlice0State->SetSurfaceStateSet(pVolumeRS); 
  pVolumeSlice0State->SetShapeInstance(pVolumeSlice0);
  
  pVolumeSlice1State->SetSurfaceStateSet(pVolumeRS); 
  pVolumeSlice1State->SetShapeInstance(pVolumeSlice1);
  // Create view context (rst):
  Hue::ProxyLib::RSTViewContext *
    pRSTViewContext = pViewer->ViewContextContainerInsideAndOutsideMagnifier()->ViewContexts().CreateRSTViewContext();

  pRSTViewContext->SetRST(pRenderStateTree); 
  pRSTViewContext->SetMaterialDiffuseReflectivity(0.8f);
  pRSTViewContext->SetMaterialAmbientReflectivity(0.2f);
  pRSTViewContext->SetMaterialShininess(0.0f);
  pRSTViewContext->SetMaterialSpecularity(0.0f);

  // Create render state tree:
  Hue::ProxyLib::BranchInsideProbeNode
    *pBranchInsideProbeNode = pVolumeSlice1State->ChildNode().CreateBranchInsideProbeNode(); // This node type defines which surface render state to use for the front faces when rendering a 3D shape object. All objects are by default not visible in the Render state tree, so this node must be used to instantiate an object.

  pBranchInsideProbeNode->SetProbe(pProbe);
  pProbe->SetVBCacheBias(1.0f);

  // Create render state tree:
  Hue::ProxyLib::SetVolumeStateNode
    *pSetVolumeStateNode = pBranchInsideProbeNode->Inside().CreateSetVolumeStateNode(); // This node type defines which surface render state to use for the front faces when rendering a 3D shape object. All objects are by default not visible in the Render state tree, so this node must be used to instantiate an object.

  pSetVolumeStateNode->SetVolumeStateSet(pDistVolumeRS); // When the active node type is Set front-/backface state, this parameter defines which surface render state to set


  // Add to Editcontrolloer that this objects can be manipulated
  pViewer->Edit3DController()->Edit3DHitObjectList().Add(pVolumeSlice0);
  pViewer->Edit3DController()->Edit3DHitObjectList().Add(pVolumeSlice1);
  pViewer->Edit3DController()->Edit3DHitObjectList().Add(pProbe);

  // Create Seismic attribute
  Hue::ProxyLib::VDSFilter
    * pVDSAttribute;
  
  char
    *zFilename = "SignalEnergy8";

  struct stat buf;

  if (stat(zFilename, &buf) != -1)
  {
    pVDSAttribute = (Hue::ProxyLib::VDSFilter *)pProject->RestoreVDSFromFileName(zFilename);
  }
  else
  {
    pVDSAttribute = pProject->VDSs().CreateVDSFilter();
    pVDSAttribute->SetLODLevels(LODLevels::LODLevel2);
    pVDSAttribute->SetVDSFilterPlugin(RootObject::Instance()->Plugins().FindByName("Energy"));
    pVDSAttribute->SetInputVDSConnection0(pSeismicVDS);
    pVDSAttribute->SetCacheFileName(zFilename);
  }

  pSeismicVB2->SetValueVDS(pVDSAttribute);
}


void HorizonSurfaceWidget::UpdateCombineChain(int iTopHorizon, int iBottomHorizon)
{
    // delete previosly combined vds's
  if (m_pFilterContainer)
  {
    m_pFilterContainer->Delete();
  }

  Hue::ProxyLib::Container
    *pFilterContainer = m_pProject->Containers().Create();

  pFilterContainer->SetName("Container with VDSFilter plugins");

   m_pFilterContainer = pFilterContainer;

  int
    nAndVDS = m_nFaults + 2;

  // combine And VDS's in a binary tree
  Hue::ProxyLib::VDSFilter
    **apVDSCombine = new Hue::ProxyLib::VDSFilter * [nAndVDS * nAndVDS];

  int
    nCombine = 0;

  // Add faults
  for (int i=0; i<m_nFaults;i++)
  {
    if (m_abFaultDistanceOn[i])
    {
      apVDSCombine[nCombine] = m_apFaultDistanceVDS[i];
      nCombine++;
    }
  }

  for (int i=0; i<m_nWellShapes;i++)
  {
    apVDSCombine[nCombine] = m_apWellDistanceVDS[i];
    nCombine++;
  }

  // Add top bottom Horizon
  apVDSCombine[nCombine++] = m_apHorizonDistanceVDS[iTopHorizon];
  apVDSCombine[nCombine++] = m_apHorizonDistanceVDS[iBottomHorizon];


  Hue::ProxyLib::PluginParameterFloat
    * paramDistanceSignTop = static_cast<Hue::ProxyLib::PluginParameterFloat *>(m_apHorizonDistanceVDS[iTopHorizon]->PluginParameters().FindByName("DistanceSign")),
    * paramDistanceSignBottom = static_cast<Hue::ProxyLib::PluginParameterFloat *>(m_apHorizonDistanceVDS[iBottomHorizon]->PluginParameters().FindByName("DistanceSign"));

  paramDistanceSignTop->SetValue(m_cSceneScale.Z);
  paramDistanceSignBottom->SetValue(-m_cSceneScale.Z);

  int
    iCombinePos = 0;

  while ((nCombine - iCombinePos) > 1)
  {
    int
      nCombineLeft = nCombine - iCombinePos;

    int
      nCombineNow = nCombineLeft & 0xfffffffe; // make even number

    for (int i=0; i < nCombineNow; i+=2)
    {
      Hue::ProxyLib::VDSFilter
        * pDistanceOr = pFilterContainer->VDSs().CreateVDSFilter(); // This VDS takes a VDSPlugin as input

      pDistanceOr->SetVDSFilterPlugin(RootObject::Instance()->Plugins().FindByName("DistanceOr"));
      pDistanceOr->SetInputVDSConnection0(apVDSCombine[i+iCombinePos]);
      pDistanceOr->SetInputVDSConnection1(apVDSCombine[i+iCombinePos+1]);
      apVDSCombine[nCombine++] = pDistanceOr;
    }

    iCombinePos += nCombineNow;
  }

  m_pDistanceVolumeBox->SetValueVDS(apVDSCombine[nCombine-1]);
  delete apVDSCombine;

}

void HorizonSurfaceWidget::CreateCombineChain( Hue::ProxyLib::Project* pProject, Hue::ProxyLib::Viewer* pViewer, int nResolution)
{
  Hue::ProxyLib::FileShape
    *pcFileShape = pProject->Shapes().CreateFileShape();

  pcFileShape->SetInputFileName("C:/Project/trunk/Doc/HueSamples/data/Meshes/cow.smf");

  // Find sample area
  DoubleVector3 
    cVBPos = m_pSeismicVB->Position();

  DoubleVector3
    cVBScale = m_pSeismicVB->Scale();

  DoubleMatrix3x3
    cVBMatrix = m_pSeismicVB->Orientation();

  cVBMatrix.X.Scale(cVBScale.X);
  cVBMatrix.Y.Scale(cVBScale.Y);
  cVBMatrix.Z.Scale(cVBScale.Z);

  cVBPos.Subtract(cVBMatrix.X);
  cVBPos.Subtract(cVBMatrix.Y);
  cVBPos.Subtract(cVBMatrix.Z);

  cVBMatrix.X.Scale(2.0f);
  cVBMatrix.Y.Scale(2.0f);
  cVBMatrix.Z.Scale(2.0f);

  int
    sizeDim0 = m_pSeismicVDS->Dimension0Size(),
    sizeDim1 = m_pSeismicVDS->Dimension1Size(),
    sizeDim2 = m_pSeismicVDS->Dimension2Size();

  if (nResolution < 1) nResolution = 0;

  sizeDim0 /= nResolution;
  sizeDim1 /= nResolution;
  sizeDim2 /= nResolution;
 
  for (int i=0; i<m_nFaults;i++)
  {
    m_abFaultDistanceOn[i] = false;
    // Create distance VDS
    Hue::ProxyLib::VDSFilter
      * pVDSDistance = pProject->VDSs().CreateVDSFilter();

    pVDSDistance->SetLODLevels(LODLevels::LODLevelNone);
    pVDSDistance->SetVDSFilterPlugin(RootObject::Instance()->Plugins().FindByName("MinimalDistance"));

    Hue::ProxyLib::PluginParameterShape
      * paramShape = static_cast<Hue::ProxyLib::PluginParameterShape*>(pVDSDistance->PluginParameters().FindByName("Shape"));

    Hue::ProxyLib::PluginParameterDoubleVector3 
      * paramOrigin = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("Origin")),
      * paramDim0Extent = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("Dim0Extent")),
      * paramDim1Extent = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("Dim1Extent")),
      * paramDim2Extent = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("Dim2Extent"));
 
    Hue::ProxyLib::PluginParameterInt
      * paramSizeDim0 = static_cast<Hue::ProxyLib::PluginParameterInt *>(pVDSDistance->PluginParameters().FindByName("SizeDim0")),
      * paramSizeDim1 = static_cast<Hue::ProxyLib::PluginParameterInt *>(pVDSDistance->PluginParameters().FindByName("SizeDim1")),
      * paramSizeDim2 = static_cast<Hue::ProxyLib::PluginParameterInt *>(pVDSDistance->PluginParameters().FindByName("SizeDim2"));

    Hue::ProxyLib::PluginParameterDoubleVector3 
      * paramVertexScale = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("VertexScale"));
 
    Hue::ProxyLib::PluginParameterDouble
      * paramMaxDistance = static_cast<Hue::ProxyLib::PluginParameterDouble *>(pVDSDistance->PluginParameters().FindByName("MaxDistance"));

    paramMaxDistance->SetValue(10000000.0);

    paramVertexScale->SetValue(DoubleVector3(m_cSceneScale.X, m_cSceneScale.Y, m_cSceneScale.Z));

    paramSizeDim0->SetValue(sizeDim0);
    paramSizeDim1->SetValue(sizeDim1);
    paramSizeDim2->SetValue(sizeDim2);

    paramShape->SetShape(m_paAndShapes[i]);

    paramOrigin->SetValue(cVBPos);
    paramDim0Extent->SetValue(cVBMatrix.X);
    paramDim1Extent->SetValue(cVBMatrix.Y);
    paramDim2Extent->SetValue(cVBMatrix.Z);

    m_apFaultDistanceVDS[i] = pVDSDistance;
  }

  for (int i=0; i<m_nWellShapes;i++)
  {
    // Create distance VDS
    Hue::ProxyLib::VDSFilter
      * pVDSDistance = pProject->VDSs().CreateVDSFilter();

    pVDSDistance->SetLODLevels(LODLevels::LODLevelNone);
    pVDSDistance->SetVDSFilterPlugin(RootObject::Instance()->Plugins().FindByName("MinimalDistance"));

    Hue::ProxyLib::PluginParameterShape
      * paramShape = static_cast<Hue::ProxyLib::PluginParameterShape*>(pVDSDistance->PluginParameters().FindByName("Shape"));

    Hue::ProxyLib::PluginParameterDoubleVector3 
      * paramOrigin = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("Origin")),
      * paramDim0Extent = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("Dim0Extent")),
      * paramDim1Extent = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("Dim1Extent")),
      * paramDim2Extent = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("Dim2Extent"));

    Hue::ProxyLib::PluginParameterInt
      * paramSizeDim0 = static_cast<Hue::ProxyLib::PluginParameterInt *>(pVDSDistance->PluginParameters().FindByName("SizeDim0")),
      * paramSizeDim1 = static_cast<Hue::ProxyLib::PluginParameterInt *>(pVDSDistance->PluginParameters().FindByName("SizeDim1")),
      * paramSizeDim2 = static_cast<Hue::ProxyLib::PluginParameterInt *>(pVDSDistance->PluginParameters().FindByName("SizeDim2"));

    Hue::ProxyLib::PluginParameterDoubleVector3 
      * paramVertexScale = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("VertexScale"));

    paramVertexScale->SetValue(DoubleVector3(m_cSceneScale.X, m_cSceneScale.Y, m_cSceneScale.Z));

    paramSizeDim0->SetValue(sizeDim0);
    paramSizeDim1->SetValue(sizeDim1);
    paramSizeDim2->SetValue(sizeDim2);

    paramShape->SetShape(m_apWellShapes[i]);

    paramOrigin->SetValue(cVBPos);
    paramDim0Extent->SetValue(cVBMatrix.X);
    paramDim1Extent->SetValue(cVBMatrix.Y);
    paramDim2Extent->SetValue(cVBMatrix.Z);

    Hue::ProxyLib::PluginParameterDouble
      * paramScale = static_cast<Hue::ProxyLib::PluginParameterDouble *>(pVDSDistance->PluginParameters().FindByName("DistanceScale")),
      * paramOffset = static_cast<Hue::ProxyLib::PluginParameterDouble *>(pVDSDistance->PluginParameters().FindByName("DistanceOffset"));

    paramScale->SetValue(-1.0);
    paramOffset->SetValue(400);

    m_apWellDistanceVDS[i] = pVDSDistance;
  }


  for (int iHorizon = 0; iHorizon<m_nHorizonVDS; iHorizon++)
  {
    // Create Horizon Distance plugin
    Hue::ProxyLib::VDSFilter
      * pVDSDistance = pProject->VDSs().CreateVDSFilter();

    pVDSDistance->SetLODLevels(LODLevels::LODLevelNone);

    pVDSDistance->SetVDSFilterPlugin(RootObject::Instance()->Plugins().FindByName("MinimalDistanceHorizon"));

    Hue::ProxyLib::PluginParameterDoubleVector3 
      * paramOrigin = static_cast<Hue::ProxyLib::PluginParameterDoubleVector3 *>(pVDSDistance->PluginParameters().FindByName("Origin"));

    Hue::ProxyLib::PluginParameterDouble
      * paramDim0Extent = static_cast<Hue::ProxyLib::PluginParameterDouble *>(pVDSDistance->PluginParameters().FindByName("Dim0Extent"));

    Hue::ProxyLib::PluginParameterDoubleVector2 
      * paramDim1Extent = static_cast<Hue::ProxyLib::PluginParameterDoubleVector2 *>(pVDSDistance->PluginParameters().FindByName("Dim1Extent")),
      * paramDim2Extent = static_cast<Hue::ProxyLib::PluginParameterDoubleVector2 *>(pVDSDistance->PluginParameters().FindByName("Dim2Extent"));

    Hue::ProxyLib::PluginParameterInt
      * paramSizeDim0 = static_cast<Hue::ProxyLib::PluginParameterInt *>(pVDSDistance->PluginParameters().FindByName("SizeDim0")),
      * paramSizeDim1 = static_cast<Hue::ProxyLib::PluginParameterInt *>(pVDSDistance->PluginParameters().FindByName("SizeDim1")),
      * paramSizeDim2 = static_cast<Hue::ProxyLib::PluginParameterInt *>(pVDSDistance->PluginParameters().FindByName("SizeDim2"));

    Hue::ProxyLib::PluginParameterFloat
      * paramDistanceSign = static_cast<Hue::ProxyLib::PluginParameterFloat *>(pVDSDistance->PluginParameters().FindByName("DistanceSign"));

    paramSizeDim0->SetValue(sizeDim0);
    paramSizeDim1->SetValue(sizeDim1);
    paramSizeDim2->SetValue(sizeDim2);

    paramOrigin->SetValue(cVBPos);
    paramDim0Extent->SetValue(-cVBMatrix.X.Z);
    paramDim1Extent->SetValue(DoubleVector2(cVBMatrix.Y.X, cVBMatrix.Y.Y));
    paramDim2Extent->SetValue(DoubleVector2(cVBMatrix.Z.X, cVBMatrix.Z.Y));

    paramDistanceSign->SetValue(1.0f);

    pVDSDistance->SetInputVDSConnection0(m_apHorizonVDS[iHorizon]);
    m_apHorizonDistanceVDS[iHorizon] = pVDSDistance;
  }
}


void HorizonSurfaceWidget::CreateHorizons( Hue::ProxyLib::Project* pProject, Hue::ProxyLib::Viewer* pViewer)
{
  Hue::ProxyLib::ViewContextContainer
    *pHorizonViewContextContainer = pViewer->ViewContextContainerInsideAndOutsideMagnifier()->ViewContexts().CreateViewContextContainer();

  pHorizonViewContextContainer->SetName("myHorizonViewContextContainer");

  Hue::ProxyLib::TransferFunction *
    pHorizonTF = pProject->TransferFunctions().Create(); // A transfer function is an object which contains information about how to map from raw, original data values to final colors and transparency. They can be applied to volume boxes and mesh surfaces using VM programs.

  pHorizonTF->ColorHandles().Clear();
  pHorizonTF->ColorHandles().Add(ColorHandle(1.0, RGBColor(255,128,0), (RGBColor(255,128,0))));
  pHorizonTF->ColorHandles().Add(ColorHandle(0.5, RGBColor(255,255,255), (RGBColor(255,255,255))));
  pHorizonTF->ColorHandles().Add(ColorHandle(0.0, RGBColor(0,92,232), (RGBColor(0,92,232))));
  pHorizonTF->AlphaHandles().Clear();
  pHorizonTF->AlphaHandles().Add(AlphaHandle(0.0, 1.0f, 1.0f));

  m_nHorizonVDS = 0;

  for (int iHorizon = 0; iHorizon < 6; iHorizon++)
  {
    char
      zName[256];

    sprintf_s(zName, "c:/project/PQT/Data/Horizon_%d_Charisma_format", iHorizon + 1);
    
    Hue::ProxyLib::VDS
      *pHorizonVDS = pProject->RestoreVDSFromFileName(zName);

    m_apHorizonVDS[iHorizon] = pHorizonVDS;

    Hue::ProxyLib::Horizon
      *pHorizonSI = pProject->MovableObjects().CreateHorizon();

    pHorizonSI->SetSurfaceShadingMode(SurfaceShadingMode::Smooth);

    pHorizonSI->SetHeightVDS(pHorizonVDS);
    pHorizonSI->GetVDSCoordinateSystem();
    pHorizonSI->SetPlane(HorizonPlane::YZ);

    Hue::ProxyLib::HorizonViewContext
      *pVC = pHorizonViewContextContainer->ViewContexts().CreateHorizonViewContext();

    m_apHorizonViewContext[iHorizon] = pVC;

    pVC->SetLineRGBAColor(RGBAColor(255,255,255,64));

    pVC->SetHorizon(pHorizonSI);
    pVC->SetTransferFunction(pHorizonTF);

    m_nHorizonVDS++;
  }
}

void HorizonSurfaceWidget::CreateSeismicTransferFunctions( Hue::ProxyLib::TransferFunction * pSeismicTF0, Hue::ProxyLib::VDS * pSeismicVDS, Hue::ProxyLib::TransferFunction * pSeismicTF1, Hue::ProxyLib::TransferFunction * pSeismicTFSlice, Hue::ProxyLib::TransferFunction * pDistanceFieldTF)
{
  pSeismicTF0->SetColorValueRange(DoubleRange(pSeismicVDS->ValueRange().Min, pSeismicVDS->ValueRange().Max));
  pSeismicTF0->SetAlphaValueRange(DoubleRange(pSeismicVDS->ValueRange().Min, pSeismicVDS->ValueRange().Max));
  pSeismicTF0->SetReadoutMode(TransferFunctionReadoutMode::SpecifiedRange);

  pSeismicTF1->SetColorValueRange(DoubleRange(pSeismicVDS->ValueRange().Min, pSeismicVDS->ValueRange().Max));
  pSeismicTF1->SetAlphaValueRange(DoubleRange(pSeismicVDS->ValueRange().Min, pSeismicVDS->ValueRange().Max));
  pSeismicTF1->SetReadoutMode(TransferFunctionReadoutMode::SpecifiedRange);

  pSeismicTFSlice->SetColorValueRange(DoubleRange(pSeismicVDS->ValueRange().Min, pSeismicVDS->ValueRange().Max));
  pSeismicTFSlice->SetAlphaValueRange(DoubleRange(pSeismicVDS->ValueRange().Min, pSeismicVDS->ValueRange().Max));
  pSeismicTFSlice->SetReadoutMode(TransferFunctionReadoutMode::SpecifiedRange);

  AddBasicSeismicColors(pSeismicTF0);
  AddBasicSeismicColors(pSeismicTF1);
  AddBasicSeismicColors(pSeismicTFSlice);

  SetSeismicTFAlphaMagic(pSeismicTF0, 1.0f);
  SetSeismicTFAlphaMagic(pSeismicTF1, 0.0f);

  m_pSeismicTFInside = pSeismicTF0;
  m_pSeismicTFOutside = pSeismicTF1;

  // Create Distance field transferfunction
  pDistanceFieldTF->ColorHandles().Clear();
  for (int i=0; i<512; i++)
  {
    float
      rPos = (float)i / 512.0;

    RGBColor
      cColor((i&1) * 127 + 127, (i&2) * 63 + 127, (i&4) * 31 + 127);

    pDistanceFieldTF->ColorHandles().Add(ColorHandle(rPos, cColor, cColor));
  }
  pDistanceFieldTF->SetTransferFunctionResolution(TransferFunctionResolution::Resolution12Bit);
  pDistanceFieldTF->SetColorValueRange(DoubleRange(-10000,10000));
  pDistanceFieldTF->SetAlphaValueRange(DoubleRange(-10000,10000));
  pDistanceFieldTF->SetReadoutMode(TransferFunctionReadoutMode::SpecifiedRange);

}

void HorizonSurfaceWidget::SetSeismicTFAlphaMagic( Hue::ProxyLib::TransferFunction * pSeismicTF0, float rAlpha0)
{
  float 
    rAlpha1 = rAlpha0;

  float
    rFac = 0.5f;

  float
    rOffset0 = rFac - rAlpha0;

  if (rOffset0 < 0.0) rOffset0 = 0.0;

  rAlpha0 *= 5.0;
  if (rAlpha0 > 1.0) rAlpha0 = 1.0;

  rAlpha1 -= 0.6f;
  if (rAlpha1 < 0.0f) rAlpha1 = 0;

  rAlpha1 /= 1.0 - 0.6f;
  rAlpha1 *= rAlpha1;

  pSeismicTF0->AlphaHandles().Clear();
  pSeismicTF0->AlphaHandles().Add(AlphaHandle(0.450f - rOffset0,rAlpha0, rAlpha0));
  pSeismicTF0->AlphaHandles().Add(AlphaHandle(0.451f - rOffset0,rAlpha1, rAlpha1));
  pSeismicTF0->AlphaHandles().Add(AlphaHandle(0.549f + rOffset0,rAlpha1, rAlpha1));
  pSeismicTF0->AlphaHandles().Add(AlphaHandle(0.550f + rOffset0,rAlpha0, rAlpha0));

//   rAlpha *= rAlpha;
//   pSeismicTF0->AlphaHandles().Clear();
//   pSeismicTF0->AlphaHandles().Add(AlphaHandle(0.47f,1.0f, rAlpha));
//   pSeismicTF0->AlphaHandles().Add(AlphaHandle(0.53f,rAlpha, 1.0f));
}

void HorizonSurfaceWidget::CreateDistanceFieldVMProgramOnRenderState(Hue::ProxyLib::VolumeRS * pVolumeRS, bool isGradientShading, bool isDistanceFunction, bool isSecondDataset, bool isOutside)
{
  if (pVolumeRS->VMVolumeProgram())
  {
    pVolumeRS->VMVolumeProgram()->Delete();
  }
  
  Hue::ProxyLib::VMVolumeProgram
    *myVMProgram = m_pProject->VolumeVMPrograms().Create("DistanceVMProgram");

  pVolumeRS->SetVMVolumeProgram(myVMProgram);

  Hue::ProxyLib::VMParameter
    *paramDistVolumeBox = myVMProgram->Parameters().Create("DistanceVolumeBox", VMTypeInVMVariable::VMTypeVolumeBoxReference),
    *paramSeismicVolumeBox = myVMProgram->Parameters().Create("SeismicVolumeBox", VMTypeInVMVariable::VMTypeVolumeBoxReference),
    *paramTF0 = myVMProgram->Parameters().Create("TF0", VMTypeInVMVariable::VMTypeTransferFunctionReference),
    *paramTF1 = myVMProgram->Parameters().Create("TF1", VMTypeInVMVariable::VMTypeTransferFunctionReference),
    *paramThreshold = myVMProgram->Parameters().Create("Threshold", VMTypeInVMVariable::VMTypeFloat,  VectorElement::VMVectorElement1),
    *paramTF2,
    *paramSeismicVolumeBox2;
  
  if (isDistanceFunction)
  {
     paramTF2 = myVMProgram->Parameters().Create("TF2", VMTypeInVMVariable::VMTypeTransferFunctionReference);
  }

  if (isSecondDataset)
  {
    paramSeismicVolumeBox2 = myVMProgram->Parameters().Create("SeismicVolumeBox2", VMTypeInVMVariable::VMTypeVolumeBoxReference);
  }

  Hue::ProxyLib::VMSystemVariable
    *resultRGBA = myVMProgram->SystemVariables().FindByName("_RGBA");

  Hue::ProxyLib::VMTempVariable
    *tempV0 = myVMProgram->TempVariables().Create("v0", VMTypeInVMVariable::VMTypeFloat, VectorElement::VMVectorElement1),
    *tempV1 = myVMProgram->TempVariables().Create("v1", VMTypeInVMVariable::VMTypeFloat, VectorElement::VMVectorElement1);

  myVMProgram->Commands().CreateVMCommandReadValue(tempV0, paramDistVolumeBox);

  if (isSecondDataset)
  {
    Hue::ProxyLib::VMCommandIfGreater
      *commandGreater = myVMProgram->Commands().CreateVMCommandIfGreater(tempV0, paramThreshold);
    
    commandGreater->Then().CreateVMCommandReadValue(tempV1, paramSeismicVolumeBox2);
    commandGreater->Then().CreateVMCommandTransferFunctionLookup(resultRGBA, paramTF0, tempV1);

    if (isOutside)
    {
      commandGreater->Else().CreateVMCommandReadValue(tempV1, paramSeismicVolumeBox);
      commandGreater->Else().CreateVMCommandTransferFunctionLookup(resultRGBA, paramTF1, tempV1);
    }
  }
  else
  {
    myVMProgram->Commands().CreateVMCommandReadValue(tempV1, paramSeismicVolumeBox);
    
    Hue::ProxyLib::VMCommandIfGreater
      *commandGreater = myVMProgram->Commands().CreateVMCommandIfGreater(tempV0, paramThreshold);

    commandGreater->Then().CreateVMCommandTransferFunctionLookup(resultRGBA, paramTF0, tempV1);

    if (isOutside)
    {
      commandGreater->Else().CreateVMCommandTransferFunctionLookup(resultRGBA, paramTF1, tempV1);
    }
  }

  if (isDistanceFunction)
  {
    Hue::ProxyLib::VMTempVariable
      *tempColor = myVMProgram->TempVariables().Create("distcolor", VMTypeInVMVariable::VMTypeFloat, VectorElement::VMVectorElement4);

    myVMProgram->Commands().CreateVMCommandTransferFunctionLookup(tempColor, paramTF2, tempV0);
    myVMProgram->Commands().CreateVMCommandMultiply(resultRGBA, resultRGBA, tempColor);
  }

  if (isGradientShading)
  {
    Hue::ProxyLib::VMConstant
      *constantZero = myVMProgram->Constants().Create("constantRange", VMTypeInVMVariable::VMTypeFloat, VectorElement::VMVectorElement1);

    constantZero->SetValue0(0);

    Hue::ProxyLib::VMCommandIfGreater
      *commandVisible = myVMProgram->Commands().CreateVMCommandIfGreater(resultRGBA->SubVariables().FindByName("_RGBA[3]"), constantZero);

    Hue::ProxyLib::VMConstant
      *constantRange = myVMProgram->Constants().Create("constantRange", VMTypeInVMVariable::VMTypeFloat, VectorElement::VMVectorElement1),
      *constantOne = myVMProgram->Constants().Create("constantRange", VMTypeInVMVariable::VMTypeFloat, VectorElement::VMVectorElement1);

    constantRange->SetValue0(1.0f / 100.0f);
    constantOne->SetValue0(1.0f);

    commandVisible->Then().CreateVMCommandSubtract(tempV0, tempV0, paramThreshold);
    commandVisible->Then().CreateVMCommandAbs(tempV0,tempV0);
    commandVisible->Then().CreateVMCommandMultiply(tempV0, tempV0, constantRange);
   
    Hue::ProxyLib::VMCommandIfGreater
      *commandGreater =commandVisible->Then().CreateVMCommandIfGreater(tempV0, constantOne);

    commandGreater->Then().CreateVMCommandCalculateAndApplyShadingIntensity(resultRGBA,resultRGBA,paramSeismicVolumeBox);
    commandGreater->Else().CreateVMCommandCalculateAndApplyShadingIntensity(resultRGBA,resultRGBA,paramDistVolumeBox);
  }
}

void HorizonSurfaceWidget::AddBasicSeismicColors( Hue::ProxyLib::TransferFunction * pSeismicTF)
{
  pSeismicTF->ColorHandles().Clear();
  pSeismicTF->ColorHandles().Add(ColorHandle(0.0f, RGBColor(0,255,0), (RGBColor(0,255,0))));
  pSeismicTF->ColorHandles().Add(ColorHandle(0.30f, RGBColor(121,121,255), (RGBColor(121,121,255))));
  pSeismicTF->ColorHandles().Add(ColorHandle(0.5f, RGBColor(255,255,255), (RGBColor(255,255,255))));
  pSeismicTF->ColorHandles().Add(ColorHandle(0.70f, RGBColor(198,0,0), (RGBColor(198,00,0))));
  pSeismicTF->ColorHandles().Add(ColorHandle(1.0f, RGBColor(255,255,0), (RGBColor(255,255,0))));
}
