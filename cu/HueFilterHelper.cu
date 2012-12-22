#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define HUEFILTERHELPER_CUDACOMPILE

#include "HueFilterHelper.h"

struct HueFilterHelper_CopyInfo_c
{
  int _anTargetPitch[6];
  int _anInputPitch[6];
  int _anSize[6];
};

__constant__ HueFilterHelper_CopyInfo_c
  cKernelCopyInfo;

template <class TYPE>
__global__ void
HueFilterHelper_CopyAreaCUDAKernel(TYPE *pTarget, const TYPE *pInput, int nYPitch)
{
  int
    anPos[4];

  anPos[0] =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  anPos[2] = blockIdx.y / nYPitch;

  int nBlockIdY = blockIdx.y - (anPos[2] * nYPitch);

  anPos[1] = __mul24(nBlockIdY,blockDim.y) + threadIdx.y;

  //Only if inside buffer
  if (anPos[0] < cKernelCopyInfo._anSize[0] &&
      anPos[1] < cKernelCopyInfo._anSize[1] &&
      anPos[2] < cKernelCopyInfo._anSize[2])
  {
    int
      nReadOffset = anPos[0] * cKernelCopyInfo._anInputPitch[0] +
                    anPos[1] * cKernelCopyInfo._anInputPitch[1] +
                    anPos[2] * cKernelCopyInfo._anInputPitch[2];

    int
      nWriteOffset = anPos[0] * cKernelCopyInfo._anTargetPitch[0] +
                     anPos[1] * cKernelCopyInfo._anTargetPitch[1] +
                     anPos[2] * cKernelCopyInfo._anTargetPitch[2];

    pTarget[nWriteOffset] = pInput[nReadOffset];
  }
}

extern "C" void
HueFilterHelper_CudaCopyFromCPU(void *pxCudaDst, void *pxCPUSrc, int nSize)
{
  cudaMemcpy(pxCudaDst, pxCPUSrc, nSize, cudaMemcpyHostToDevice);
}

extern "C" void *
HueFilterHelper_GetPointer(HueFilterHelper * pcFilterHelper, HueFilterHelper_Type_e eOutput, int iOutput)
{
  assert(iOutput < HUEFILTERHELPER_INPUT_MAX);

  if (eOutput == HUEFILTERHELPER_OUTPUT) return pcFilterHelper->_apxOutputData[iOutput];
  else if (eOutput == HUEFILTERHELPER_TEMP) return pcFilterHelper->_apxTempData[iOutput];
  else if (eOutput == HUEFILTERHELPER_INPUT) return pcFilterHelper->_apxInputData[iOutput];
  else assert(0);

  return NULL;
}

extern "C" void
HueFilterHelper_GetPitch(int *pnOutputPitch, int *pnOutputBitPitch, HueFilterHelper * pcFilterHelper, HueFilterHelper_Type_e eOutput, int iOutput)
{
  assert(iOutput < HUEFILTERHELPER_INPUT_MAX);

  int *pnPitch = NULL,
      *pnBitPitch = NULL;

  if (eOutput == HUEFILTERHELPER_OUTPUT)
  {
    pnPitch = pcFilterHelper->_aanOutputPitch[iOutput];
    pnBitPitch = pcFilterHelper->_aanOutputBitPitch[iOutput];
  }
  else if (eOutput == HUEFILTERHELPER_TEMP)
  {
    pnPitch = pcFilterHelper->_aanTempPitch[iOutput];
    pnBitPitch = pcFilterHelper->_aanTempBitPitch[iOutput];
  }
  else if (eOutput == HUEFILTERHELPER_INPUT)
  {
    pnPitch = pcFilterHelper->_aanInputPitch[iOutput];
    pnBitPitch = pcFilterHelper->_aanInputBitPitch[iOutput];
  }
  else assert(0);

  for (int i = 0; i < HUEFILTERHELPER_DIMENSIONALITY_MAX; i++)
  {
    pnOutputPitch[i] = pnPitch[i];
    pnOutputBitPitch[i] = pnBitPitch[i];
  }
}

extern "C" void
HueFilterHelper_GetArea(int *pnOutputMin, int *pnOutputMax, HueFilterHelper * pcFilterHelper, HueFilterHelper_Type_e eOutput, int iOutput)
{
  int
    *pnMin = NULL,
    *pnMax = NULL;

  if (eOutput == HUEFILTERHELPER_OUTPUT)
  {
    pnMin = pcFilterHelper->_aanOutputMin[iOutput];
    pnMax = pcFilterHelper->_aanOutputMax[iOutput];
  }
  else if (eOutput == HUEFILTERHELPER_TEMP)
  {
    pnMin = pcFilterHelper->_aanTempMin[iOutput];
    pnMax = pcFilterHelper->_aanTempMax[iOutput];
  }
  else if (eOutput == HUEFILTERHELPER_INPUT)
  {
    pnMin = pcFilterHelper->_aanInputMin[iOutput];
    pnMax = pcFilterHelper->_aanInputMax[iOutput];
  }
  else assert(0);

  for (int i = 0; i < HUEFILTERHELPER_DIMENSIONALITY_MAX; i++)
  {
    pnOutputMin[i] = pnMin[i];
    pnOutputMax[i] = pnMax[i];
  }
}


extern "C" cudaChannelFormatDesc
HueFilterHelper_GetChannelDesc(HueFilterHelper_ProcessArea * pProcessArea,  HueFilterHelper_Type_e eInput, int iInput)
{
  cudaChannelFormatDesc
    cChannelDesc;

  HueFilterHelper_DataFormat_c
    cFormat;

  if (eInput == HUEFILTERHELPER_OUTPUT)
  {
    cFormat = pProcessArea->_pcFilterHelper->_acOutputDataFormat[iInput];
  }
  else if (eInput == HUEFILTERHELPER_TEMP)
  {
    cFormat = pProcessArea->_pcFilterHelper->_acTempDataFormat[iInput];
  }
  else if (eInput == HUEFILTERHELPER_INPUT)
  {
    cFormat = pProcessArea->_pcFilterHelper->_acInputDataFormat[iInput];
  }

  int
    nSize0 = 32;

  if (cFormat._eFormat == HUEFILTERHELPER_FORMAT_1BIT ||
      cFormat._eFormat == HUEFILTERHELPER_FORMAT_BYTE) nSize0 = 8;
  else if (cFormat._eFormat == HUEFILTERHELPER_FORMAT_WORD) nSize0 = 16;


  int
    nSize1 = 0,
    nSize2 = 0,
    nSize3 = 0;

  if (cFormat._eComponents == HUEFILTERHELPER_COMPONENTS_2)
  {
    nSize1 = nSize0;
  }
  else if (cFormat._eComponents == HUEFILTERHELPER_COMPONENTS_4)
  {
    nSize1 = nSize0;
    nSize2 = nSize0;
    nSize3 = nSize0;
  }

  if (cFormat._eFormat == HUEFILTERHELPER_FORMAT_FLOAT)
  {
    cChannelDesc = cudaCreateChannelDesc(nSize0, nSize1, nSize2, nSize3, cudaChannelFormatKindFloat);
  }
  else
  {
    cChannelDesc = cudaCreateChannelDesc(nSize0, nSize1, nSize2, nSize3, cudaChannelFormatKindUnsigned);
  }

  return cChannelDesc;
}

void
HueFilterHelper_CudaCopyArea(void *pTarget, const int *pnTargetPitch, const void * pInput, const int *pnInputPitch, const int *pnSize, int nElementSize)
{
  HueFilterHelper_CopyInfo_c
    cCopyInfo;

  for(int iDimension = 0; iDimension < 6; iDimension++)
  {
    cCopyInfo._anTargetPitch[iDimension] = pnTargetPitch[iDimension];
    cCopyInfo._anInputPitch[iDimension] = pnInputPitch[iDimension];
    cCopyInfo._anSize[iDimension] = pnSize[iDimension];
  }

  cudaMemcpyToSymbol(cKernelCopyInfo, &cCopyInfo, sizeof(HueFilterHelper_CopyInfo_c));

  dim3
    cGridSize,
    cBlockSize;

  cBlockSize.x = 16;
  cBlockSize.y = 16;
  cBlockSize.z = 1;

  cGridSize.x = pnSize[0];
  cGridSize.y = pnSize[1];
  cGridSize.z = 1;

  cGridSize.x = (cGridSize.x + cBlockSize.x - 1) / cBlockSize.x;
  cGridSize.y = (cGridSize.y + cBlockSize.y - 1) / cBlockSize.y;
  cGridSize.z = 1;

  int nYPitch = cGridSize.y;

  cGridSize.y *= pnSize[2];
  
  switch(nElementSize)
  {
  case 1:
    HueFilterHelper_CopyAreaCUDAKernel<<<cGridSize, cBlockSize>>>((unsigned char *)pTarget, (const unsigned char *)pInput, nYPitch);
    break;
  case 2:
    HueFilterHelper_CopyAreaCUDAKernel<<<cGridSize, cBlockSize>>>((unsigned short *)pTarget, (const unsigned short *)pInput, nYPitch);
    break;
  case 4:
    HueFilterHelper_CopyAreaCUDAKernel<<<cGridSize, cBlockSize>>>((float *)pTarget, (const float *)pInput, nYPitch);
    break;
  }
}

extern "C" void
HueFilterHelper_CudaCopyAreaU8(unsigned char *pTarget, const int *pnTargetPitch, const unsigned char * pInput, const int *pnInputPitch, const int *pnSize)
{
  HueFilterHelper_CudaCopyArea(pTarget, pnTargetPitch, pInput, pnInputPitch, pnSize, sizeof(char));
}

extern "C" void
HueFilterHelper_CudaCopyAreaU16(unsigned short *pTarget, const int *pnTargetPitch, const unsigned short * pInput, const int *pnInputPitch, const int *pnSize)
{
  HueFilterHelper_CudaCopyArea(pTarget, pnTargetPitch, pInput, pnInputPitch, pnSize, sizeof(short));
}

extern "C" void
HueFilterHelper_CudaCopyAreaR32(float *pTarget, const int *pnTargetPitch, const float * pInput, const int *pnInputPitch, const int *pnSize)
{
  HueFilterHelper_CudaCopyArea(pTarget, pnTargetPitch, pInput, pnInputPitch, pnSize, sizeof(float));
}
