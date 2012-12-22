/////////////////////////////////////////////////////////////////////////////
// HueFilterHelper
// (c) Hue AS 2008

// INCLUDES /////////////////////////////////////////////////////////////////

#include <malloc.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <algorithm>

#include "HueComponentInterfaces.h"
#include "PluginInterface.h"
#include "HueSpaceLib.h"
#include "HueFilterHelper.h"

// DEFINES //////////////////////////////////////////////////////////////////

// CLASSES //////////////////////////////////////////////////////////////////

template <class TYPE>
static void
HueFilterHelper_CopyArea(TYPE *pTarget, const int *pnTargetPitch, const TYPE * pInput, const int *pnInputPitch, const int *pnSize)
{
  for (int i = 0; i < pnSize[2]; i++)
  {
    for (int j = 0; j < pnSize[1]; j++)
    {
      for (int k = 0; k < pnSize[0]; k++)
      {
        pTarget[i * pnTargetPitch[2] + j * pnTargetPitch[1] + k * pnTargetPitch[0]] = pInput[i * pnInputPitch[2] + j * pnInputPitch[1] + k * pnInputPitch[0]];
      }
    }
  }
}

extern "C" void HueFilterHelper_CudaCopyFromCPU(void *pxCudaDst, void *pxCPUSrc, int nSize);

extern "C" void HueFilterHelper_CudaCopyAreaU8(unsigned char *pTarget, const int *pnTargetPitch, const unsigned char * pInput, const int *pnInputPitch, const int *pnSize);
extern "C" void HueFilterHelper_CudaCopyAreaU16(unsigned short *pTarget, const int *pnTargetPitch, const unsigned short * pInput, const int *pnInputPitch, const int *pnSize);
extern "C" void HueFilterHelper_CudaCopyAreaR32(float *pTarget, const int *pnTargetPitch, const float * pInput, const int *pnInputPitch, const int *pnSize);

static HueFilterHelper_DataFormat_c
HueFilterHelper_GetDataFormat(Hue::HueSpaceLib::DataBlock::Format eFormat, Hue::HueSpaceLib::DataBlock::Components eComponents)
{
  HueFilterHelper_DataFormat_c
    cFormat;

  if (eFormat == Hue::HueSpaceLib::DataBlock::FORMAT_1BIT)
  {
    cFormat._eFormat = HUEFILTERHELPER_FORMAT_1BIT;
  }
  else if (eFormat == Hue::HueSpaceLib::DataBlock::FORMAT_U8)
  {
    cFormat._eFormat = HUEFILTERHELPER_FORMAT_BYTE;
  }
  else if (eFormat == Hue::HueSpaceLib::DataBlock::FORMAT_U16)
  {
    cFormat._eFormat = HUEFILTERHELPER_FORMAT_WORD;
  }
  else if (eFormat == Hue::HueSpaceLib::DataBlock::FORMAT_R32)
  {
    cFormat._eFormat = HUEFILTERHELPER_FORMAT_FLOAT;
  }
  else
  {
    // Error - unsupported filter format;
    assert(0 && "unsupported filter format");
  }

  if (eComponents == Hue::HueSpaceLib::DataBlock::COMPONENTS_1)
  {
    cFormat._eComponents = HUEFILTERHELPER_COMPONENTS_1;
  }
  else if (eComponents == Hue::HueSpaceLib::DataBlock::COMPONENTS_2)
  {
    cFormat._eComponents = HUEFILTERHELPER_COMPONENTS_2;
  }
  else if (eComponents == Hue::HueSpaceLib::DataBlock::COMPONENTS_4)
  {
    cFormat._eComponents = HUEFILTERHELPER_COMPONENTS_4;
  }
  else
  {
    // Error - unsupported filter number of components;
    assert(0 && "unsupported filter number of components");
  }

  return cFormat;
}

static Hue::HueSpaceLib::DataBlock::Format
HueFilterHelper_GetDataBlockFormat(HueFilterHelper_DataFormat_c const &cFormat)
{
  switch(cFormat._eFormat)
  {
  default:
    assert(0 && "illegal format");
  case HUEFILTERHELPER_FORMAT_1BIT:
    return Hue::HueSpaceLib::DataBlock::FORMAT_1BIT;
  case HUEFILTERHELPER_FORMAT_BYTE:
    return Hue::HueSpaceLib::DataBlock::FORMAT_U8;
  case HUEFILTERHELPER_FORMAT_WORD:
    return Hue::HueSpaceLib::DataBlock::FORMAT_U16;
  case HUEFILTERHELPER_FORMAT_FLOAT:
    return Hue::HueSpaceLib::DataBlock::FORMAT_R32;
  }
}

// HueFilterHelper
HueFilterHelper::HueFilterHelper(void *pxFilterPlugin, void ** papxOutputVolumeDataCacheItem, int nOutputVolumeDataCacheItem, const void ** papxInputVolumeDataCacheItem, int nInputVolumeDataCacheItem, void *pxContext, bool isCuda)
{
  Hue::HueSpaceLib::VDSFilterPlugin::Context
    *pcContext = (Hue::HueSpaceLib::VDSFilterPlugin::Context *)pxContext;

  Hue::HueSpaceLib::VDSFilterPlugin
    *pcFilterPlugin = (Hue::HueSpaceLib::VDSFilterPlugin *)pxFilterPlugin;

  Hue::HueSpaceLib::DataBlockFactory
    *pcDataBlockFactory = pcFilterPlugin->GetDataBlockFactory();

  _pxDataBlockFactory = (void *)pcDataBlockFactory;

  // Check if we want to use cuda.
  _isCUDA = isCuda && pcDataBlockFactory->IsCUDASupported();
  //_isCUDA = false;

  _iProcessingUnit = pcContext->GetProcessingUnit();

  _nOutputData = 0;
  _nTempData = 0;

  const Hue::HueSpaceLib::VolumeDataCacheItem **
    papcInputVolumeDataCacheItem = (const Hue::HueSpaceLib::VolumeDataCacheItem **)papxInputVolumeDataCacheItem;

  Hue::HueSpaceLib::VolumeDataCacheItem **
    papcOutputVolumeDataCacheItem = (Hue::HueSpaceLib::VolumeDataCacheItem **)papxOutputVolumeDataCacheItem;

  for (int iInput = 0; iInput < HUEFILTERHELPER_INPUT_MAX; iInput++)
  {
    _aisInputAllocated[iInput] = false;
    _apxInputData[iInput] = NULL;

    for (int iDim = 0; iDim < HUEFILTERHELPER_DIMENSIONALITY_MAX; iDim++)
    {
      _aanInputMin[iInput][iDim] = 0;
      _aanInputMax[iInput][iDim] = 1;
      _aanInputPitch[iInput][iDim] = 0;
      _aanInputBitPitch[iInput][iDim] = 0;
    }
  }

  for (int iTemp = 0; iTemp < HUEFILTERHELPER_TEMP_MAX; iTemp++)
  {
    _apxTempData[iTemp] = NULL;

    for (int iDim = 0; iDim < HUEFILTERHELPER_DIMENSIONALITY_MAX; iDim++)
    {
      _aanTempMin[iTemp][iDim] = 0;
      _aanTempMax[iTemp][iDim] = 1;
      _aanTempPitch[iTemp][iDim] = 0;
      _aanTempBitPitch[iTemp][iDim] = 0;
    }
  }

  assert(nOutputVolumeDataCacheItem <= HUEFILTERHELPER_OUTPUT_MAX);
  _nOutputData = nOutputVolumeDataCacheItem;

  for (int iOutput = 0; iOutput < HUEFILTERHELPER_OUTPUT_MAX; iOutput++)
  {
    _apxOutputData[iOutput] = NULL;

    for (int iDim = 0; iDim < HUEFILTERHELPER_DIMENSIONALITY_MAX; iDim++)
    {
      _aanOutputMin[iOutput][iDim] = 0;
      _aanOutputMax[iOutput][iDim] = 1;
      _aanOutputPitch[iOutput][iDim] = 0;
      _aanOutputBitPitch[iOutput][iDim] = 0;
    }
  }

  // Find all input items coming from the same layout and channel
  int
    aiInputCacheItemStart[HUEFILTERHELPER_INPUT_MAX],
    aiInputCacheItemEnd[HUEFILTERHELPER_INPUT_MAX];

  for(int iInputChannel = 0; iInputChannel < HUEFILTERHELPER_INPUT_MAX; iInputChannel++)
  {
    aiInputCacheItemStart[iInputChannel] = -1;
    aiInputCacheItemEnd[iInputChannel] = -1;
  }

  for (int iInputCacheItemStart = 0, iInputCacheItemEnd = 1; iInputCacheItemEnd <= nInputVolumeDataCacheItem; iInputCacheItemStart = iInputCacheItemEnd++)
  {
    const Hue::HueSpaceLib::VolumeDataLayout *
      pcVolumeDataLayout = papcInputVolumeDataCacheItem[iInputCacheItemStart]->GetDataChunk().GetLayout();

    int
      iInputChannel = pcContext->GetInputChannelForInputVolumeDataCacheItem(iInputCacheItemStart);

    while(iInputCacheItemEnd < nInputVolumeDataCacheItem && pcContext->GetInputChannelForInputVolumeDataCacheItem(iInputCacheItemEnd) == iInputChannel)
    {
      iInputCacheItemEnd++;
    }

    aiInputCacheItemStart[iInputChannel] = iInputCacheItemStart;
    aiInputCacheItemEnd[iInputChannel] = iInputCacheItemEnd;
  }

  // Go through inputs and determine the (combined) min/max
  int
    aanCombinedInputMin[HUEFILTERHELPER_INPUT_MAX][Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX],
    aanCombinedInputMax[HUEFILTERHELPER_INPUT_MAX][Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX],
    aanCombinedInputMinExcludingMargin[HUEFILTERHELPER_INPUT_MAX][Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX],
    aanCombinedInputMaxExcludingMargin[HUEFILTERHELPER_INPUT_MAX][Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX];

  bool
    aisCreateCombinedBuffer[HUEFILTERHELPER_INPUT_MAX];

  int
    aiInputVDS[HUEFILTERHELPER_INPUT_MAX];

  for(int iInputChannel = 0; iInputChannel < HUEFILTERHELPER_INPUT_MAX; iInputChannel++)
  {
    if(aiInputCacheItemStart[iInputChannel] == -1)
    {
      continue;
    }

    Hue::HueSpaceLib::InputChannelDescriptor
      cInputChannelDescriptor = pcFilterPlugin->DescribeInputChannel(iInputChannel);

    _acInputDataFormat[iInputChannel] = HueFilterHelper_GetDataFormat(cInputChannelDescriptor.GetFormat(), cInputChannelDescriptor.GetComponents());

    aiInputVDS[iInputChannel] = cInputChannelDescriptor.GetInputVDSIndex();

    aisCreateCombinedBuffer[iInputChannel] = false;

    Hue::HueSpaceLib::VolumeDataCacheItem const *
      pcInputCacheItem = papcInputVolumeDataCacheItem[aiInputCacheItemStart[iInputChannel]];

    pcInputCacheItem->GetDataChunk().GetMinMax(aanCombinedInputMin[iInputChannel], aanCombinedInputMax[iInputChannel]);
    pcInputCacheItem->GetDataChunk().GetMinMaxExcludingMargin(aanCombinedInputMinExcludingMargin[iInputChannel], aanCombinedInputMaxExcludingMargin[iInputChannel]);

    for (int iInputCacheItem = aiInputCacheItemStart[iInputChannel] + 1; iInputCacheItem < aiInputCacheItemEnd[iInputChannel]; iInputCacheItem++)
    {
      pcInputCacheItem = papcInputVolumeDataCacheItem[iInputCacheItem];
      aisCreateCombinedBuffer[iInputChannel] = true;

      int
        anInputMin[Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX],
        anInputMax[Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX],
        anInputMinExcludingMargin[Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX],
        anInputMaxExcludingMargin[Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX];

      pcInputCacheItem->GetDataChunk().GetMinMax(anInputMin, anInputMax);
      pcInputCacheItem->GetDataChunk().GetMinMaxExcludingMargin(anInputMinExcludingMargin, anInputMaxExcludingMargin);

      for (int iDimension = 0; iDimension < Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX; iDimension++)
      {
        aanCombinedInputMin[iInputChannel][iDimension] = std::min(aanCombinedInputMin[iInputChannel][iDimension], anInputMin[iDimension]);
        aanCombinedInputMax[iInputChannel][iDimension] = std::max(aanCombinedInputMax[iInputChannel][iDimension], anInputMax[iDimension]);
        aanCombinedInputMinExcludingMargin[iInputChannel][iDimension] = std::min(aanCombinedInputMinExcludingMargin[iInputChannel][iDimension], anInputMinExcludingMargin[iDimension]);
        aanCombinedInputMaxExcludingMargin[iInputChannel][iDimension] = std::max(aanCombinedInputMaxExcludingMargin[iInputChannel][iDimension], anInputMaxExcludingMargin[iDimension]);
      }
    }

    // Correct size for window because we can expand margins

    const Hue::HueSpaceLib::VolumeDataLayout *
      pcVolumeDataLayout = pcInputCacheItem->GetDataChunk().GetLayout();

    // find if input datablock is larger than 1 << 27 size, if so, switch to cpu.
    int
      nVoxels = 1;

    if(!aisCreateCombinedBuffer[iInputChannel])
    {
      // Only one input
      Hue::HueSpaceLib::VolumeDataCacheItem const *
        pcInputCacheItem = papcInputVolumeDataCacheItem[aiInputCacheItemStart[iInputChannel]];

      const Hue::HueSpaceLib::DataBlock
        &cInputDataBlock = pcInputCacheItem->GetDataBlock();

      int
        anAllocatedSize[Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX];

      cInputDataBlock.GetAllocatedSize(anAllocatedSize[0], anAllocatedSize[1], anAllocatedSize[2], anAllocatedSize[3]);

      for (int iDimension = 0; iDimension < Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX; iDimension++)
      {
        nVoxels *= anAllocatedSize[iDimension];
      }
    }
    else
    {
      for (int iDimension = 0; iDimension < Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX; iDimension++)
      {
        nVoxels *= aanCombinedInputMax[iInputChannel][iDimension] - aanCombinedInputMin[iInputChannel][iDimension];
      }
    }

    if (nVoxels > (1 << 27))
    {
	    _isCUDA =false;
    }
  }

  // Get buffers
  for(int iInputChannel = 0; iInputChannel < HUEFILTERHELPER_INPUT_MAX; iInputChannel++)
  {
    if(aiInputCacheItemStart[iInputChannel] == -1)
    {
      continue;
    }

    Hue::HueSpaceLib::DataBlock::Format
      eInputFormat = papcInputVolumeDataCacheItem[aiInputCacheItemStart[iInputChannel]]->GetDataBlock().GetFormat();

    int
      anCombinedInputPitch[Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX],
      anCombinedInputBitPitch[Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX];

    if(!aisCreateCombinedBuffer[iInputChannel])
    {
      // Only one input
      Hue::HueSpaceLib::VolumeDataCacheItem const *
        pcInputCacheItem = papcInputVolumeDataCacheItem[aiInputCacheItemStart[iInputChannel]];

      const Hue::HueSpaceLib::DataBlock
        &cInputDataBlock = pcInputCacheItem->GetDataBlock();

      if (!_isCUDA)
      {
        _apxInputData[iInputChannel] = const_cast<void *>(cInputDataBlock.GetBufferCPUReadOnly(HueFilterHelper_GetDataBlockFormat(_acInputDataFormat[iInputChannel])) );
      }
      else
      {
        _apxInputData[iInputChannel] = const_cast<void *>(cInputDataBlock.GetBufferCUDAReadOnly(cInputDataBlock.GetFormat()));
      }

      cInputDataBlock.GetPitch(anCombinedInputPitch[0], anCombinedInputPitch[1], anCombinedInputPitch[2], anCombinedInputPitch[3]);
      for (int iDimension = 0; iDimension < Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX; iDimension++)
      {
        anCombinedInputBitPitch[iDimension] = anCombinedInputPitch[iDimension] * (iDimension == 0 ? 1 : 8);
      }
    }
    else
    {
      // Allocate input
      int
        nCombinedInputPitch = 1;

      for(int iDataBlockDimension = 0; iDataBlockDimension < Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX; iDataBlockDimension++)
      {
        anCombinedInputPitch[iDataBlockDimension] = nCombinedInputPitch;
        anCombinedInputBitPitch[iDataBlockDimension] = anCombinedInputPitch[iDataBlockDimension] * (iDataBlockDimension == 0 ? 1 : 8);

        int iDimension = papcInputVolumeDataCacheItem[aiInputCacheItemStart[iInputChannel]]->GetDataChunk().GetDimension(iDataBlockDimension);

        if(iDimension >= 0 && iDimension < HUEFILTERHELPER_DIMENSIONALITY_MAX)
        {
          nCombinedInputPitch *= aanCombinedInputMax[iInputChannel][iDimension] - aanCombinedInputMin[iInputChannel][iDimension];
        }

        if (eInputFormat == Hue::HueSpaceLib::DataBlock::FORMAT_1BIT && iDataBlockDimension == 0)
        {
          nCombinedInputPitch = (nCombinedInputPitch + 7) / 8;
        }
      }

      int nByteSize = nCombinedInputPitch;
      if (eInputFormat == Hue::HueSpaceLib::DataBlock::FORMAT_U16)      nByteSize *= 2;
      else if (eInputFormat == Hue::HueSpaceLib::DataBlock::FORMAT_R32) nByteSize *= 4;
 
      if(_isCUDA)
      {
        Hue::HueSpaceLib::DataBlockFactory
          *pcDataBlockFactory = (Hue::HueSpaceLib::DataBlockFactory *)_pxDataBlockFactory;

        _apxInputData[iInputChannel] = pcDataBlockFactory->AllocateCUDAMemory(nByteSize);
      }
      else
      {
        _apxInputData[iInputChannel] = malloc(nByteSize);
      }

      // Set that is item is allocated
      _aisInputAllocated[iInputChannel] = true;

      // copy inputs to combined input
      for (int iInputCacheItem = aiInputCacheItemStart[iInputChannel]; iInputCacheItem < aiInputCacheItemEnd[iInputChannel]; iInputCacheItem++)
      {
        Hue::HueSpaceLib::VolumeDataChunk const
          &cInputDataChunk = papcInputVolumeDataCacheItem[iInputCacheItem]->GetDataChunk();

        Hue::HueSpaceLib::DataBlock const
          &cInputDataBlock = papcInputVolumeDataCacheItem[iInputCacheItem]->GetDataBlock();

        int
          anInputMin[Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX],
          anInputMax[Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX],
          anInputMinExcludingMargin[Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX],
          anInputMaxExcludingMargin[Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX];

        int
          anOverlapMin[HUEFILTERHELPER_DIMENSIONALITY_MAX], anOverlapMax[HUEFILTERHELPER_DIMENSIONALITY_MAX];

        cInputDataChunk.GetMinMax(anInputMin, anInputMax);
        cInputDataChunk.GetMinMaxExcludingMargin(anInputMinExcludingMargin, anInputMaxExcludingMargin);

        for(int iDimension = 0; iDimension < HUEFILTERHELPER_DIMENSIONALITY_MAX; iDimension++)
        {
          int
            nInputMinOverlap = anInputMin[iDimension],
            nInputMaxOverlap = anInputMax[iDimension];

          // Only copy margins from source at TARGETS margins
          if (anInputMinExcludingMargin[iDimension] > aanCombinedInputMinExcludingMargin[iInputChannel][iDimension])
          {
            nInputMinOverlap = anInputMinExcludingMargin[iDimension];
          }

          if (anInputMaxExcludingMargin[iDimension] < aanCombinedInputMaxExcludingMargin[iInputChannel][iDimension])
          {
            nInputMaxOverlap = anInputMaxExcludingMargin[iDimension];
          }

          anOverlapMin[iDimension] = std::max(aanCombinedInputMin[iInputChannel][iDimension], nInputMinOverlap);
          anOverlapMax[iDimension] = std::min(aanCombinedInputMax[iInputChannel][iDimension], nInputMaxOverlap);
        }

        int
          anInputPitch[Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX],
          anTargetPitch[Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX];

        int
          anOverlapSize[Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX];

        int
          nInputOffset = 0, nTargetOffset = 0;

        cInputDataBlock.GetPitch(anInputPitch[0], anInputPitch[1], anInputPitch[2], anInputPitch[3]);

        for (int i = 0; i < Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX; i++)
        {
          int
            iDimension = cInputDataChunk.GetDimension(i);

          if(iDimension != -1)
          {
            anTargetPitch[i] = anCombinedInputPitch[i];
            anOverlapSize[i] = anOverlapMax[iDimension] - anOverlapMin[iDimension];
            nInputOffset += (anOverlapMin[iDimension] - anInputMin[iDimension]) * anInputPitch[i];
            nTargetOffset += (anOverlapMin[iDimension] - aanCombinedInputMin[iInputChannel][iDimension]) * anTargetPitch[i];
          }
          else
          {
            anTargetPitch[i] = 0;
            anOverlapSize[i] = 1;
          }
        }

        if(_isCUDA)
        {
          if (eInputFormat == Hue::HueSpaceLib::DataBlock::FORMAT_U8)
          {
            HueFilterHelper_CudaCopyAreaU8(((unsigned char *)_apxInputData[iInputChannel]) + nTargetOffset, anTargetPitch, cInputDataBlock.GetBufferU8CUDAReadOnly() + nInputOffset, anInputPitch, anOverlapSize);
          }
          else if (eInputFormat == Hue::HueSpaceLib::DataBlock::FORMAT_U16)
          {
            HueFilterHelper_CudaCopyAreaU16(((unsigned short *)_apxInputData[iInputChannel]) + nTargetOffset, anTargetPitch, cInputDataBlock.GetBufferU16CUDAReadOnly() + nInputOffset, anInputPitch, anOverlapSize);
          }
          else if (eInputFormat == Hue::HueSpaceLib::DataBlock::FORMAT_R32)
          {
            HueFilterHelper_CudaCopyAreaR32(((float *)_apxInputData[iInputChannel]) + nTargetOffset, anTargetPitch, cInputDataBlock.GetBufferR32CUDAReadOnly() + nInputOffset, anInputPitch, anOverlapSize);
          }
        }
        else
        {
          if (eInputFormat == Hue::HueSpaceLib::DataBlock::FORMAT_U8)
          {
            HueFilterHelper_CopyArea(((unsigned char *)_apxInputData[iInputChannel]) + nTargetOffset, anTargetPitch, cInputDataBlock.GetBufferU8CPUReadOnly() + nInputOffset, anInputPitch, anOverlapSize);
          }
          else if (eInputFormat == Hue::HueSpaceLib::DataBlock::FORMAT_U16)
          {
            HueFilterHelper_CopyArea(((unsigned short *)_apxInputData[iInputChannel]) + nTargetOffset, anTargetPitch, cInputDataBlock.GetBufferU16CPUReadOnly() + nInputOffset, anInputPitch, anOverlapSize);
          }
          else if (eInputFormat == Hue::HueSpaceLib::DataBlock::FORMAT_R32)
          {
            HueFilterHelper_CopyArea(((float *)_apxInputData[iInputChannel]) + nTargetOffset, anTargetPitch, cInputDataBlock.GetBufferR32CPUReadOnly() + nInputOffset, anInputPitch, anOverlapSize);
          }
        }
      }
    }

    // Set up the correct pitch and offset information for the input block
    for(int iDataBlockDimension = 0; iDataBlockDimension < Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX; iDataBlockDimension++)
    {
      int iDimension = papcInputVolumeDataCacheItem[aiInputCacheItemStart[iInputChannel]]->GetDataChunk().GetDimension(iDataBlockDimension);
      if(iDimension >= 0 && iDimension < HUEFILTERHELPER_DIMENSIONALITY_MAX)
      {
        _aanInputPitch[iInputChannel][iDimension] = anCombinedInputPitch[iDataBlockDimension];
        _aanInputBitPitch[iInputChannel][iDimension] = anCombinedInputBitPitch[iDataBlockDimension];
      }
    }

    for (int iDimension = 0; iDimension < HUEFILTERHELPER_DIMENSIONALITY_MAX; iDimension++)
    {
      _aanInputMin[iInputChannel][iDimension] = aanCombinedInputMin[iInputChannel][iDimension];
      _aanInputMax[iInputChannel][iDimension] = aanCombinedInputMax[iInputChannel][iDimension];
    }

    int
      anLayoutSize[HUEFILTERHELPER_DIMENSIONALITY_MAX];

    for (int i=0; i < HUEFILTERHELPER_DIMENSIONALITY_MAX; i++)
    {
      anLayoutSize[i] = papcInputVolumeDataCacheItem[aiInputCacheItemStart[iInputChannel]]->GetDataChunk().GetLayout()->GetDimensionNumSamples(i);
    }

    int
      anInputOffset[Hue::HueSpaceLib::VolumeDataLayout::DIMENSIONALITY_MAX];

    pcContext->GetInputOffset(aiInputVDS[iInputChannel], anInputOffset);

    for (int iDimension = 0; iDimension < HUEFILTERHELPER_DIMENSIONALITY_MAX; iDimension++)
    {
      _aanInputMin[iInputChannel][iDimension] -= anInputOffset[iDimension];
      _aanInputMax[iInputChannel][iDimension] -= anInputOffset[iDimension];
    }
  }

  // Go through outputs

  for (int iOutput = 0; iOutput < _nOutputData; iOutput++)
  {
    Hue::HueSpaceLib::VolumeDataCacheItem *
      pcVolumeDataCacheItem = (Hue::HueSpaceLib::VolumeDataCacheItem *)papcOutputVolumeDataCacheItem[iOutput];

    pcVolumeDataCacheItem->GetDataChunk().GetMinMax(_aanOutputMin[iOutput], _aanOutputMax[iOutput]);

    Hue::HueSpaceLib::DataBlock *
      pcOutputDataBlock = pcVolumeDataCacheItem->CreateDataBlock();

    _acOutputDataFormat[iOutput] = HueFilterHelper_GetDataFormat(pcOutputDataBlock->GetFormat(), pcOutputDataBlock->GetComponents());

    int
      anOutputDataBlockPitch[Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX];

    pcOutputDataBlock->GetPitch(anOutputDataBlockPitch[0], anOutputDataBlockPitch[1], anOutputDataBlockPitch[2], anOutputDataBlockPitch[3]);

    for(int iDataBlockDimension = 0; iDataBlockDimension < Hue::HueSpaceLib::DataBlock::DIMENSIONALITY_MAX; iDataBlockDimension++)
    {
      int iDimension = papcOutputVolumeDataCacheItem[iOutput]->GetDataChunk().GetDimension(iDataBlockDimension);
      if(iDimension >= 0 && iDimension < HUEFILTERHELPER_DIMENSIONALITY_MAX)
      {
        _aanOutputPitch[iOutput][iDimension] = anOutputDataBlockPitch[iDataBlockDimension];
        _aanOutputBitPitch[iOutput][iDimension] = anOutputDataBlockPitch[iDataBlockDimension] * (iDataBlockDimension == 0 ? 1 : 8);
      }
    }

    if (!_isCUDA)
    {
      _apxOutputData[iOutput] = pcOutputDataBlock->GetBufferCPU(pcOutputDataBlock->GetFormat());
    }
    else
    {
      _apxOutputData[iOutput] = pcOutputDataBlock->GetBufferCUDA(pcOutputDataBlock->GetFormat());
    }
  }
}


//~HueFilterHelper()
HueFilterHelper::~HueFilterHelper()
{
  Hue::HueSpaceLib::DataBlockFactory
    *pcDataBlockFactory = (Hue::HueSpaceLib::DataBlockFactory *)_pxDataBlockFactory;

  for (int i=0; i<HUEFILTERHELPER_INPUT_MAX; i++)
  {
    if (_aisInputAllocated[i])
    {
      if (_isCUDA)
      {
        pcDataBlockFactory->FreeCUDAMemory(_apxInputData[i]);
      }
      else
      {
        free(_apxInputData[i]);
      }
    }
  }

  for (int i = 0; i < HUEFILTERHELPER_TEMP_MAX; i++)
  {
    if (_apxTempData[i])
    {
      if (_isCUDA)
      {
        pcDataBlockFactory->FreeCUDAMemory(_apxTempData[i]);
      }
      else
      {
        free(_apxTempData[i]);
      }
    }
  }
}

// HueFilterHelper::CreateTempBuffer
void
HueFilterHelper::CreateTempBuffer(int iTempBuffer, const int *pnInputMin, const int *pnInputMax, HueFilterHelper_DataFormat_c cFormat)
{
  assert(!_apxTempData[iTempBuffer] && "Can't create the same temp buffer twice!");

  this->_acTempDataFormat[iTempBuffer] = cFormat;

  int
    nTempPitch = 1,
    nTempBitPitch = 1;

  for (int i = 0; i < HUEFILTERHELPER_DIMENSIONALITY_MAX; i++)
  {
    _aanTempMin[iTempBuffer][i] = pnInputMin[i];
    _aanTempMax[iTempBuffer][i] = pnInputMax[i];

    _aanTempPitch[iTempBuffer][i] = nTempPitch;
    _aanTempBitPitch[iTempBuffer][i] = nTempBitPitch;

    if(pnInputMax[i] - pnInputMin[i] > 1)
    {
      nTempPitch *= pnInputMax[i] - pnInputMin[i];
      if(cFormat._eFormat == HUEFILTERHELPER_FORMAT_1BIT && nTempBitPitch == 1)
      {
        nTempPitch = (nTempPitch + 7) / 8;
      }
      nTempBitPitch = nTempPitch * 8;
    }
  }

  int nByteSize = nTempPitch;
  if (cFormat._eFormat == HUEFILTERHELPER_FORMAT_WORD)       nByteSize *= 2;
  else if (cFormat._eFormat == HUEFILTERHELPER_FORMAT_FLOAT) nByteSize *= 4;
  nByteSize *= cFormat._eComponents;

  if (_isCUDA)
  {
    Hue::HueSpaceLib::DataBlockFactory
      *pcDataBlockFactory = (Hue::HueSpaceLib::DataBlockFactory *)_pxDataBlockFactory;

    _apxTempData[iTempBuffer] = pcDataBlockFactory->AllocateCUDAMemory(nByteSize);
  }
  else
  {
    _apxTempData[iTempBuffer] = malloc(nByteSize);
  }

  _nTempData++;
}

HueFilterHelper_ProcessArea::HueFilterHelper_ProcessArea(HueFilterHelper *pcFilterHelper)
{
  _pcFilterHelper = pcFilterHelper;

  for (int i = 0; i < HUEFILTERHELPER_INPUT_MAX; i++)
  {
    _eInput[i] = HUEFILTERHELPER_NONE;
    _eOutput[i] = HUEFILTERHELPER_NONE;
    _iInput[i] = i;
    _iOutput[i] = i;
  }

  Set(HUEFILTERHELPER_DIMENSION_ALL, HUEFILTERHELPER_OUTPUT, 0);
}



// Changes the area you process next

void
HueFilterHelper_ProcessArea::Add(int nMin0, int nMin1, int nMin2, int nMin3, int nMin4, int nMin5,
                                 int nMax0, int nMax1, int nMax2, int nMax3, int nMax4, int nMax5)
{
  _anProcessMin[0] += nMin0;
  _anProcessMin[1] += nMin1;
  _anProcessMin[2] += nMin2;
  _anProcessMin[3] += nMin3;
  _anProcessMin[4] += nMin4;
  _anProcessMin[5] += nMin5;
  
  _anProcessMax[0] += nMax0;
  _anProcessMax[1] += nMax1;
  _anProcessMax[2] += nMax2;
  _anProcessMax[3] += nMax3;
  _anProcessMax[4] += nMax4;
  _anProcessMax[5] += nMax5;
}

// HueFilterHelper_ProcessArea::Set

void
HueFilterHelper_ProcessArea::Set(HueFilterHelper_Dimension_e eDimension, HueFilterHelper_Type_e eType, int iInputOutput)
{
  int
    *pnMin,
    *pnMax;

  if (eType == HUEFILTERHELPER_INPUT)
  {
    pnMin = _pcFilterHelper->_aanInputMin[iInputOutput];
    pnMax = _pcFilterHelper->_aanInputMax[iInputOutput];
  }
  else if (eType == HUEFILTERHELPER_TEMP)
  {
    pnMin = _pcFilterHelper->_aanTempMin[iInputOutput];
    pnMax = _pcFilterHelper->_aanTempMax[iInputOutput];
  }
  else if (eType == HUEFILTERHELPER_OUTPUT)
  {
    pnMin = _pcFilterHelper->_aanOutputMin[iInputOutput];
    pnMax = _pcFilterHelper->_aanOutputMax[iInputOutput];
  }
  else
  {
    assert(0);
  }

  for (int i = 0; i < HUEFILTERHELPER_DIMENSIONALITY_MAX; i++)
  {
    if ((int)eDimension == i ||
        eDimension == HUEFILTERHELPER_DIMENSION_ALL)
    {
      _anProcessMin[i] = pnMin[i];
      _anProcessMax[i] = pnMax[i];
    }
  }
}

void
HueFilterHelper::CopyToTempBuffer(int iTempBuffer, void *pxData, int nElements)
{
  int
    nByteSize = (int)this->_acTempDataFormat[iTempBuffer]._eFormat *
                     this->_acTempDataFormat[iTempBuffer]._eComponents;

  if (_isCUDA)
  {
    HueFilterHelper_CudaCopyFromCPU(_apxTempData[iTempBuffer], pxData, nElements * nByteSize);
  }
  else
  {
    memcpy(_apxTempData[iTempBuffer], pxData, nElements * nByteSize);
  }
}


HueFilterHelper_DataFormat_c
HueFilterHelper::GetFormat(HueFilterHelper_Type_e eInput, int iInput)
{
  HueFilterHelper_DataFormat_c
    cFormat;

  if (eInput == HUEFILTERHELPER_OUTPUT)
  {
    cFormat = _acOutputDataFormat[iInput];
  }
  else if (eInput == HUEFILTERHELPER_TEMP)
  {
    cFormat = _acTempDataFormat[iInput];
  }
  else if (eInput == HUEFILTERHELPER_INPUT) 
  {
    cFormat = _acInputDataFormat[iInput];
  }

  return cFormat;
}

const int *   
HueFilterHelper::GetMin(HueFilterHelper_Type_e eInput, int iInput)
{
  if (eInput == HUEFILTERHELPER_OUTPUT)
  {
    return this->_aanOutputMin[iInput];
  }
  else if (eInput == HUEFILTERHELPER_TEMP)
  {
    return this->_aanTempMin[iInput];
  }
  else if (eInput == HUEFILTERHELPER_INPUT) 
  {
    return this->_aanInputMin[iInput];
  }

  assert(0);
  return NULL;
}

const int *   
HueFilterHelper::GetMax(HueFilterHelper_Type_e eInput, int iInput)
{
  if (eInput == HUEFILTERHELPER_OUTPUT)
  {
    return this->_aanOutputMax[iInput];
  }
  else if (eInput == HUEFILTERHELPER_TEMP)
  {
    return this->_aanTempMax[iInput];
  }
  else if (eInput == HUEFILTERHELPER_INPUT) 
  {
    return this->_aanInputMax[iInput];
  }

  assert(0);
  return NULL;
}

int
HueFilterHelper::GetElementByteSize(HueFilterHelper_Type_e eInput, int iInput)
{
  HueFilterHelper_DataFormat_c
    cFormat = GetFormat(eInput,iInput);

  return cFormat._eFormat * cFormat._eComponents;
}
