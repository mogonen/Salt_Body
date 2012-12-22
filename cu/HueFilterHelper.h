//////////////////////////////////////////////////////////////////////////////
//
// HueFilterHelper.h

#ifndef _HUEFILTERHELPER_H
#define _HUEFILTERHELPER_H

// TYPEDEFS /////////////////////////////////////////////////////////////////
#define HUEFILTERHELPER_INPUT_MAX  16
#define HUEFILTERHELPER_TEMP_MAX   16
#define HUEFILTERHELPER_OUTPUT_MAX  16

#define HUEFILTERHELPER_DIMENSIONALITY_MAX 6

typedef unsigned short ushort;
typedef unsigned char uchar;

#ifndef  HUEFILTERHELPER_CUDACOMPILE
class uchar2
{
public:
  unsigned char
    x,y;
};

class uchar4
{
public:
  unsigned char
    x,y,z,w;
};

class ushort2
{
public:
  unsigned short
    x,y;
};

class ushort4
{
public:
  unsigned short
    x,y,z,w;
};

class float2
{
public:
  float
    x,y;
};

class float4
{
public:
  float
    x,y,z,w;
};
#endif

// shared object visibillity settings for gcc
#ifdef USE_GCC_VISIBILITY
  #define HUEFILTERHELPER_EXPORT_SYMBOLS __attribute__ ((visibility("hidden")))
#else
  #define HUEFILTERHELPER_EXPORT_SYMBOLS
#endif

enum HueFilterHelper_Dimension_e
{
  HUEFILTERHELPER_DIMENSION0 = 0,
  HUEFILTERHELPER_DIMENSION1 = 1,
  HUEFILTERHELPER_DIMENSION2 = 2,
  HUEFILTERHELPER_DIMENSION3 = 3,
  HUEFILTERHELPER_DIMENSION4 = 4,
  HUEFILTERHELPER_DIMENSION5 = 5,
  HUEFILTERHELPER_DIMENSION_ALL = 6
};

enum HueFilterHelper_Format_e
{
  HUEFILTERHELPER_FORMAT_BYTE = 1,
  HUEFILTERHELPER_FORMAT_WORD = 2,
  HUEFILTERHELPER_FORMAT_FLOAT = 4,
  HUEFILTERHELPER_FORMAT_1BIT = 8
};

enum HueFilterHelper_Components_e
{
  HUEFILTERHELPER_COMPONENTS_1 = 1,
  HUEFILTERHELPER_COMPONENTS_2 = 2,
  HUEFILTERHELPER_COMPONENTS_4 = 4
};

enum HueFilterHelper_Type_e
{
  HUEFILTERHELPER_INPUT,
  HUEFILTERHELPER_OUTPUT,
  HUEFILTERHELPER_TEMP,
  HUEFILTERHELPER_NONE
};

// CLASS ///////////////////////////////////////////////////////////////////

class HueFilterHelper_DataFormat_c
{
public:
  
  HueFilterHelper_Format_e
    _eFormat;
  HueFilterHelper_Components_e
    _eComponents;

  HueFilterHelper_DataFormat_c() {_eFormat = HUEFILTERHELPER_FORMAT_FLOAT; _eComponents = HUEFILTERHELPER_COMPONENTS_1;}
  HueFilterHelper_DataFormat_c(HueFilterHelper_Format_e eFormat, HueFilterHelper_Components_e eComponents) {_eFormat = eFormat; _eComponents = eComponents;}

};

class HueFilterHelper
{

public:
  HueFilterHelper(void *pxFilterPlugin,
                  void ** papxOutputVolumeDataCacheItem, int nOutputVolumeDataCacheItem,
                  const void ** papxInputVolumeDataCacheItem, int nInputVolumeDataCacheItem, void *pxContext, bool isCuda = true);

  ~HueFilterHelper();

  void          CreateTempBuffer(int iTempBuffer, const int *pnInputMin, const int *pnInputMax, const HueFilterHelper_DataFormat_c cFormat = HueFilterHelper_DataFormat_c(HUEFILTERHELPER_FORMAT_FLOAT, HUEFILTERHELPER_COMPONENTS_1));
  void          CopyToTempBuffer(int iTempBuffer, void *pxData, int nElements);

  int           _aanInputMin[HUEFILTERHELPER_INPUT_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanInputMax[HUEFILTERHELPER_INPUT_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanInputPitch[HUEFILTERHELPER_INPUT_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanInputBitPitch[HUEFILTERHELPER_INPUT_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanTempMin[HUEFILTERHELPER_TEMP_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanTempMax[HUEFILTERHELPER_TEMP_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanTempPitch[HUEFILTERHELPER_TEMP_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanTempBitPitch[HUEFILTERHELPER_TEMP_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanOutputMin[HUEFILTERHELPER_OUTPUT_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanOutputMax[HUEFILTERHELPER_OUTPUT_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanOutputPitch[HUEFILTERHELPER_OUTPUT_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _aanOutputBitPitch[HUEFILTERHELPER_OUTPUT_MAX][HUEFILTERHELPER_DIMENSIONALITY_MAX];

//protected:

//private:
  bool          _isCUDA;

  int           _iProcessingUnit;

  void         *_pxDataBlockFactory;

  HueFilterHelper_DataFormat_c
                _acInputDataFormat[HUEFILTERHELPER_INPUT_MAX],
                _acTempDataFormat[HUEFILTERHELPER_TEMP_MAX],
                _acOutputDataFormat[HUEFILTERHELPER_OUTPUT_MAX];

  bool          _aisInputAllocated[HUEFILTERHELPER_INPUT_MAX];

  void          *_apxInputData[HUEFILTERHELPER_INPUT_MAX];

  void          *_apxTempData[HUEFILTERHELPER_TEMP_MAX];

  int           _nTempData;

  void          *_apxOutputData[HUEFILTERHELPER_OUTPUT_MAX];

  int           _nOutputData;

  HueFilterHelper_DataFormat_c
                GetFormat(HueFilterHelper_Type_e eInput, int iInput);
  int           GetElementByteSize(HueFilterHelper_Type_e eInput, int iInput);

  const int *   GetMin(HueFilterHelper_Type_e eInput, int iInput);
  const int *   GetMax(HueFilterHelper_Type_e eInput, int iInput);


};

class HueFilterHelper_ProcessArea
{
private:

public:
  HueFilterHelper
                *_pcFilterHelper;

  int           _anProcessMin[HUEFILTERHELPER_DIMENSIONALITY_MAX],
                _anProcessMax[HUEFILTERHELPER_DIMENSIONALITY_MAX];

  HueFilterHelper_Type_e 
                _eInput[HUEFILTERHELPER_INPUT_MAX],
                _eOutput[HUEFILTERHELPER_INPUT_MAX];

  int           _iInput[HUEFILTERHELPER_INPUT_MAX],
                _iOutput[HUEFILTERHELPER_INPUT_MAX];

                HueFilterHelper_ProcessArea(HueFilterHelper *pcFilterHelper);

  void          SetInput(int iInput, HueFilterHelper_Type_e eInputBuffer, int iInputBuffer)
                {
                  _eInput[iInput] = eInputBuffer;
                  _iInput[iInput] = iInputBuffer;
                }

  void          SetOutput(int iOutput, HueFilterHelper_Type_e eOutputBuffer, int iOutputBuffer)
                {
                  _eOutput[iOutput] = eOutputBuffer;
                  _iOutput[iOutput] = iOutputBuffer;
                }

  void          Set(HueFilterHelper_Dimension_e eDimension, HueFilterHelper_Type_e eType, int iInputOutput);
  void          SetProcessArea(int *pnMin, int *pnMax)
                {
                  for (int i=0;i<6;i++)
                  {
                    this->_anProcessMin[i] = pnMin[i];
                    this->_anProcessMax[i] = pnMax[i];
                  }
                }

  void          Add(int nMin0, int nMin1, int nMin2, int nMin4, int nMin5, int nMin6,
                    int nMax0, int nMax1, int nMax2, int nMax4, int nMax5, int nMax6);
  //void          RunFilter(int iFilterProgram, int iFilterData, int iOutput, int iInput);

};

extern "C" void  HueFilterHelper_GetArea(int *pnOutputMin, int *pnOutputMax, HueFilterHelper * pcFilterHelper, HueFilterHelper_Type_e eOutput, int iOutput);
extern "C" void * HueFilterHelper_GetPointer(HueFilterHelper * pcFilterHelper, HueFilterHelper_Type_e eOutput, int iOutput);
extern "C" void  HueFilterHelper_GetPitch(int *pnOutputPitch, int *pnOutputBitPitch, HueFilterHelper * pcFilterHelper, HueFilterHelper_Type_e eOutput, int iOutput);

#ifndef NULL
#define NULL 0
#endif

class HUEFILTERHELPER_EXPORT_SYMBOLS HueFilterHelper_Iterator
{
public:

  void
    *_apxInput[HUEFILTERHELPER_INPUT_MAX],
    *_apxOutput[HUEFILTERHELPER_INPUT_MAX];

  int
    _anProcessMin[6],
    _anProcessMax[6],
    _aanInputMin[HUEFILTERHELPER_INPUT_MAX][6],
    _aanInputMax[HUEFILTERHELPER_INPUT_MAX][6],
    _aanOutputMin[HUEFILTERHELPER_INPUT_MAX][6],
	_aanOutputMax[HUEFILTERHELPER_INPUT_MAX][6];

  int
    _aanInputPitch[HUEFILTERHELPER_INPUT_MAX][6],
    _aanInputBitPitch[HUEFILTERHELPER_INPUT_MAX][6],
    _aanOutputPitch[HUEFILTERHELPER_INPUT_MAX][6],
    _aanOutputBitPitch[HUEFILTERHELPER_INPUT_MAX][6];

  void Clear()
  {
    for (int i=0;i < HUEFILTERHELPER_INPUT_MAX;i++)
    {
      _apxInput[i] = NULL;
      _apxOutput[i] = NULL;
    }
  }
};

#endif // _HUEFILTERHELPER_H
