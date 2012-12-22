#include <assert.h>

#ifdef  HUEFILTERHELPER_CUDACOMPILE

#define HUEFILTERHELPER_VARIABLE __constant__
#define HUEFILTERHELPER_FUNCTION __device__

#define HF_INPUTINDEX(__input,__dim0,__dim1,__dim2,__dim3) \
  (__mul24(__cIterator._aanInputPitch[__input][0], __dim0 - __cIterator._aanInputMin[__input][0]) + \
   __mul24(__cIterator._aanInputPitch[__input][1], __dim1 - __cIterator._aanInputMin[__input][1]) + \
   __mul24(__cIterator._aanInputPitch[__input][2], __dim2 - __cIterator._aanInputMin[__input][2]) + \
   __mul24(__cIterator._aanInputPitch[__input][3], __dim3 - __cIterator._aanInputMin[__input][3]))

#define HF_INPUTBITINDEX(__input,__dim0,__dim1,__dim2,__dim3) \
  (__mul24(__cIterator._aanInputBitPitch[__input][0], __dim0 - __cIterator._aanInputMin[__input][0]) + \
   __mul24(__cIterator._aanInputBitPitch[__input][1], __dim1 - __cIterator._aanInputMin[__input][1]) + \
   __mul24(__cIterator._aanInputBitPitch[__input][2], __dim2 - __cIterator._aanInputMin[__input][2]) + \
   __mul24(__cIterator._aanInputBitPitch[__input][3], __dim3 - __cIterator._aanInputMin[__input][3]))

#define HF_OUTPUTINDEX(__output,__dim0,__dim1,__dim2,__dim3) \
  (__mul24(__cIterator._aanOutputPitch[__output][0], __dim0 - __cIterator._aanOutputMin[__output][0]) + \
   __mul24(__cIterator._aanOutputPitch[__output][1], __dim1 - __cIterator._aanOutputMin[__output][1]) + \
   __mul24(__cIterator._aanOutputPitch[__output][2], __dim2 - __cIterator._aanOutputMin[__output][2]) + \
   __mul24(__cIterator._aanOutputPitch[__output][3], __dim3 - __cIterator._aanOutputMin[__output][3]))

#define HF_READ(__type, __input, __index) \
  (tex1Dfetch(cudaKernel_InputTexture##__input##__type, __index))

#define HUEFILTERHELPER_CONSTANTARRAY(__type,__name,__size) __constant__ __type __name[__size]; \
  extern "C" void SetConstantArray_##__name##CUDA(__type * __ptConstantArray) \
  {   \
    cudaMemcpyToSymbol(__name, __ptConstantArray, __size * sizeof(__type)); \
  }
extern "C" cudaChannelFormatDesc \
HueFilterHelper_GetChannelDesc(HueFilterHelper_ProcessArea * pProcessArea,  HueFilterHelper_Type_e eInput, int iInput);\

__constant__ HueFilterHelper_Iterator __cIterator; 

#ifdef HUEFILTERHELPER_USE_DATA_FLOAT

texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture0float;   
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture1float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture2float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture3float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture4float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture5float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture6float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture7float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture8float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture9float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture10float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture11float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture12float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture13float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture14float;                                                                  
texture<float,1,cudaReadModeElementType> cudaKernel_InputTexture15float;                                                                  

#endif 

#ifdef HUEFILTERHELPER_USE_DATA_FLOAT2

texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture0float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture1float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture2float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture3float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture4float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture5float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture6float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture7float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture8float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture9float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture10float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture11float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture12float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture13float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture14float2;                                                                  
texture<float2,1,cudaReadModeElementType> cudaKernel_InputTexture15float2;                                                                  

#endif

#ifdef HUEFILTERHELPER_USE_DATA_FLOAT4

texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture0float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture1float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture2float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture3float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture4float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture5float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture6float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture7float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture8float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture9float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture10float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture11float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture12float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture13float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture14float4;                                                                  
texture<float4,1,cudaReadModeElementType> cudaKernel_InputTexture15float4;                                                                  
#endif

#ifdef HUEFILTERHELPER_USE_DATA_USHORT

texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture0ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture1ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture2ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture3ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture4ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture5ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture6ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture7ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture8ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture9ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture10ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture11ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture12ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture13ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture14ushort;                                                                  
texture<ushort,1,cudaReadModeElementType> cudaKernel_InputTexture15ushort;                                                                  

#endif 

#ifdef HUEFILTERHELPER_USE_DATA_USHORT2

texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture0ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture1ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture2ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture3ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture4ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture5ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture6ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture7ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture8ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture9ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture10ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture11ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture12ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture13ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture14ushort2;                                                                  
texture<ushort2,1,cudaReadModeElementType> cudaKernel_InputTexture15ushort2;                                                                  

#endif

#ifdef HUEFILTERHELPER_USE_DATA_USHORT4

texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture0ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture1ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture2ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture3ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture4ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture5ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture6ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture7ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture8ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture9ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture10ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture11ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture12ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture13ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture14ushort4;                                                                  
texture<ushort4,1,cudaReadModeElementType> cudaKernel_InputTexture15ushort4;                                                                  

#endif

#ifdef HUEFILTERHELPER_USE_DATA_UCHAR

texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture0uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture1uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture2uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture3uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture4uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture5uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture6uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture7uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture8uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture9uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture10uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture11uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture12uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture13uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture14uchar;                                                                  
texture<uchar,1,cudaReadModeElementType> cudaKernel_InputTexture15uchar;                                                                  

#endif

#ifdef HUEFILTERHELPER_USE_DATA_UCHAR2

texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture0uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture1uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture2uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture3uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture4uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture5uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture6uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture7uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture8uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture9uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture10uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture11uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture12uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture13uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture14uchar2;                                                                  
texture<uchar2,1,cudaReadModeElementType> cudaKernel_InputTexture15uchar2;                                                                  

#endif

#ifdef HUEFILTERHELPER_USE_DATA_UCHAR4

texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture0uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture1uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture2uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture3uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture4uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture5uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture6uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture7uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture8uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture9uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture10uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture11uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture12uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture13uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture14uchar4;                                                                  
texture<uchar4,1,cudaReadModeElementType> cudaKernel_InputTexture15uchar4;                                                                  

#endif

static void
__BindTexture(int iInput,const HueFilterHelper_Iterator &cIterator, HueFilterHelper_ProcessArea * pProcessArea, bool isUnbind)
{ 
  if (!cIterator._apxInput[iInput])
  {
    return;
  }

  HueFilterHelper_DataFormat_c
    cFormat = pProcessArea->_pcFilterHelper->GetFormat(pProcessArea->_eInput[iInput], pProcessArea->_iInput[iInput]);

  cudaChannelFormatDesc 
    cChannelDesc = HueFilterHelper_GetChannelDesc(pProcessArea,  pProcessArea->_eInput[iInput], pProcessArea->_iInput[iInput]);

  void *
    pxInput = cIterator._apxInput[iInput];

  switch(cFormat._eFormat)
  {
    case HUEFILTERHELPER_FORMAT_1BIT:
    case HUEFILTERHELPER_FORMAT_BYTE:
      switch(cFormat._eComponents)
      {
        case HUEFILTERHELPER_COMPONENTS_1:

#ifdef HUEFILTERHELPER_USE_DATA_UCHAR
          switch(iInput)
          {
            case 0:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture0uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture0uchar, pxInput, cChannelDesc);
              break;
            case 1:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture1uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture1uchar, pxInput, cChannelDesc);
              break;
            case 2:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture2uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture2uchar, pxInput, cChannelDesc);
              break;
            case 3:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture3uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture3uchar, pxInput, cChannelDesc);
              break;
            case 4:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture4uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture4uchar, pxInput, cChannelDesc);
              break;
            case 5:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture5uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture5uchar, pxInput, cChannelDesc);
              break;
            case 6:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture6uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture6uchar, pxInput, cChannelDesc);
              break;
            case 7:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture7uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture7uchar, pxInput, cChannelDesc);
              break;
            case 8:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture8uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture8uchar, pxInput, cChannelDesc);
              break;
            case 9:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture9uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture9uchar, pxInput, cChannelDesc);
              break;
            case 10:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture10uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture10uchar, pxInput, cChannelDesc);
              break;
            case 11:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture11uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture11uchar, pxInput, cChannelDesc);
              break;
            case 12:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture12uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture12uchar, pxInput, cChannelDesc);
              break;
            case 13:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture13uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture13uchar, pxInput, cChannelDesc);
              break;
            case 14:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture14uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture14uchar, pxInput, cChannelDesc);
              break;
            case 15:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture15uchar);
              else          cudaBindTexture(0,cudaKernel_InputTexture15uchar, pxInput, cChannelDesc);
              break;
          }
#endif 
          
          break;

        case HUEFILTERHELPER_COMPONENTS_2:

#ifdef HUEFILTERHELPER_USE_DATA_UCHAR2

          switch(iInput)
          {
            case 0:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture0uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture0uchar2, pxInput, cChannelDesc);
              break;
            case 1:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture1uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture1uchar2, pxInput, cChannelDesc);
              break;
            case 2:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture2uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture2uchar2, pxInput, cChannelDesc);
              break;
            case 3:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture3uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture3uchar2, pxInput, cChannelDesc);
              break;
            case 4:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture4uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture4uchar2, pxInput, cChannelDesc);
              break;
            case 5:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture5uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture5uchar2, pxInput, cChannelDesc);
              break;
            case 6:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture6uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture6uchar2, pxInput, cChannelDesc);
              break;
            case 7:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture7uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture7uchar2, pxInput, cChannelDesc);
              break;
            case 8:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture8uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture8uchar2, pxInput, cChannelDesc);
              break;
            case 9:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture9uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture9uchar2, pxInput, cChannelDesc);
              break;
            case 10:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture10uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture10uchar2, pxInput, cChannelDesc);
              break;
            case 11:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture11uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture11uchar2, pxInput, cChannelDesc);
              break;
            case 12:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture12uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture12uchar2, pxInput, cChannelDesc);
              break;
            case 13:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture13uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture13uchar2, pxInput, cChannelDesc);
              break;
            case 14:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture14uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture14uchar2, pxInput, cChannelDesc);
              break;
            case 15:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture15uchar2);
              else          cudaBindTexture(0,cudaKernel_InputTexture15uchar2, pxInput, cChannelDesc);
              break;
          }
#endif
          
          break;

       case HUEFILTERHELPER_COMPONENTS_4:

#ifdef HUEFILTERHELPER_USE_DATA_UCHAR4

          switch(iInput)
          {
            case 0:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture0uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture0uchar4, pxInput, cChannelDesc);
              break;
            case 1:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture1uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture1uchar4, pxInput, cChannelDesc);
              break;
            case 2:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture2uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture2uchar4, pxInput, cChannelDesc);
              break;
            case 3:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture3uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture3uchar4, pxInput, cChannelDesc);
              break;
            case 4:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture4uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture4uchar4, pxInput, cChannelDesc);
              break;
            case 5:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture5uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture5uchar4, pxInput, cChannelDesc);
              break;
            case 6:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture6uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture6uchar4, pxInput, cChannelDesc);
              break;
            case 7:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture7uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture7uchar4, pxInput, cChannelDesc);
              break;
            case 8:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture8uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture8uchar4, pxInput, cChannelDesc);
              break;
            case 9:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture9uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture9uchar4, pxInput, cChannelDesc);
              break;
            case 10:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture10uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture10uchar4, pxInput, cChannelDesc);
              break;
            case 11:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture11uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture11uchar4, pxInput, cChannelDesc);
              break;
            case 12:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture12uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture12uchar4, pxInput, cChannelDesc);
              break;
            case 13:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture13uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture13uchar4, pxInput, cChannelDesc);
              break;
            case 14:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture14uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture14uchar4, pxInput, cChannelDesc);
              break;
            case 15:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture15uchar4);
              else          cudaBindTexture(0,cudaKernel_InputTexture15uchar4, pxInput, cChannelDesc);
              break;
          }
#endif
          
          break;

        default:
          assert(0);
      }
      break;

    case HUEFILTERHELPER_FORMAT_WORD:
      switch(cFormat._eComponents)
      {
        case HUEFILTERHELPER_COMPONENTS_1:

#ifdef HUEFILTERHELPER_USE_DATA_USHORT
          
          switch(iInput)
          {
            case 0:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture0ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture0ushort, pxInput, cChannelDesc);
              break;
            case 1:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture1ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture1ushort, pxInput, cChannelDesc);
              break;
            case 2:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture2ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture2ushort, pxInput, cChannelDesc);
              break;
            case 3:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture3ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture3ushort, pxInput, cChannelDesc);
              break;
            case 4:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture4ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture4ushort, pxInput, cChannelDesc);
              break;
            case 5:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture5ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture5ushort, pxInput, cChannelDesc);
              break;
            case 6:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture6ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture6ushort, pxInput, cChannelDesc);
              break;
            case 7:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture7ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture7ushort, pxInput, cChannelDesc);
              break;
            case 8:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture8ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture8ushort, pxInput, cChannelDesc);
              break;
            case 9:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture9ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture9ushort, pxInput, cChannelDesc);
              break;
            case 10:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture10ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture10ushort, pxInput, cChannelDesc);
              break;
            case 11:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture11ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture11ushort, pxInput, cChannelDesc);
              break;
            case 12:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture12ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture12ushort, pxInput, cChannelDesc);
              break;
            case 13:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture13ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture13ushort, pxInput, cChannelDesc);
              break;
            case 14:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture14ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture14ushort, pxInput, cChannelDesc);
              break;
            case 15:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture15ushort);
              else          cudaBindTexture(0,cudaKernel_InputTexture15ushort, pxInput, cChannelDesc);
              break;
          }
#endif
          
          break;

        case HUEFILTERHELPER_COMPONENTS_2:

#ifdef HUEFILTERHELPER_USE_DATA_USHORT2

          switch(iInput)
          {
            case 0:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture0ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture0ushort2, pxInput, cChannelDesc);
              break;
            case 1:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture1ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture1ushort2, pxInput, cChannelDesc);
              break;
            case 2:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture2ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture2ushort2, pxInput, cChannelDesc);
              break;
            case 3:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture3ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture3ushort2, pxInput, cChannelDesc);
              break;
            case 4:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture4ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture4ushort2, pxInput, cChannelDesc);
              break;
            case 5:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture5ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture5ushort2, pxInput, cChannelDesc);
              break;
            case 6:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture6ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture6ushort2, pxInput, cChannelDesc);
              break;
            case 7:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture7ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture7ushort2, pxInput, cChannelDesc);
              break;
            case 8:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture8ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture8ushort2, pxInput, cChannelDesc);
              break;
            case 9:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture9ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture9ushort2, pxInput, cChannelDesc);
              break;
            case 10:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture10ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture10ushort2, pxInput, cChannelDesc);
              break;
            case 11:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture11ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture11ushort2, pxInput, cChannelDesc);
              break;
            case 12:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture12ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture12ushort2, pxInput, cChannelDesc);
              break;
            case 13:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture13ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture13ushort2, pxInput, cChannelDesc);
              break;
            case 14:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture14ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture14ushort2, pxInput, cChannelDesc);
              break;
            case 15:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture15ushort2);
              else          cudaBindTexture(0,cudaKernel_InputTexture15ushort2, pxInput, cChannelDesc);
              break;
          }

#endif
          
          break;

       case HUEFILTERHELPER_COMPONENTS_4:

#ifdef HUEFILTERHELPER_USE_DATA_USHORT4

          switch(iInput)
          {
            case 0:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture0ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture0ushort4, pxInput, cChannelDesc);
              break;
            case 1:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture1ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture1ushort4, pxInput, cChannelDesc);
              break;
            case 2:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture2ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture2ushort4, pxInput, cChannelDesc);
              break;
            case 3:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture3ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture3ushort4, pxInput, cChannelDesc);
              break;
            case 4:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture4ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture4ushort4, pxInput, cChannelDesc);
              break;
            case 5:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture5ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture5ushort4, pxInput, cChannelDesc);
              break;
            case 6:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture6ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture6ushort4, pxInput, cChannelDesc);
              break;
            case 7:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture7ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture7ushort4, pxInput, cChannelDesc);
              break;
            case 8:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture8ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture8ushort4, pxInput, cChannelDesc);
              break;
            case 9:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture9ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture9ushort4, pxInput, cChannelDesc);
              break;
            case 10:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture10ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture10ushort4, pxInput, cChannelDesc);
              break;
            case 11:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture11ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture11ushort4, pxInput, cChannelDesc);
              break;
            case 12:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture12ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture12ushort4, pxInput, cChannelDesc);
              break;
            case 13:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture13ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture13ushort4, pxInput, cChannelDesc);
              break;
            case 14:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture14ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture14ushort4, pxInput, cChannelDesc);
              break;
            case 15:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture15ushort4);
              else          cudaBindTexture(0,cudaKernel_InputTexture15ushort4, pxInput, cChannelDesc);
              break;
          }

#endif
          
          break;

        default:
          assert(0);
      }
      
      break;

    case HUEFILTERHELPER_FORMAT_FLOAT:
      switch(cFormat._eComponents)
      {
        case HUEFILTERHELPER_COMPONENTS_1:

#ifdef HUEFILTERHELPER_USE_DATA_FLOAT
          switch(iInput)
          {
            case 0:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture0float);
              else          cudaBindTexture(0,cudaKernel_InputTexture0float, pxInput, cChannelDesc);
              break;
            case 1:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture1float);
              else          cudaBindTexture(0,cudaKernel_InputTexture1float, pxInput, cChannelDesc);
              break;
            case 2:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture2float);
              else          cudaBindTexture(0,cudaKernel_InputTexture2float, pxInput, cChannelDesc);
              break;
            case 3:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture3float);
              else          cudaBindTexture(0,cudaKernel_InputTexture3float, pxInput, cChannelDesc);
              break;
            case 4:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture4float);
              else          cudaBindTexture(0,cudaKernel_InputTexture4float, pxInput, cChannelDesc);
              break;
            case 5:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture5float);
              else          cudaBindTexture(0,cudaKernel_InputTexture5float, pxInput, cChannelDesc);
              break;
            case 6:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture6float);
              else          cudaBindTexture(0,cudaKernel_InputTexture6float, pxInput, cChannelDesc);
              break;
            case 7:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture7float);
              else          cudaBindTexture(0,cudaKernel_InputTexture7float, pxInput, cChannelDesc);
              break;
            case 8:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture8float);
              else          cudaBindTexture(0,cudaKernel_InputTexture8float, pxInput, cChannelDesc);
              break;
            case 9:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture9float);
              else          cudaBindTexture(0,cudaKernel_InputTexture9float, pxInput, cChannelDesc);
              break;
            case 10:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture10float);
              else          cudaBindTexture(0,cudaKernel_InputTexture10float, pxInput, cChannelDesc);
              break;
            case 11:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture11float);
              else          cudaBindTexture(0,cudaKernel_InputTexture11float, pxInput, cChannelDesc);
              break;
            case 12:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture12float);
              else          cudaBindTexture(0,cudaKernel_InputTexture12float, pxInput, cChannelDesc);
              break;
            case 13:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture13float);
              else          cudaBindTexture(0,cudaKernel_InputTexture13float, pxInput, cChannelDesc);
              break;
            case 14:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture14float);
              else          cudaBindTexture(0,cudaKernel_InputTexture14float, pxInput, cChannelDesc);
              break;
            case 15:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture15float);
              else          cudaBindTexture(0,cudaKernel_InputTexture15float, pxInput, cChannelDesc);
              break;
          }
#endif
          
          break;

        case HUEFILTERHELPER_COMPONENTS_2:

#ifdef HUEFILTERHELPER_USE_DATA_FLOAT2

          switch(iInput)
          {
            case 0:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture0float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture0float2, pxInput, cChannelDesc);
              break;
            case 1:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture1float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture1float2, pxInput, cChannelDesc);
              break;
            case 2:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture2float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture2float2, pxInput, cChannelDesc);
              break;
            case 3:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture3float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture3float2, pxInput, cChannelDesc);
              break;
            case 4:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture4float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture4float2, pxInput, cChannelDesc);
              break;
            case 5:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture5float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture5float2, pxInput, cChannelDesc);
              break;
            case 6:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture6float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture6float2, pxInput, cChannelDesc);
              break;
            case 7:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture7float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture7float2, pxInput, cChannelDesc);
              break;
            case 8:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture8float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture8float2, pxInput, cChannelDesc);
              break;
            case 9:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture9float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture9float2, pxInput, cChannelDesc);
              break;
            case 10:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture10float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture10float2, pxInput, cChannelDesc);
              break;
            case 11:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture11float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture11float2, pxInput, cChannelDesc);
              break;
            case 12:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture12float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture12float2, pxInput, cChannelDesc);
              break;
            case 13:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture13float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture13float2, pxInput, cChannelDesc);
              break;
            case 14:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture14float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture14float2, pxInput, cChannelDesc);
              break;
            case 15:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture15float2);
              else          cudaBindTexture(0,cudaKernel_InputTexture15float2, pxInput, cChannelDesc);
              break;
          }

#endif
          
          break;

       case HUEFILTERHELPER_COMPONENTS_4:

#ifdef HUEFILTERHELPER_USE_DATA_FLOAT4

          switch(iInput)
          {
            case 0:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture0float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture0float4, pxInput, cChannelDesc);
              break;
            case 1:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture1float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture1float4, pxInput, cChannelDesc);
              break;
            case 2:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture2float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture2float4, pxInput, cChannelDesc);
              break;
            case 3:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture3float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture3float4, pxInput, cChannelDesc);
              break;
            case 4:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture4float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture4float4, pxInput, cChannelDesc);
              break;
            case 5:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture5float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture5float4, pxInput, cChannelDesc);
              break;
            case 6:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture6float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture6float4, pxInput, cChannelDesc);
              break;
            case 7:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture7float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture7float4, pxInput, cChannelDesc);
              break;
            case 8:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture8float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture8float4, pxInput, cChannelDesc);
              break;
            case 9:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture9float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture9float4, pxInput, cChannelDesc);
              break;
            case 10:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture10float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture10float4, pxInput, cChannelDesc);
              break;
            case 11:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture11float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture11float4, pxInput, cChannelDesc);
              break;
            case 12:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture12float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture12float4, pxInput, cChannelDesc);
              break;
            case 13:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture13float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture13float4, pxInput, cChannelDesc);
              break;
            case 14:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture14float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture14float4, pxInput, cChannelDesc);
              break;
            case 15:
              if (isUnbind) cudaUnbindTexture(cudaKernel_InputTexture15float4);
              else          cudaBindTexture(0,cudaKernel_InputTexture15float4, pxInput, cChannelDesc);
              break;
          }

#endif
          
          break;

        default:
          assert(0);
          
      }
      break;
    
    default:
      assert(0);
  }
  
}


#define HUEFILTERHELPER_FUNCTION_BEGIN(FunctionName) \
__global__ void FunctionName##CUDAKernel(const int nDim2Pitch, const int nDim3Pitch); \
extern "C" void FunctionName##CUDA(HueFilterHelper_ProcessArea * pProcessArea) \
{                                                                                                                                               \
  HueFilterHelper_Iterator                                                                                                                 \
  __cIteratorCPU;                                                                                                                                  \
  __cIteratorCPU.Clear();                                                                                             \
  for (int __i = 0; __i < HUEFILTERHELPER_INPUT_MAX; __i++)  \
  { \
    if (pProcessArea->_eOutput[__i] != HUEFILTERHELPER_NONE) \
    { \
      __cIteratorCPU._apxOutput[__i] = HueFilterHelper_GetPointer(pProcessArea->_pcFilterHelper, pProcessArea->_eOutput[__i], pProcessArea->_iOutput[__i]);                                           \
      HueFilterHelper_GetArea(__cIteratorCPU._aanOutputMin[__i], __cIteratorCPU._aanOutputMax[__i], pProcessArea->_pcFilterHelper, pProcessArea->_eOutput[__i], pProcessArea->_iOutput[__i]);                                       \
      HueFilterHelper_GetPitch(__cIteratorCPU._aanOutputPitch[__i], __cIteratorCPU._aanOutputBitPitch[__i], pProcessArea->_pcFilterHelper, pProcessArea->_eOutput[__i], pProcessArea->_iOutput[__i]);                                      \
    } \
    if (pProcessArea->_eInput[__i] != HUEFILTERHELPER_NONE) \
    { \
      __cIteratorCPU._apxInput[__i] = HueFilterHelper_GetPointer(pProcessArea->_pcFilterHelper, pProcessArea->_eInput[__i], pProcessArea->_iInput[__i]);                                             \
      HueFilterHelper_GetArea(__cIteratorCPU._aanInputMin[__i], __cIteratorCPU._aanInputMax[__i], pProcessArea->_pcFilterHelper, pProcessArea->_eInput[__i], pProcessArea->_iInput[__i]);                                           \
      HueFilterHelper_GetPitch(__cIteratorCPU._aanInputPitch[__i], __cIteratorCPU._aanInputBitPitch[__i], pProcessArea->_pcFilterHelper, pProcessArea->_eInput[__i], pProcessArea->_iInput[__i]);                                         \
    } \
  }                                                                                                                                                \
                                                                                                                                                \
  for (int __i = 0; __i < 6; __i++)                                                                                                             \
  {                                                                                                                                             \
    __cIteratorCPU._anProcessMin[__i] = pProcessArea->_anProcessMin[__i];                                               \
    __cIteratorCPU._anProcessMax[__i] = pProcessArea->_anProcessMax[__i];                                               \
  }                                                                                                                                             \
                                                                                                                                                \
  dim3                                                                                                                                          \
    cGridSize,                                                                                                                                  \
    cBlockSize;                                                                                                                                 \
                                                                                                                                                \
  if((__cIteratorCPU._anProcessMax[0] - __cIteratorCPU._anProcessMin[0]) * (__cIteratorCPU._anProcessMax[2] - __cIteratorCPU._anProcessMin[2]) < 16) \
  {                                                                                                                                              \
    cBlockSize.x = 1;                                                                                                                            \
    cBlockSize.y = 128;                                                                                                                          \
  }                                                                                                                                              \
  else if((__cIteratorCPU._anProcessMax[1] - __cIteratorCPU._anProcessMin[1]) * (__cIteratorCPU._anProcessMax[3] - __cIteratorCPU._anProcessMin[3]) < 16) \
  {                                                                                                                                              \
    cBlockSize.x = 128;                                                                                                                          \
    cBlockSize.y = 1;                                                                                                                            \
  }                                                                                                                                              \
  else                                                                                                                                           \
  {                                                                                                                                              \
    cBlockSize.x = 16;                                                                                                                            \
    cBlockSize.y = 8;                                                                                                                            \
  }                                                                                                                                             \
  cBlockSize.z = 1;                                                                                                                             \
                                                                                                                                                \
  cGridSize.x = __cIteratorCPU._anProcessMax[0] - __cIteratorCPU._anProcessMin[0];                                                                                                         \
  cGridSize.y = __cIteratorCPU._anProcessMax[1] - __cIteratorCPU._anProcessMin[1];                                                                                                         \
                                                                                                                                                \
  cGridSize.x = (cGridSize.x + cBlockSize.x - 1) / cBlockSize.x;                                                                                \
  cGridSize.y = (cGridSize.y + cBlockSize.y - 1) / cBlockSize.y;                                                                                \
  cGridSize.z = 1;                                                                                                                              \
                                                                                                                                                \
  int nDim2Pitch = cGridSize.x;                                                                                                                   \
  cGridSize.x *= __cIteratorCPU._anProcessMax[2] - __cIteratorCPU._anProcessMin[2];                                                                                                        \
  int nDim3Pitch = cGridSize.y;                                                                                                                   \
  cGridSize.y *= __cIteratorCPU._anProcessMax[3] - __cIteratorCPU._anProcessMin[3];                                                                                                        \
                                                                                                                                                \
  int nSharedMem = 0;                                                                                                                           \
                                                                                                                                                \
  cudaMemcpyToSymbol(__cIterator, &__cIteratorCPU, sizeof(HueFilterHelper_Iterator));  \
  for (int i=0;i<16;i++) __BindTexture(i, __cIteratorCPU, pProcessArea, false); \
  FunctionName##CUDAKernel<<<cGridSize, cBlockSize, nSharedMem>>>(nDim2Pitch, nDim3Pitch);                    \
  for (int i=0;i<16;i++) __BindTexture(i, __cIteratorCPU, pProcessArea, true); \
}                                                                                                                                               \
__global__ void FunctionName##CUDAKernel(const int nDim2Pitch, const int nDim3Pitch)                                                            \
{                                                                                                                                               \
  int __iDim2 = blockIdx.x / nDim2Pitch;                                                                                                          \
  int nBlockIdX = blockIdx.x - (__iDim2 * nDim2Pitch);                                                                                            \
  int __iDim0 = __mul24(nBlockIdX,blockDim.x) + threadIdx.x;                                                                                    \
  int __iDim3 = blockIdx.y / nDim3Pitch;                                                                                                          \
  int nBlockIdY = blockIdx.y - (__iDim3 * nDim3Pitch);                                                                                            \
  int __iDim1 = __mul24(nBlockIdY,blockDim.y) + threadIdx.y;                                                                                    \
                                                                  \
  __iDim0 += __cIterator._anProcessMin[0];    \
  __iDim1 += __cIterator._anProcessMin[1];    \
  __iDim2 += __cIterator._anProcessMin[2];    \
  __iDim3 += __cIterator._anProcessMin[3];    \
                                                          \
  if (__iDim0 < __cIterator._anProcessMax[0] &&                                                                                                       \
      __iDim1 < __cIterator._anProcessMax[1] &&                                                                                                       \
      __iDim2 < __cIterator._anProcessMax[2] &&                                                                                                         \
      __iDim3 < __cIterator._anProcessMax[3])                                                                                                         \
  {                                                                                                                                             \


#define HUEFILTERHELPER_FUNCTION_END }}

#define HF_GLOBALOUTPUTASSERT(__ioutput,__dim0,__dim1,__dim2,__dim3)

#else

#define HUEFILTERHELPER_VARIABLE static
#define HUEFILTERHELPER_FUNCTION static

#define HF_INPUTINDEX(__input,__dim0,__dim1,__dim2,__dim3) \
  (__cIterator._aanInputPitch[__input][0] * (__dim0 - __cIterator._aanInputMin[__input][0]) + \
   __cIterator._aanInputPitch[__input][1] * (__dim1 - __cIterator._aanInputMin[__input][1]) + \
   __cIterator._aanInputPitch[__input][2] * (__dim2 - __cIterator._aanInputMin[__input][2]) + \
   __cIterator._aanInputPitch[__input][3] * (__dim3 - __cIterator._aanInputMin[__input][3]))

#define HF_INPUTBITINDEX(__input,__dim0,__dim1,__dim2,__dim3) \
  (__cIterator._aanInputBitPitch[__input][0] * (__dim0 - __cIterator._aanInputMin[__input][0]) + \
   __cIterator._aanInputBitPitch[__input][1] * (__dim1 - __cIterator._aanInputMin[__input][1]) + \
   __cIterator._aanInputBitPitch[__input][2] * (__dim2 - __cIterator._aanInputMin[__input][2]) + \
   __cIterator._aanInputBitPitch[__input][3] * (__dim3 - __cIterator._aanInputMin[__input][3]))

#define HF_OUTPUTINDEX(__output,__dim0,__dim1,__dim2,__dim3) \
  (__cIterator._aanOutputPitch[__output][0] * (__dim0 - __cIterator._aanOutputMin[__output][0]) + \
   __cIterator._aanOutputPitch[__output][1] * (__dim1 - __cIterator._aanOutputMin[__output][1]) + \
   __cIterator._aanOutputPitch[__output][2] * (__dim2 - __cIterator._aanOutputMin[__output][2]) + \
   __cIterator._aanOutputPitch[__output][3] * (__dim3 - __cIterator._aanOutputMin[__output][3]))

#define HF_READ(__type, __input, __index) \
  (((__type *)__cIterator._apxInput[__input])[__index])

#define HUEFILTERHELPER_CONSTANTARRAY(__type,__name,__size) static __type __name[__size]; \
  extern "C" void SetConstantArray_##__name##CUDA(__type * __ptConstantArray); \
  void SetConstantArray_##__name(__type * __ptConstantArray, HueFilterHelper * pcFilterHelper) \
  {   \
       if (pcFilterHelper->_isCUDA) \
       { \
         SetConstantArray_##__name##CUDA(__ptConstantArray); \
       } \
      memcpy(__name, __ptConstantArray, __size * sizeof(__type)); \
  }

#define HUEFILTERHELPER_FUNCTION_BEGIN(FunctionName) \
  extern "C" void FunctionName##CUDA(HueFilterHelper_ProcessArea * pProcessArea); \
  void FunctionName(HueFilterHelper_ProcessArea * pProcessArea) \
{                                                                                                                                               \
  if (pProcessArea->_pcFilterHelper->_isCUDA)                                                                                                   \
  {                                                                                                                                             \
    FunctionName##CUDA(pProcessArea);                                                                          \
    return;                                                                                                                                     \
  }          \
  HueFilterHelper_Iterator                                                                                                                  \
    __cIterator;                                                                                                                                \
  __cIterator.Clear();                           \
  for (int __i = 0; __i < HUEFILTERHELPER_INPUT_MAX; __i++)  \
  { \
  if (pProcessArea->_eOutput[__i] != HUEFILTERHELPER_NONE) \
    { \
    __cIterator._apxOutput[__i] = HueFilterHelper_GetPointer(pProcessArea->_pcFilterHelper, pProcessArea->_eOutput[__i], pProcessArea->_iOutput[__i]);                                           \
    HueFilterHelper_GetArea(__cIterator._aanOutputMin[__i], __cIterator._aanOutputMax[__i], pProcessArea->_pcFilterHelper, pProcessArea->_eOutput[__i], pProcessArea->_iOutput[__i]);                                       \
    HueFilterHelper_GetPitch(__cIterator._aanOutputPitch[__i], __cIterator._aanOutputBitPitch[__i], pProcessArea->_pcFilterHelper, pProcessArea->_eOutput[__i], pProcessArea->_iOutput[__i]);                                      \
    } \
    if (pProcessArea->_eInput[__i] != HUEFILTERHELPER_NONE) \
    { \
    __cIterator._apxInput[__i] = HueFilterHelper_GetPointer(pProcessArea->_pcFilterHelper, pProcessArea->_eInput[__i], pProcessArea->_iInput[__i]);                                             \
    HueFilterHelper_GetArea(__cIterator._aanInputMin[__i], __cIterator._aanInputMax[__i], pProcessArea->_pcFilterHelper, pProcessArea->_eInput[__i], pProcessArea->_iInput[__i]);                                           \
    HueFilterHelper_GetPitch(__cIterator._aanInputPitch[__i], __cIterator._aanInputBitPitch[__i], pProcessArea->_pcFilterHelper, pProcessArea->_eInput[__i], pProcessArea->_iInput[__i]);                                         \
    } \
  } \
  for (int __i = 0; __i < 6; __i++)                                                                                                             \
  {                                                                                                                                             \
    __cIterator._anProcessMin[__i] = pProcessArea->_anProcessMin[__i];                                               \
    __cIterator._anProcessMax[__i] = pProcessArea->_anProcessMax[__i];                                               \
  }                                                                                                                                             \
                                                                                                                                                \
  for (int __iDim3 = __cIterator._anProcessMin[3]; __iDim3 < __cIterator._anProcessMax[3]; __iDim3++)                                                                            \
  {                                                                                                                                             \
    for (int __iDim2 = __cIterator._anProcessMin[2]; __iDim2 < __cIterator._anProcessMax[2]; __iDim2++)                                                                            \
    {                                                                                                                                             \
      for (int __iDim1 = __cIterator._anProcessMin[1]; __iDim1 < __cIterator._anProcessMax[1]; __iDim1++)                                                                          \
      {                                                                                                                                           \
        for (int __iDim0 = __cIterator._anProcessMin[0]; __iDim0 < __cIterator._anProcessMax[0]; __iDim0++)                                                                        \
        {                                                                                                                                         \
                                                                                                                                                \

#define HUEFILTERHELPER_FUNCTION_END }}}}}

#endif

#define HF_INPUT(__type, __input,__dim0,__dim1,__dim2,__dim3) \
  HF_READ(__type, __input, HF_INPUTINDEX(__input,__dim0 + __iDim0,__dim1 + __iDim1, __dim2 + __iDim2, __dim3 + __iDim3))

#define HF_INPUT1BIT(__input,__dim0,__dim1,__dim2,__dim3) \
  ((HF_READ(uchar, __input, HF_INPUTBITINDEX(__input,__dim0 + __iDim0,__dim1 + __iDim1, __dim2 + __iDim2, __dim3 + __iDim3) / 8) & (1 << (HF_INPUTBITINDEX(__input,__dim0 + __iDim0,__dim1 + __iDim1, __dim2 + __iDim2, __dim3 + __iDim3) % 8))) != 0)

#define HF_GLOBALINPUT(__type,__input,__dim0,__dim1,__dim2,__dim3) \
  HF_READ(__type, __input, HF_INPUTINDEX(__input,__dim0,__dim1,__dim2,__dim3))

#define HF_GLOBALINPUT1BIT(__input,__dim0,__dim1,__dim2,__dim3) \
  ((HF_READ(uchar, __input, HF_INPUTBITINDEX(__input,__dim0,__dim1,__dim2,__dim3) / 8) & (1 << (HF_INPUTBITINDEX(__input,__dim0,__dim1,__dim2,__dim3) % 8))) != 0)

#define HF_OUTPUT(__type, __output) \
  (((__type *)__cIterator._apxOutput[__output])[HF_OUTPUTINDEX(__output, __iDim0, __iDim1, __iDim2, __iDim3)])

#define HF_GLOBALOUTPUT(__type, __output,__dim0,__dim1,__dim2,__dim3) \
  (((__type *)__cIterator._apxOutput[__output])[HF_OUTPUTINDEX(__output,__dim0,__dim1,__dim2,__dim3)])
