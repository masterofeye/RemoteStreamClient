#pragma once

#include "common/inc/NvHWEncoder.h"
#include "ENC_Queue.h"
#include "ENC_CudaAutoLock.h"
#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

#define MAX_ENCODE_QUEUE 32
#define NV_ENC_CUDA 2

namespace RW{
	namespace ENC{

		typedef struct _EncodeFrameConfig
		{
			CUarray  cuYUVArray;
			uint32_t stride[3];
			uint32_t width;
			uint32_t height;
		}EncodeFrameConfig;

		typedef struct _EncodedBitStream
		{
			void *pBitStreamBuffer;
			uint32_t u32BitStreamSizeInBytes;
		}EncodedBitStream;

		typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
		{
			EncodeConfig encodeConfig;
		}tstMyInitialiseControlStruct;

		typedef struct stMyControlStruct : public CORE::tstControlStruct
		{
			CUarray cuYUVArray;
			EncodedBitStream stBitStream;			
		}tstMyControlStruct;

		typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
		{
			EncodedBitStream stBitStream;
		}tstMyDeinitialiseControlStruct;

		class ENC_CudaInterop : public RW::CORE::AbstractModule
		{
			Q_OBJECT
			
		public:

			explicit ENC_CudaInterop(std::shared_ptr<spdlog::logger> Logger);
			~ENC_CudaInterop();
			virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
			virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
			virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;


		private:
			//HANDLE											m_hInput;
			CNvHWEncoder                                        *m_pNvHWEncoder;
			uint32_t                                            m_uEncodeBufferCount;
			CUcontext                                           m_cuContext;
			CUmodule                                            m_cuModule;
			CUfunction                                          m_cuInterleaveUVFunction;
			CUdeviceptr                                         m_ChromaDevPtr[2];
			//EncodeConfig                                      m_stEncoderInput;
			EncodeBuffer                                        m_stEncodeBuffer[MAX_ENCODE_QUEUE];
			ENC_Queue<EncodeBuffer>                             m_EncodeBufferQueue;
			EncodeOutputBuffer                                  m_stEOSOutputBfr;
			CUarray                                             m_cuYUVArray;
			EncodeConfig										m_encodeConfig;
			uint32_t											m_u32NumFramesEncoded;
#ifdef TRACE_PERFORMANCE
			RW::CORE::HighResClock::time_point					m_tStart;
#endif

			NVENCSTATUS                                         CuDestroy();
			NVENCSTATUS                                         InitCuda(uint32_t deviceID);
			NVENCSTATUS                                         AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight);
			NVENCSTATUS                                         ReleaseIOBuffers();
			NVENCSTATUS                                         FlushEncoder(NV_ENC_LOCK_BITSTREAM *pstBitStreamData);
			NVENCSTATUS                                         ConvertYUVToNV12(EncodeBuffer *pEncodeBuffer, CUarray cuArray, int width, int height);
			NVENCSTATUS											PreparePreProcCuda();

		};
	}
}
typedef NVENCSTATUS(NVENCAPI *MYPROC)(NV_ENCODE_API_FUNCTION_LIST*);
