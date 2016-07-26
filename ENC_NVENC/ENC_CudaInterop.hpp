#pragma once

#include "NvHWEncoder.h"
#include "ENC_Queue.h"
#include "ENC_CudaAutoLock.h"
#include "AbstractModule.hpp"

namespace RW{
    namespace ENC{

#define MAX_ENCODE_QUEUE 32

		typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
		{
			EncodeConfig *pstEncodeConfig;
		}tstMyInitialiseControlStruct;

        typedef struct stMyControlStruct : public CORE::tstControlStruct
		{
            CUdeviceptr pcuYUVArray;
            tstBitStream *pstBitStream;
            tstBitStream *pPayload;
            REMOTE_API void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
		}tstMyControlStruct;

		typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
		{
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
			CNvHWEncoder                                        *m_pNvHWEncoder;
			uint32_t                                            m_uEncodeBufferCount;
			CUcontext                                           m_cuContext;
			CUmodule                                            m_cuModule;
    CUfunction                                           m_cuInterleaveUVFunction;
    CUdeviceptr                                          m_ChromaDevPtr[2];
    EncodeConfig                                         m_stEncoderInput;
			EncodeBuffer                                        m_stEncodeBuffer[MAX_ENCODE_QUEUE];
			ENC_Queue<EncodeBuffer>                             m_EncodeBufferQueue;
			EncodeOutputBuffer                                  m_stEOSOutputBfr;
			EncodeConfig										m_encodeConfig;
			uint32_t											m_u32NumFramesEncoded;

			NVENCSTATUS                                         CuDestroy();
			NVENCSTATUS                                         InitCuda(uint32_t deviceID);
			NVENCSTATUS                                         AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight);
			NVENCSTATUS                                         ReleaseIOBuffers();
			NVENCSTATUS                                         FlushEncoder();
            NVENCSTATUS                                          ConvertYUVToNV12(EncodeBuffer *pEncodeBuffer, CUdeviceptr cuDevPtr, int width, int height);

		};
	}
}
typedef NVENCSTATUS(NVENCAPI *MYPROC)(NV_ENCODE_API_FUNCTION_LIST*);
