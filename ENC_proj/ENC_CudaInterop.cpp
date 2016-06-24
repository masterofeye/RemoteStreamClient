

#include <string>
#include "ENC_CudaInterop.hpp"
#include "ENC_CudaAutoLock.h"
#include "common/inc/nvUtils.h"
#include "common/inc/nvFileIO.h"
#include "common/inc/helper_string.h"
#include "common/inc/dynlink_builtin_types.h"
#include "..\DEC_Intel\DEC_Intel.hpp"

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

using namespace std;

namespace RW{
	namespace ENC{

		ENC_CudaInterop::ENC_CudaInterop(std::shared_ptr<spdlog::logger> Logger) :
			RW::CORE::AbstractModule(Logger) 
            , m_cuContext(nullptr)
            , m_cuModule(nullptr)
            //, m_cuInterleaveUVFunction(nullptr)
            , m_uEncodeBufferCount(0)

		{
                m_pNvHWEncoder = new CNvHWEncoder(m_Logger);

				//memset(&m_stEncoderInput, 0, sizeof(m_stEncoderInput));
				memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));

				memset(&m_stEncodeBuffer, 0, sizeof(m_stEncodeBuffer));
				//memset(m_ChromaDevPtr, 0, sizeof(m_ChromaDevPtr));

				memset(&m_encodeConfig, 0, sizeof(m_encodeConfig));
			}


		ENC_CudaInterop::~ENC_CudaInterop()
		{
			if (m_pNvHWEncoder)
			{
				delete m_pNvHWEncoder;
				m_pNvHWEncoder = nullptr;
			}
		}

		void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
		{
			switch (SubModuleType)
			{
			case CORE::tenSubModule::nenDecoder_INTEL:
			case CORE::tenSubModule::nenDecoder_NVIDIA:
			{
				RW::DEC::tstMyControlStruct *data = static_cast<RW::DEC::tstMyControlStruct*>(*Data);
				data->pstEncodedStream = (this->pstBitStream);
				break;
			}
			default:
				break;
			}
		}

		CORE::tstModuleVersion ENC_CudaInterop::ModulVersion() {
			CORE::tstModuleVersion version = { 0, 1 };
			return version;
		}

		CORE::tenSubModule ENC_CudaInterop::SubModulType()
		{
			return CORE::tenSubModule::nenEncode_NVIDIA;
		}

		NVENCSTATUS ENC_CudaInterop::InitCuda(unsigned int deviceID)
		{
			CUdevice        cuDevice = 0;
			CUcontext       cuContextCurr;
			int  deviceCount = 0;
			int  SMminor = 0, SMmajor = 0;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			typedef HMODULE CUDADRIVER;
#else
			typedef void *CUDADRIVER;
#endif
			CUDADRIVER hHandleDriver = 0;

			// CUDA interfaces
			CUresult err = cuInit(0, __CUDA_API_VERSION, hHandleDriver);
			if (err != CUDA_SUCCESS)
			{
				m_Logger->error("ENC_CudaInterop::InitCuda: cuInit(...)") << " has returned CUDA error " << err; 
				return NV_ENC_ERR_GENERIC;
			}

			err = cuDeviceGetCount(&deviceCount);
			if (err != CUDA_SUCCESS)
			{
				m_Logger->error("ENC_CudaInterop::InitCuda: cuDeviceGetCount(...)") << " has returned CUDA error " << err;
				return NV_ENC_ERR_GENERIC;
			}

			if (deviceCount == 0)
			{
				return NV_ENC_ERR_NO_ENCODE_DEVICE;
			}

			if (deviceID > (unsigned int)deviceCount - 1)
			{
				m_Logger->error("InitCuda: Invalid Device Id = ") << deviceID;
				return NV_ENC_ERR_INVALID_ENCODERDEVICE;
			}

			// Now we get the actual device
			err = cuDeviceGet(&cuDevice, deviceID);
			if (err != CUDA_SUCCESS)
			{
				m_Logger->error("ENC_CudaInterop::InitCuda: cuDeviceGet(...)") << " has returned CUDA error " << err;
				return NV_ENC_ERR_GENERIC;
			}

			//on client PC this function is failing. Has to be done on server PC
			err = cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID);
			if (err != CUDA_SUCCESS)
			{
				m_Logger->error("ENC_CudaInterop::InitCuda: cuDeviceComputeCapability(...)") << " has returned CUDA error " << err;
				return NV_ENC_ERR_GENERIC;
			}
			if (((SMmajor << 4) + SMminor) < 0x30)
			{
				m_Logger->error("InitCuda: Insufficient NVENC capabilities exiting of GPU ") << deviceID;
				return NV_ENC_ERR_NO_ENCODE_DEVICE;
			}

			// Create the CUDA Context and Pop the current one
			err = cuCtxCreate(&m_cuContext, 0, cuDevice);
			if (err != CUDA_SUCCESS)
			{
				m_Logger->error("ENC_CudaInterop::InitCuda: cuCtxCreate(...)") << " has returned CUDA error " << err;
				return NV_ENC_ERR_GENERIC;
			}

			err = cuCtxPopCurrent(&cuContextCurr);
			if (err != CUDA_SUCCESS)
			{
				m_Logger->error("ENC_CudaInterop::InitCuda: cuCtxPopCurrent(...)") << " has returned CUDA error " << err;
				return NV_ENC_ERR_GENERIC;
			}
			return NV_ENC_SUCCESS;
		}

		NVENCSTATUS ENC_CudaInterop::AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight)
		{
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

			m_EncodeBufferQueue.Initialize(m_stEncodeBuffer, m_uEncodeBufferCount);

			ENC_CudaAutoLock cuLock(m_cuContext);

			CUresult err;
			for (uint32_t i = 0; i < m_uEncodeBufferCount; i++)
			{
				err = cuMemAllocPitch(&m_stEncodeBuffer[i].stInputBfr.pNV12devPtr, (size_t *)&m_stEncodeBuffer[i].stInputBfr.uNV12Stride, uInputWidth, uInputHeight * 3 / 2, 8);
				if (err != CUDA_SUCCESS)
				{
					m_Logger->error("ENC_CudaInterop::AllocateIOBuffers: cuMemAllocPitch(...)") << " has returned CUDA error " << err;
					return NV_ENC_ERR_GENERIC;
				}

				nvStatus = m_pNvHWEncoder->NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, (void*)m_stEncodeBuffer[i].stInputBfr.pNV12devPtr,
					uInputWidth, uInputHeight, m_stEncodeBuffer[i].stInputBfr.uNV12Stride, &m_stEncodeBuffer[i].stInputBfr.nvRegisteredResource);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;

				m_stEncodeBuffer[i].stInputBfr.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12_PL;
				m_stEncodeBuffer[i].stInputBfr.dwWidth = uInputWidth;
				m_stEncodeBuffer[i].stInputBfr.dwHeight = uInputHeight;

                nvStatus = m_pNvHWEncoder->NvEncCreateBitstreamBuffer(m_encodeConfig.uBitstreamBufferSize, &m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;
                m_stEncodeBuffer[i].stOutputBfr.dwBitstreamBufferSize = m_encodeConfig.uBitstreamBufferSize;

#if defined(NV_WINDOWS)
				nvStatus = m_pNvHWEncoder->NvEncRegisterAsyncEvent(&m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;
				m_stEncodeBuffer[i].stOutputBfr.bWaitOnEvent = true;
#else
				m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = nullptr;
#endif
			}

			m_stEOSOutputBfr.bEOSFlag = TRUE;
#if defined(NV_WINDOWS)
			nvStatus = m_pNvHWEncoder->NvEncRegisterAsyncEvent(&m_stEOSOutputBfr.hOutputEvent);
			if (nvStatus != NV_ENC_SUCCESS)
				return nvStatus;
#else
			m_stEOSOutputBfr.hOutputEvent = nullptr;
#endif

			return NV_ENC_SUCCESS;
		}

		NVENCSTATUS ENC_CudaInterop::ReleaseIOBuffers()
		{
			ENC_CudaAutoLock cuLock(m_cuContext);
			CUresult err = CUDA_SUCCESS;

			for (uint32_t i = 0; i < m_uEncodeBufferCount; i++)
			{
				err = cuMemFree(m_stEncodeBuffer[i].stInputBfr.pNV12devPtr);
				if (err != CUDA_SUCCESS)
				{
					m_Logger->error("ENC_CudaInterop::AllocateIOBuffers: cuMemAllocPitch(...)") << " has returned CUDA error " << err;
					return NV_ENC_ERR_GENERIC;
				}

				m_pNvHWEncoder->NvEncDestroyBitstreamBuffer(m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
				m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer = nullptr;

#if defined(NV_WINDOWS)
				m_pNvHWEncoder->NvEncUnregisterAsyncEvent(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				nvCloseFile(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = nullptr;
#endif
			}

			if (m_stEOSOutputBfr.hOutputEvent)
			{
#if defined(NV_WINDOWS)
				m_pNvHWEncoder->NvEncUnregisterAsyncEvent(m_stEOSOutputBfr.hOutputEvent);
				nvCloseFile(m_stEOSOutputBfr.hOutputEvent);
				m_stEOSOutputBfr.hOutputEvent = nullptr;
#endif
			}

			return NV_ENC_SUCCESS;
		}

		NVENCSTATUS ENC_CudaInterop::FlushEncoder()
		{
			NVENCSTATUS nvStatus = m_pNvHWEncoder->NvEncFlushEncoderQueue(m_stEOSOutputBfr.hOutputEvent);
			if (nvStatus != NV_ENC_SUCCESS)
			{
                m_Logger->error("FlushEncoder: m_pNvHWEncoder->NvEncFlushEncoderQueue(...) did not succeed!");
                return nvStatus;
			}

			EncodeBuffer *pEncodeBuffer = m_EncodeBufferQueue.GetPending();
			while (pEncodeBuffer)
			{
                pEncodeBuffer = m_EncodeBufferQueue.GetPending();

				// UnMap the input buffer after frame is done
				if (pEncodeBuffer && pEncodeBuffer->stInputBfr.hInputSurface)
				{
					nvStatus = m_pNvHWEncoder->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
					pEncodeBuffer->stInputBfr.hInputSurface = nullptr;
				}
			}
#if defined(NV_WINDOWS)
			if (WaitForSingleObject(m_stEOSOutputBfr.hOutputEvent, 500) != WAIT_OBJECT_0)
			{
                m_Logger->error("FlushEncoder: WaitForSingleObject != WAIT_OBJECT_0");
                nvStatus = NV_ENC_ERR_GENERIC;
			}
#endif
			return nvStatus;
		}

		tenStatus ENC_CudaInterop::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
		{
            m_Logger->debug("Initialise nenEncode_NVIDIA");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

			tenStatus enStatus = tenStatus::nenSuccess;
			stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);

			if (data == nullptr)
			{
				m_Logger->error("Initialise: Data of tstMyInitialiseControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}
            if (data->pstEncodeConfig == nullptr)
            {
                m_Logger->error("Initialise: EncodeConfig of Data of tstMyInitialiseControlStruct is empty!");
                enStatus = tenStatus::nenError;
                return enStatus;
            }

			m_encodeConfig = *data->pstEncodeConfig;

			// initialize Cuda
			nvStatus = InitCuda(m_encodeConfig.deviceID);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("Initialise: InitCuda(...) did not succeed!");
				return enStatus;
			}

            nvStatus = m_pNvHWEncoder->Initialize((void*)m_cuContext, NV_ENC_DEVICE_TYPE_CUDA);
            if (nvStatus != NV_ENC_SUCCESS)
            {
                m_Logger->error("Initialise: m_pNvHWEncoder->Initialize(...) did not succeed!");
            }
            
            if (m_encodeConfig.width == 0 || m_encodeConfig.height == 0)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("Initialise: Invalid frame size parameters!");
			}

			m_encodeConfig.presetGUID = m_pNvHWEncoder->GetPresetGUID(m_encodeConfig.encoderPreset, m_encodeConfig.codec);

			m_Logger->debug(" ENC_CudaInterop::Initialise: >>>ENCODING INFO<<<") \
				<< "\n codec            :" << ((m_encodeConfig.codec == NV_ENC_HEVC) ? "HEVC" : "H264")\
				<< "\n size             :" << m_encodeConfig.width << m_encodeConfig.height \
				<< "\n bitrate (bit/sec):" << m_encodeConfig.bitrate \
				<< "\n vbvMaxBitrate    :" << m_encodeConfig.vbvMaxBitrate \
				<< "\n vbvSize (bits)   :" << m_encodeConfig.vbvSize \
				<< "\n fps (frame/sec)  :" << m_encodeConfig.fps \
				<< "\n rcMode           :" << ((m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_CONSTQP) ? "CONSTQP" :
					(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_VBR) ? "VBR" :
					(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_CBR) ? "CBR" :
					(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_VBR_MINQP) ? "VBR MINQP" :
					(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_QUALITY) ? "TWO_PASS_QUALITY" :
					(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_FRAMESIZE_CAP) ? "TWO_PASS_FRAMESIZE_CAP" :
					(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_VBR) ? "TWO_PASS_VBR" : "UNKNOWN") \
				<< "\n goplength        :" << m_encodeConfig.gopLength \
				<< "\n B frames         :" << m_encodeConfig.numB\
				<< "\n QP               :" << m_encodeConfig.qp\
				<< "\n preset           :" << ((m_encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HQ_GUID) ? "LOW_LATENCY_HQ" :
					(m_encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HP_GUID) ? "LOW_LATENCY_HP" :
					(m_encodeConfig.presetGUID == NV_ENC_PRESET_HQ_GUID) ? "HQ_PRESET" :
					(m_encodeConfig.presetGUID == NV_ENC_PRESET_HP_GUID) ? "HP_PRESET" :
					(m_encodeConfig.presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID) ? "LOSSLESS_HP" :
					(m_encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID) ? "LOW_LATENCY_DEFAULT" : "DEFAULT");

			nvStatus = m_pNvHWEncoder->CreateEncoder(&m_encodeConfig);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("Initialise: m_pNvHWEncoder->CreateEncoder(...) did not succeed!");
				return enStatus;
			}

			m_uEncodeBufferCount = m_encodeConfig.numB + 4;

			nvStatus = AllocateIOBuffers(m_encodeConfig.width, m_encodeConfig.height);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("Initialise: AllocateIOBuffers(...) did not succeed!");
				return enStatus;
			}

			m_u32NumFramesEncoded = 0;
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Initialise time of Module ENC DoRender: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
			return enStatus;
		}

		tenStatus ENC_CudaInterop::DoRender(CORE::tstControlStruct * ControlStruct)
		{
            m_Logger->debug("DoRender nenEncode_NVIDIA");
			tenStatus enStatus = tenStatus::nenSuccess;
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
			stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
			if (data == nullptr)
			{
				m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
				return tenStatus::nenError;
			}
			if (!data->pcuYUVArray)
			{
				m_Logger->error("DoRender: device pointer of stMyControlStruct is empty!");
				return tenStatus::nenError;
			}

			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
			EncodeBuffer *pEncodeBuffer;

			pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
			if (!pEncodeBuffer)
			{
				pEncodeBuffer = m_EncodeBufferQueue.GetPending();

				pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
			}

			pEncodeBuffer->stInputBfr.pNV12devPtr = data->pcuYUVArray;

			nvStatus = m_pNvHWEncoder->NvEncMapInputResource(pEncodeBuffer->stInputBfr.nvRegisteredResource, &pEncodeBuffer->stInputBfr.hInputSurface);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				m_Logger->error("DoRender: m_pNvHWEncoder->NvEncMapInputResource(...) did not succeed!");
				return tenStatus::nenError;
			}

			nvStatus = m_pNvHWEncoder->NvEncEncodeFrame(pEncodeBuffer, nullptr, m_encodeConfig.width, m_encodeConfig.height, data->pPayload);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				m_Logger->error("DoRender: m_pNvHWEncoder->NvEncEncodeFrame(...) did not succeed!");
				return tenStatus::nenError;
			}

            if (pEncodeBuffer != nullptr)
            {
                NV_ENC_LOCK_BITSTREAM stBitStreamData;
                nvStatus = m_pNvHWEncoder->ProcessOutput(pEncodeBuffer, &stBitStreamData);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("DoRender: m_pNvHWEncoder->ProcessOutput(...) did not succeed!");
                    return tenStatus::nenError;
                }
                data->pstBitStream->pBuffer = stBitStreamData.bitstreamBufferPtr;
                data->pstBitStream->u32Size = stBitStreamData.bitstreamSizeInBytes;

                // UnMap the input buffer after frame done
                if (pEncodeBuffer->stInputBfr.hInputSurface)
                {
                    nvStatus = m_pNvHWEncoder->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
                    if (nvStatus != NV_ENC_SUCCESS)
                    {
                        m_Logger->error("DoRender: m_pNvHWEncoder->NvEncUnmapInputResource(...) did not succeed!");
                        return tenStatus::nenError;
                    }

                    pEncodeBuffer->stInputBfr.hInputSurface = nullptr;
                }
            }
			m_u32NumFramesEncoded++;

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Time of Module ENC DoRender: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
            m_Logger->trace() << "Number of encoded files: " << m_u32NumFramesEncoded;
#endif
			return enStatus;
		}

		NVENCSTATUS ENC_CudaInterop::CuDestroy()
		{
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
			CUresult err = cuCtxDestroy(m_cuContext);
			if (err != CUDA_SUCCESS)
			{
				m_Logger->error("ENC_CudaInterop::CuDestroy: cuCtxDestroy(...)") << " has returned CUDA error " << err;
				return NV_ENC_ERR_GENERIC;
			}
			return nvStatus;
		}

		tenStatus ENC_CudaInterop::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
		{
            m_Logger->debug("Deinitialise nenEncode_NVIDIA");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyDeinitialiseControlStruct* data = static_cast<stMyDeinitialiseControlStruct*>(DeinitialiseControlStruct);

			if (data == nullptr)
			{
				m_Logger->error("Deinitialise: Data of stMyDeinitialiseControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			NVENCSTATUS nvStatus = FlushEncoder();
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("DeInitialise: FlushEncoder(...) did not succeed!");
			}

			nvStatus = ReleaseIOBuffers();
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("DeInitialise: ReleaseIOBuffers(...) did not succeed!");
			}

			nvStatus = m_pNvHWEncoder->NvEncDestroyEncoder();
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("DeInitialise: m_pNvHWEncoder->NvEncDestroyEncoder(...) did not succeed!");
			}

			nvStatus = CuDestroy();
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("DeInitialise: CuDestroy(...) did not succeed!");
			}

#ifdef TRACE_PERFORMANCE
			RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
			m_Logger->trace() << "Execution Time of Module ENC: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
			return enStatus;
		}


	}
}