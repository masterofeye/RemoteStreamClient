

#include <string>
#include "ENC_CudaInterop.hpp"
#include "ENC_CudaAutoLock.h"
#include "common/inc/nvUtils.h"
#include "common/inc/nvFileIO.h"
#include "common/inc/helper_string.h"
#include "common/inc/dynlink_builtin_types.h"

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif


#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64) || defined(__aarch64__)
#define PTX_FILE "preproc64_cuda.ptx"
#else
#define PTX_FILE "preproc32_cuda.ptx"
#endif

using namespace std;

#define BITSTREAM_BUFFER_SIZE 2*1024*1024

namespace RW{
	namespace ENC{

		ENC_CudaInterop::ENC_CudaInterop(std::shared_ptr<spdlog::logger> Logger) :
			RW::CORE::AbstractModule(Logger) 
            , m_cuContext(nullptr)
            , m_cuModule(nullptr)
            , m_cuInterleaveUVFunction(nullptr)
            , m_cuYUVArray(nullptr)
            , m_uEncodeBufferCount(0)

		{
                m_pNvHWEncoder = new CNvHWEncoder(m_Logger);

				//memset(&m_stEncoderInput, 0, sizeof(m_stEncoderInput));
				memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));

				memset(&m_stEncodeBuffer, 0, sizeof(m_stEncodeBuffer));
				memset(m_ChromaDevPtr, 0, sizeof(m_ChromaDevPtr));
				//memset(m_cuYUVArray, 0, sizeof(m_cuYUVArray));

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
			__cu(cuInit(0, __CUDA_API_VERSION, hHandleDriver));

			__cu(cuDeviceGetCount(&deviceCount));
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
			__cu(cuDeviceGet(&cuDevice, deviceID));

			//on client PC this function is failing. Has to be done on server PC
			__cu(cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID));
			if (((SMmajor << 4) + SMminor) < 0x30)
			{
				m_Logger->error("InitCuda: Insufficient NVENC capabilities exiting of GPU ") << deviceID;
				return NV_ENC_ERR_NO_ENCODE_DEVICE;
			}

			// Create the CUDA Context and Pop the current one
			__cu(cuCtxCreate(&m_cuContext, 0, cuDevice));

			PreparePreProcCuda();

			__cu(cuModuleGetFunction(&m_cuInterleaveUVFunction, m_cuModule, "InterleaveUV"));

			__cu(cuCtxPopCurrent(&cuContextCurr));
			return NV_ENC_SUCCESS;
		}

		NVENCSTATUS ENC_CudaInterop::PreparePreProcCuda()
		{

			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
			CUresult        cuResult = CUDA_SUCCESS;

			// in this branch we use compilation with parameters
			const unsigned int jitNumOptions = 3;
			CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
			void **jitOptVals = new void *[jitNumOptions];

			// set up size of compilation log buffer
			jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
			int jitLogBufferSize = 1024;
			jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

			// set up pointer to the compilation log buffer
			jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
			char *jitLogBuffer = new char[jitLogBufferSize];
			jitOptVals[1] = jitLogBuffer;

			// set up pointer to set the Maximum # of registers for a particular kernel
			jitOptions[2] = CU_JIT_MAX_REGISTERS;
			int jitRegCount = 32;
			jitOptVals[2] = (void *)(size_t)jitRegCount;

			//Has to be fixed. For some reason i could not bring it to load from relative path.
			char * ptx_path = "C:/Projekte/RemoteStreamClient/build/x64/Debug/preproc64_cuda.ptx"; //(char*)PTX_FILE;
			FILE *fp = fopen(ptx_path, "rb");
			if (!fp)
			{
				m_Logger->error("PreparePreProcCuda: Unable to read ptx file ") << ptx_path;
				return NV_ENC_ERR_INVALID_PARAM;
			}
			fseek(fp, 0, SEEK_END);
			int file_size = ftell(fp);
			char *buf = new char[file_size + 1];
			fseek(fp, 0, SEEK_SET);
			fread(buf, sizeof(char), file_size, fp);
			fclose(fp);
			buf[file_size] = '\0';
			string ptx_source = buf;
			delete[] buf;

			cuResult = cuModuleLoadDataEx(&m_cuModule, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);
			if (cuResult != CUDA_SUCCESS)
			{
				return NV_ENC_ERR_OUT_OF_MEMORY;
			}

			delete[] jitOptions;
			delete[] jitOptVals;
			delete[] jitLogBuffer;

			return nvStatus;
		}

		NVENCSTATUS ENC_CudaInterop::AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight)
		{
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

			m_EncodeBufferQueue.Initialize(m_stEncodeBuffer, m_uEncodeBufferCount);

			ENC_CudaAutoLock cuLock(m_cuContext);

			__cu(cuMemAlloc(&m_ChromaDevPtr[0], uInputWidth*uInputHeight / 4));
			__cu(cuMemAlloc(&m_ChromaDevPtr[1], uInputWidth*uInputHeight / 4));

			// should not be necessary if data is already on gpu
			//__cu(cuMemAllocHost((void **)&m_yuv[0], uInputWidth*uInputHeight));
			//__cu(cuMemAllocHost((void **)&m_yuv[1], uInputWidth*uInputHeight / 4));
			//__cu(cuMemAllocHost((void **)&m_yuv[2], uInputWidth*uInputHeight / 4));

			for (uint32_t i = 0; i < m_uEncodeBufferCount; i++)
			{
				__cu(cuMemAllocPitch(&m_stEncodeBuffer[i].stInputBfr.pNV12devPtr, (size_t *)&m_stEncodeBuffer[i].stInputBfr.uNV12Stride, uInputWidth, uInputHeight * 3 / 2, 8));

				nvStatus = m_pNvHWEncoder->NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, (void*)m_stEncodeBuffer[i].stInputBfr.pNV12devPtr,
					uInputWidth, uInputHeight, m_stEncodeBuffer[i].stInputBfr.uNV12Stride, &m_stEncodeBuffer[i].stInputBfr.nvRegisteredResource);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;

				m_stEncodeBuffer[i].stInputBfr.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12_PL;
				m_stEncodeBuffer[i].stInputBfr.dwWidth = uInputWidth;
				m_stEncodeBuffer[i].stInputBfr.dwHeight = uInputHeight;

				nvStatus = m_pNvHWEncoder->NvEncCreateBitstreamBuffer(BITSTREAM_BUFFER_SIZE, &m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;
				m_stEncodeBuffer[i].stOutputBfr.dwBitstreamBufferSize = BITSTREAM_BUFFER_SIZE;

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

			for (uint32_t i = 0; i < m_uEncodeBufferCount; i++)
			{
				__cu(cuMemFree(m_stEncodeBuffer[i].stInputBfr.pNV12devPtr));

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
				//nvStatus = m_pNvHWEncoder->ProcessOutput(pEncodeBuffer, pstBitStreamData);
				//if (nvStatus != NV_ENC_SUCCESS)
				//{
				//	m_Logger->error("FlushEncoder: m_pNvHWEncoder->ProcessOutput(...) did not succeed!");
				//}

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

		NVENCSTATUS ENC_CudaInterop::ConvertYUVToNV12(EncodeBuffer *pEncodeBuffer, CUarray cuArray, int width, int height)
		{
			ENC_CudaAutoLock cuLock(m_cuContext);

			// splitting up channels
			// NEED TO TEST IF OFFSETS ARE CORRECT! WITH A TEST FRAME
			CUarray arr[3];
			__cu(cuMemcpyAtoA(arr[0], 0, cuArray, 0, width * height));
			__cu(cuMemcpyAtoA(arr[1], 0, cuArray, sizeof(arr[0]), width * height / 4));
			__cu(cuMemcpyAtoA(arr[2], 0, cuArray, sizeof(arr[0]) + sizeof(arr[1]), width * height / 4));

			// copy luma
			CUDA_MEMCPY2D copyParam;
			memset(&copyParam, 0, sizeof(copyParam));
            copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
			copyParam.dstDevice = pEncodeBuffer->stInputBfr.pNV12devPtr;
			copyParam.dstPitch = pEncodeBuffer->stInputBfr.uNV12Stride;
			copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
			copyParam.srcArray = arr[0];
			copyParam.srcPitch = width;
			copyParam.WidthInBytes = width;
			copyParam.Height = height;
			__cu(cuMemcpy2D(&copyParam));

			// copy chroma
			__cu(cuMemcpyAtoD(m_ChromaDevPtr[0], arr[1], 0, width * height / 4));
			__cu(cuMemcpyAtoD(m_ChromaDevPtr[1], arr[2], 0, width * height / 4));

#define BLOCK_X 32
#define BLOCK_Y 16
			int chromaHeight = height / 2;
			int chromaWidth = width / 2;
			dim3 block(BLOCK_X, BLOCK_Y, 1);
			dim3 grid((chromaWidth + BLOCK_X - 1) / BLOCK_X, (chromaHeight + BLOCK_Y - 1) / BLOCK_Y, 1);
#undef BLOCK_Y
#undef BLOCK_X

			CUdeviceptr dNV12Chroma = (CUdeviceptr)((unsigned char*)pEncodeBuffer->stInputBfr.pNV12devPtr + pEncodeBuffer->stInputBfr.uNV12Stride*height);
			void *args[8] = { &m_ChromaDevPtr[0], &m_ChromaDevPtr[1], &dNV12Chroma, &chromaWidth, &chromaHeight, &chromaWidth, &chromaWidth, &pEncodeBuffer->stInputBfr.uNV12Stride };

			__cu(cuLaunchKernel(m_cuInterleaveUVFunction, grid.x, grid.y, grid.z,
				block.x, block.y, block.z,
				0,
				nullptr, args, nullptr));
			CUresult cuResult = cuStreamQuery(nullptr);
			if (!((cuResult == CUDA_SUCCESS) || (cuResult == CUDA_ERROR_NOT_READY)))
			{
				return NV_ENC_ERR_GENERIC;
			}
			return NV_ENC_SUCCESS;
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

			m_Logger->debug(" codec            :") << ((m_encodeConfig.codec == NV_ENC_HEVC) ? "HEVC" : "H264");
            m_Logger->debug(" size             :") << m_encodeConfig.width << m_encodeConfig.height;
            m_Logger->debug(" bitrate (bit/sec):") << m_encodeConfig.bitrate;
            m_Logger->debug(" vbvMaxBitrate    :") << m_encodeConfig.vbvMaxBitrate;
            m_Logger->debug(" vbvSize (bits)   :") << m_encodeConfig.vbvSize;
            m_Logger->debug(" fps (frame/sec)  :") << m_encodeConfig.fps;
            m_Logger->debug(" rcMode           :") << ((m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_CONSTQP) ? "CONSTQP" :
				(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_VBR) ? "VBR" :
				(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_CBR) ? "CBR" :
				(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_VBR_MINQP) ? "VBR MINQP" :
				(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_QUALITY) ? "TWO_PASS_QUALITY" :
				(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_FRAMESIZE_CAP) ? "TWO_PASS_FRAMESIZE_CAP" :
				(m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_VBR) ? "TWO_PASS_VBR" : "UNKNOWN");
			if (m_encodeConfig.gopLength == NVENC_INFINITE_GOPLENGTH)
                m_Logger->debug(" goplength        : INFINITE GOP \n");
			else
                m_Logger->debug(" goplength        :") << m_encodeConfig.gopLength;
            m_Logger->debug(" B frames         :") << m_encodeConfig.numB;
            m_Logger->debug(" QP               :") << m_encodeConfig.qp;
            m_Logger->debug(" preset           :") << ((m_encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HQ_GUID) ? "LOW_LATENCY_HQ" :
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
				enStatus = tenStatus::nenError;
				return enStatus;
			}
			if (data->pcuYUVArray != nullptr)
			{
				m_cuYUVArray = (CUarray)data->pcuYUVArray;
			}
			else
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("DoRender: Input Array is empty!");
				return enStatus;
			}

			//uint32_t numBytesRead = 0;
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
			//bool bError = false;
			EncodeBuffer *pEncodeBuffer;

			pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
			if (!pEncodeBuffer)
			{
				pEncodeBuffer = m_EncodeBufferQueue.GetPending();

				if (pEncodeBuffer != nullptr)
				{
					NV_ENC_LOCK_BITSTREAM stBitStreamData;
					nvStatus = m_pNvHWEncoder->ProcessOutput(pEncodeBuffer, &stBitStreamData);
					if (nvStatus != NV_ENC_SUCCESS)
					{
						enStatus = tenStatus::nenError;
						m_Logger->error("DoRender: m_pNvHWEncoder->ProcessOutput(...) did not succeed!");
						return enStatus;
					}
					data->pstBitStream->pBitStreamBuffer = stBitStreamData.bitstreamBufferPtr;
					data->pstBitStream->u32BitStreamSizeInBytes = stBitStreamData.bitstreamSizeInBytes;

					// UnMap the input buffer after frame done
					if (pEncodeBuffer->stInputBfr.hInputSurface)
					{
						nvStatus = m_pNvHWEncoder->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
						if (nvStatus != NV_ENC_SUCCESS)
						{
							enStatus = tenStatus::nenError;
							m_Logger->error("DoRender: m_pNvHWEncoder->NvEncUnmapInputResource(...) did not succeed!");
							return enStatus;
						}

						pEncodeBuffer->stInputBfr.hInputSurface = nullptr;
					}
				}
				pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
			}

			nvStatus = ConvertYUVToNV12(pEncodeBuffer, m_cuYUVArray, m_encodeConfig.width, m_encodeConfig.height);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("DoRender: ConvertYUVToNV12(...) did not succeed!");
				return enStatus;
			}

			nvStatus = m_pNvHWEncoder->NvEncMapInputResource(pEncodeBuffer->stInputBfr.nvRegisteredResource, &pEncodeBuffer->stInputBfr.hInputSurface);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("DoRender: m_pNvHWEncoder->NvEncMapInputResource(...) did not succeed!");
				return enStatus;
			}

			nvStatus = m_pNvHWEncoder->NvEncEncodeFrame(pEncodeBuffer, nullptr, m_encodeConfig.width, m_encodeConfig.height, data->pPayload);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("DoRender: m_pNvHWEncoder->NvEncEncodeFrame(...) did not succeed!");
				return enStatus;
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
			__cu(cuCtxDestroy(m_cuContext));
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