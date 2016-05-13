

#include <string>
#include "ENC_CudaInterop.hpp"
#include "ENC_CudaAutoLock.hpp"
#include "common/inc/nvUtils.h"
#include "common/inc/nvFileIO.h"
#include "common/inc/helper_string.h"
#include "common/inc/dynlink_builtin_types.h"

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64) || defined(__aarch64__)
#define PTX_FILE "preproc64_cuda.ptx"
#else
#define PTX_FILE "preproc32_cuda.ptx"
#endif

using namespace std;

#define __cu(a) do { CUresult  ret; if ((ret = (a)) != CUDA_SUCCESS) { fprintf(stderr, "%s has returned CUDA error %d\n", #a, ret); return NV_ENC_ERR_GENERIC;}} while(0)

#define BITSTREAM_BUFFER_SIZE 2*1024*1024

namespace RW{
	namespace ENC{

		ENC_CudaInterop::ENC_CudaInterop(std::shared_ptr<spdlog::logger> Logger) :
			RW::CORE::AbstractModule(Logger)
		{
			m_pNvHWEncoder = new CNvHWEncoder;

			m_cuContext = NULL;
			m_cuModule = NULL;
			m_cuInterleaveUVFunction = NULL;

			m_uEncodeBufferCount = 0;
			//memset(&m_stEncoderInput, 0, sizeof(m_stEncoderInput));
			memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));

			memset(&m_stEncodeBuffer, 0, sizeof(m_stEncodeBuffer));
			memset(m_ChromaDevPtr, 0, sizeof(m_ChromaDevPtr));
			memset(m_cuYUVArray, 0, sizeof(m_cuYUVArray));

			memset(&m_encodeConfig, 0, sizeof(m_encodeConfig));

			m_encodeConfig.endFrameIdx = INT_MAX;
			m_encodeConfig.bitrate = 5000000;						//make editable
			m_encodeConfig.rcMode = NV_ENC_PARAMS_RC_CONSTQP;
			m_encodeConfig.gopLength = NVENC_INFINITE_GOPLENGTH;
			m_encodeConfig.deviceType = NV_ENC_CUDA;
			m_encodeConfig.codec = NV_ENC_H264;
			m_encodeConfig.fps = 30;								//make editable
			m_encodeConfig.qp = 28;
			m_encodeConfig.i_quant_factor = DEFAULT_I_QFACTOR;
			m_encodeConfig.b_quant_factor = DEFAULT_B_QFACTOR;
			m_encodeConfig.i_quant_offset = DEFAULT_I_QOFFSET;
			m_encodeConfig.b_quant_offset = DEFAULT_B_QOFFSET;
			m_encodeConfig.presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
			m_encodeConfig.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
		}


		ENC_CudaInterop::~ENC_CudaInterop()
		{
			if (m_pNvHWEncoder)
			{
				delete m_pNvHWEncoder;
				m_pNvHWEncoder = NULL;
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
				PRINTERR("Invalid Device Id = %d\n", deviceID);
				return NV_ENC_ERR_INVALID_ENCODERDEVICE;
			}

			// Now we get the actual device
			__cu(cuDeviceGet(&cuDevice, deviceID));

			__cu(cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID));
			if (((SMmajor << 4) + SMminor) < 0x30)
			{
				PRINTERR("GPU %d does not have NVENC capabilities exiting\n", deviceID);
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

			const char* exec_path = ".//data//";
			char *ptx_path = sdkFindFilePath(PTX_FILE, exec_path);
			if (ptx_path == NULL) {
				PRINTERR("Unable to find ptx file path %s\n", PTX_FILE);
				return NV_ENC_ERR_INVALID_PARAM;
			}

			FILE *fp = fopen(ptx_path, "rb");
			if (!fp)
			{
				PRINTERR("Unable to read ptx file %s\n", PTX_FILE);
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
				m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = NULL;
#endif
			}

			m_stEOSOutputBfr.bEOSFlag = TRUE;
#if defined(NV_WINDOWS)
			nvStatus = m_pNvHWEncoder->NvEncRegisterAsyncEvent(&m_stEOSOutputBfr.hOutputEvent);
			if (nvStatus != NV_ENC_SUCCESS)
				return nvStatus;
#else
			m_stEOSOutputBfr.hOutputEvent = NULL;
#endif

			return NV_ENC_SUCCESS;
		}

		NVENCSTATUS ENC_CudaInterop::ReleaseIOBuffers()
		{
			ENC_CudaAutoLock cuLock(m_cuContext);

			for (uint32_t i = 0; i < m_uEncodeBufferCount; i++)
			{
				cuMemFree(m_stEncodeBuffer[i].stInputBfr.pNV12devPtr);

				m_pNvHWEncoder->NvEncDestroyBitstreamBuffer(m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
				m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer = NULL;

#if defined(NV_WINDOWS)
				m_pNvHWEncoder->NvEncUnregisterAsyncEvent(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				nvCloseFile(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = NULL;
#endif
			}

			if (m_stEOSOutputBfr.hOutputEvent)
			{
#if defined(NV_WINDOWS)
				m_pNvHWEncoder->NvEncUnregisterAsyncEvent(m_stEOSOutputBfr.hOutputEvent);
				nvCloseFile(m_stEOSOutputBfr.hOutputEvent);
				m_stEOSOutputBfr.hOutputEvent = NULL;
#endif
			}

			return NV_ENC_SUCCESS;
		}

		NVENCSTATUS ENC_CudaInterop::FlushEncoder()
		{
			NVENCSTATUS nvStatus = m_pNvHWEncoder->NvEncFlushEncoderQueue(m_stEOSOutputBfr.hOutputEvent);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				assert(0);
				return nvStatus;
			}

			EncodeBuffer *pEncodeBuffer = m_EncodeBufferQueue.GetPending();
			while (pEncodeBuffer)
			{
				m_pNvHWEncoder->ProcessOutput(pEncodeBuffer);
				pEncodeBuffer = m_EncodeBufferQueue.GetPending();
				// UnMap the input buffer after frame is done
				if (pEncodeBuffer && pEncodeBuffer->stInputBfr.hInputSurface)
				{
					nvStatus = m_pNvHWEncoder->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
					pEncodeBuffer->stInputBfr.hInputSurface = NULL;
				}
			}
#if defined(NV_WINDOWS)
			if (WaitForSingleObject(m_stEOSOutputBfr.hOutputEvent, 500) != WAIT_OBJECT_0)
			{
				assert(0);
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
			copyParam.dstMemoryType = CU_MEMORYTYPE_DEVICE;
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
				NULL, args, NULL));
			CUresult cuResult = cuStreamQuery(NULL);
			if (!((cuResult == CUDA_SUCCESS) || (cuResult == CUDA_ERROR_NOT_READY)))
			{
				return NV_ENC_ERR_GENERIC;
			}
			return NV_ENC_SUCCESS;
		}

		tenStatus ENC_CudaInterop::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
		{
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

			tenStatus enStatus = tenStatus::nenSuccess;
			stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);

			m_encodeConfig = data->encodeConfig;

			if (data->cuYUVArray != NULL)
			{
				m_cuYUVArray = data->cuYUVArray;
			}
			else
			{
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			if (/*!m_encodeConfig.inputFileName || !m_encodeConfig.outputFileName ||*/ m_encodeConfig.width == 0 || m_encodeConfig.height == 0)
			{
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			// initialize Cuda
			nvStatus = InitCuda(m_encodeConfig.deviceID);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			nvStatus = m_pNvHWEncoder->Initialize((void*)m_cuContext, NV_ENC_DEVICE_TYPE_CUDA);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			m_encodeConfig.presetGUID = m_pNvHWEncoder->GetPresetGUID(m_encodeConfig.encoderPreset, m_encodeConfig.codec);

#if defined(PRINTRESULTS)
			printf("         codec           : \"%s\"\n", m_encodeConfig.codec == NV_ENC_HEVC ? "HEVC" : "H264");
			printf("         size            : %dx%d\n", m_encodeConfig.width, m_encodeConfig.height);
			printf("         bitrate         : %d bits/sec\n", m_encodeConfig.bitrate);
			printf("         vbvMaxBitrate   : %d bits/sec\n", m_encodeConfig.vbvMaxBitrate);
			printf("         vbvSize         : %d bits\n", m_encodeConfig.vbvSize);
			printf("         fps             : %d frames/sec\n", m_encodeConfig.fps);
			printf("         rcMode          : %s\n", m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_CONSTQP ? "CONSTQP" :
				m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_VBR ? "VBR" :
				m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_CBR ? "CBR" :
				m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_VBR_MINQP ? "VBR MINQP" :
				m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_QUALITY ? "TWO_PASS_QUALITY" :
				m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_FRAMESIZE_CAP ? "TWO_PASS_FRAMESIZE_CAP" :
				m_encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_VBR ? "TWO_PASS_VBR" : "UNKNOWN");
			if (m_encodeConfig.gopLength == NVENC_INFINITE_GOPLENGTH)
				printf("         goplength       : INFINITE GOP \n");
			else
				printf("         goplength       : %d \n", m_encodeConfig.gopLength);
			printf("         B frames        : %d \n", m_encodeConfig.numB);
			printf("         QP              : %d \n", m_encodeConfig.qp);
			printf("         preset          : %s\n", (m_encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HQ_GUID) ? "LOW_LATENCY_HQ" :
				(m_encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HP_GUID) ? "LOW_LATENCY_HP" :
				(m_encodeConfig.presetGUID == NV_ENC_PRESET_HQ_GUID) ? "HQ_PRESET" :
				(m_encodeConfig.presetGUID == NV_ENC_PRESET_HP_GUID) ? "HP_PRESET" :
				(m_encodeConfig.presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID) ? "LOSSLESS_HP" :
				(m_encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID) ? "LOW_LATENCY_DEFAULT" : "DEFAULT");
			printf("\n");
#endif

			nvStatus = m_pNvHWEncoder->CreateEncoder(&m_encodeConfig);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			m_uEncodeBufferCount = m_encodeConfig.numB + 4;

			nvStatus = AllocateIOBuffers(m_encodeConfig.width, m_encodeConfig.height);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			m_Logger->debug("Initialise");
			return enStatus;
		}

		tenStatus ENC_CudaInterop::DoRender(CORE::tstControlStruct * ControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

			uint32_t numBytesRead = 0;
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
			int numFramesEncoded = 0;
			bool bError = false;
			EncodeBuffer *pEncodeBuffer;

#if defined(PRINTRESULTS)
			unsigned long long lStart, lEnd, lFreq;
			NvQueryPerformanceCounter(&lStart);
#endif

			for (int frm = m_encodeConfig.startFrameIdx; frm <= m_encodeConfig.endFrameIdx; frm++)
			{
				pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
				if (!pEncodeBuffer)
				{
					pEncodeBuffer = m_EncodeBufferQueue.GetPending();
					m_pNvHWEncoder->ProcessOutput(pEncodeBuffer);

					// UnMap the input buffer after frame done
					if (pEncodeBuffer->stInputBfr.hInputSurface)
					{
						nvStatus = m_pNvHWEncoder->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
						pEncodeBuffer->stInputBfr.hInputSurface = NULL;
					}
					pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
				}

				nvStatus = ConvertYUVToNV12(pEncodeBuffer, m_cuYUVArray, m_encodeConfig.width, m_encodeConfig.height);

				nvStatus = m_pNvHWEncoder->NvEncMapInputResource(pEncodeBuffer->stInputBfr.nvRegisteredResource, &pEncodeBuffer->stInputBfr.hInputSurface);
				if (nvStatus != NV_ENC_SUCCESS)
				{
					enStatus = tenStatus::nenError;
					return enStatus;
				}

				nvStatus = m_pNvHWEncoder->NvEncEncodeFrame(pEncodeBuffer, NULL, m_encodeConfig.width, m_encodeConfig.height);
				numFramesEncoded++;
			}

#if defined(PRINTRESULTS)
			if (numFramesEncoded > 0)
			{
				NvQueryPerformanceCounter(&lEnd);
				NvQueryPerformanceFrequency(&lFreq);
				double elapsedTime = (double)(lEnd - lStart);
				printf("Encoded %d frames in %6.2fms\n", numFramesEncoded, (elapsedTime*1000.0) / lFreq);
				printf("Avergage Encode Time : %6.2fms\n", ((elapsedTime*1000.0) / numFramesEncoded) / lFreq);
			}
#endif

			m_Logger->debug("DoRender");
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
			tenStatus enStatus = tenStatus::nenSuccess;
			FlushEncoder();

			if (m_encodeConfig.fOutput)
			{
				fclose(m_encodeConfig.fOutput);
			}

			ReleaseIOBuffers();

			NVENCSTATUS nvStatus = m_pNvHWEncoder->NvEncDestroyEncoder();
			if (nvStatus != NV_ENC_SUCCESS)
			{
				enStatus = tenStatus::nenError;
				assert(0);
			}

			nvStatus = CuDestroy();

			if (nvStatus != NV_ENC_SUCCESS)
				enStatus = tenStatus::nenError;

			m_Logger->debug("Deinitialise");
			return enStatus;
		}


	}
}