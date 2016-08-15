#include "DEC_NvDecodeD3D9.hpp"
#include "..\VPL_QT\VPL_FrameProcessor.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#endif
#include <d3dx9.h>

// CUDA Header includes
#include "dynlink_nvcuvid.h"  // <nvcuvid.h>
#include "dynlink_cuda.h"     // <cuda.h>
#include "dynlink_cudaD3D9.h" // <cudaD3D9.h>
#include "dynlink_builtin_types.h"	  // <builtin_types.h>

// CUDA utilities and system includes
#include "helper_functions.h"
#include "helper_cuda_drvapi.h"

// cudaDecodeD3D9 related helper functions
#include "FrameQueue.h"
#include "VideoParser.h"
#include "VideoDecoder.h"
//#include "ImageDX.h"

#include "cudaProcessFrame.h"
#include "cudaModuleMgr.h"



namespace RW
{
	namespace DEC
	{
        CNvDecodeD3D9::CNvDecodeD3D9(std::shared_ptr<spdlog::logger> Logger) :
            RW::CORE::AbstractModule(Logger)
		{
			g_bFirstFrame = true;
			g_bUpdateCSC = true;
			g_bUpdateAll = false;
			g_bUseDisplay = true; 
			g_bIsProgressive = true;
			g_bException = false;
			g_bWaived = false;

			g_iRepeatFactor = 1; 

			g_CtxLock = NULL;

			cuModNV12toARGB = 0;
			g_kernelNV12toARGB = 0;
			g_kernelPassThru = 0;
			g_oContext = 0;
			g_oDevice = 0;
			g_ReadbackSID = 0, g_KernelSID = 0;
			g_eColorSpace = ITU601;
			g_nHue = 0.0f;

			g_pFrameYUV = { 0 };
			//g_pFrameYUV[1] = { 0 };
			//g_pFrameYUV[2] = { 0 };
			//g_pFrameYUV[3] = { 0 };

			g_pFrameQueue = 0;
			g_pVideoParser = 0;
			g_pVideoDecoder = 0;

			//g_pInteropFrame[0] = { 0 };
			//g_pInteropFrame[1] = { 1 };

			g_nVideoWidth = 0;
			g_nVideoHeight = 0;

			g_DecodeFrameCount = 0;

		}

		CNvDecodeD3D9::~CNvDecodeD3D9()
		{
		}

        void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
        {
            switch (SubModuleType)
            {
            case RW::CORE::tenSubModule::nenPlayback_Simple:
            {
                RW::VPL::tstMyControlStruct *data = static_cast<RW::VPL::tstMyControlStruct*>(*Data);
                data->pstBitStream = this->pOutput;
                break;
            }
            default:
                break;
            }
        }

        CORE::tstModuleVersion CNvDecodeD3D9::ModulVersion() {
            CORE::tstModuleVersion version = { 0, 1 };
            return version;
        }

        CORE::tenSubModule CNvDecodeD3D9::SubModulType()
        {
            return CORE::tenSubModule::nenDecoder_NVIDIA;
        }

        tenStatus CNvDecodeD3D9::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
        {

            tenStatus enStatus = tenStatus::nenSuccess;
            m_Logger->debug("Initialise nenDecoder_NVIDIA");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

            stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);

            if (!data)
            {
                m_Logger->error("DEC_CudaInterop::Initialise: Data of tstMyInitialiseControlStruct is empty!");
                return tenStatus::nenError;
            }

            g_nVideoWidth = data->inputParams->nVideoWidth;
            g_nVideoHeight = data->inputParams->nVideoHeight;

            // Initialize the CUDA and NVCUVID
            typedef HMODULE CUDADRIVER;
            CUDADRIVER hHandleDriver = 0;
            CUresult cuResult;
            cuResult = cuInit(0, __CUDA_API_VERSION, hHandleDriver);
            cuResult = cuvidInit(0);

            if (g_eVideoCreateFlags == cudaVideoCreate_PreferDXVA)
            {
                // preferDXVA will not work with -nointerop mode. Overwrite it.
                g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
            }

            // Initialize CUDA/D3D9 context and other video memory resources
            if (initCudaResources() == E_FAIL)
            {
                g_bException = true;
                g_bWaived = true;
                m_Logger->error("initCudaResources failed!");
                return tenStatus::nenError;
            }

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "DEC_CudaInterop::Initialise: Time to Initialise for nenDecoder_NVIDIA module: " << (RW::CORE::HighResClock::diffMilli(t1, t2).count()) << "ms.";
#endif
            return enStatus;
        }

        tenStatus CNvDecodeD3D9::DoRender(CORE::tstControlStruct * ControlStruct)
        {
            tenStatus enStatus = tenStatus::nenSuccess;

            m_Logger->debug("DoRender nenDecoder_NVIDIA");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
            stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
            if (!data)
            {
                m_Logger->error("DEC_CudaInterop::DoRender: Data of stMyControlStruct is empty!");
                return tenStatus::nenError;
            }
            if (!data->pOutput)
            {
                m_Logger->error("DEC_CudaInterop::DoRender: pOutput of stMyControlStruct is empty!");
                return tenStatus::nenError;
            }

            // instead of cuvidCreateVideoSource in loadVideoSource use cuvidParseVideoData: 
            CUVIDSOURCEDATAPACKET packet;
            memset(&packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
            packet.payload_size = data->pstEncodedStream->u32Size;
            packet.payload = (unsigned char*)data->pstEncodedStream->pBuffer;
            RW::tstPayloadMsg *plMsg = (RW::tstPayloadMsg *)data->pPayload->pBuffer;
            packet.flags = CUVID_PKT_ENDOFSTREAM;
            CUresult oResult = cuvidParseVideoData(g_pVideoParser->hParser_, &packet);
            if ((packet.flags & CUVID_PKT_ENDOFSTREAM) || (oResult != CUDA_SUCCESS))
                g_pFrameQueue->endDecode();

            /////////////////////////////////////////
            enStatus = (g_pCudaModule && g_pVideoDecoder) ? tenStatus::nenSuccess : tenStatus::nenError;

            // On this case we drive the display with a while loop (no openGL calls)
            while (!g_pFrameQueue->isEmpty())
            {
                renderVideoFrame();
            }

            data->pOutput->u32Size = 4 * g_nVideoWidth*g_nVideoHeight * 1;
            {
                FILE *pFile;
                pFile = fopen("C:\\dummy\\dummyYUV0.raw", "wb");
                fwrite(g_pFrameYUV, 1, data->pOutput->u32Size / 4 * 3 / 2, pFile);
                fclose(pFile);
            }
            data->pOutput->pBuffer = g_pFrameYUV;

            //if (data->pstEncodedStream){
            //    delete(data->pstEncodedStream);
            //    data->pstEncodedStream = nullptr;
            //}

            // check if decoding has come to an end.
            // if yes, signal the app to shut down.
            if (g_pFrameQueue->isEndOfDecode() && g_pFrameQueue->isEmpty())
            {
                // Let's free the Frame Data
                if (g_ReadbackSID && g_pFrameYUV)
                {
                    cuMemFreeHost((void *)g_pFrameYUV);
                    g_pFrameYUV = NULL;
                }
            }
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "DEC_CudaInterop::DoRender: Time to DoRender for nenDecoder_NVIDIA module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return enStatus;
        }

        tenStatus CNvDecodeD3D9::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
        {
            m_Logger->debug("Deinitialise nenDecoder_NVIDIA");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

            g_pFrameQueue->endDecode();
            //g_pVideoSource->stop();

            cleanup(g_bWaived ? false : true);

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "DEC_CudaInterop::Deinitialise: Time to Deinitialise for nenDecoder_NVIDIA module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return tenStatus::nenSuccess;
        }

		void CNvDecodeD3D9::initCudaVideo()
		{
			// bind the context lock to the CUDA context
			CUresult result = cuvidCtxLockCreate(&g_CtxLock, g_oContext);

			if (result != CUDA_SUCCESS)
			{
				printf("cuvidCtxLockCreate failed: %d\n", result);
				assert(0);
			}

            std::auto_ptr<FrameQueue> apFrameQueue(new FrameQueue);
            g_pFrameQueue = apFrameQueue.release();

            // Create CUVIDEOFORMAT oFormat manually (needs to be filled by config inputs ...):
            CUVIDEOFORMAT oFormat;
            memset(&oFormat, 0, sizeof(CUVIDEOFORMAT));
            oFormat.codec = cudaVideoCodec_enum::cudaVideoCodec_H264;
            oFormat.progressive_sequence = 1;
            oFormat.coded_width = g_nVideoWidth;
            oFormat.coded_height = g_nVideoHeight;
            oFormat.display_area.left = 0;
            oFormat.display_area.top = 0;
            oFormat.display_area.right = g_nVideoWidth;
            oFormat.display_area.bottom = g_nVideoHeight;
            oFormat.chroma_format = cudaVideoChromaFormat_420;
            oFormat.display_aspect_ratio.x = g_nVideoWidth;
            oFormat.display_aspect_ratio.y = g_nVideoHeight;
            oFormat.video_signal_description.video_format = 5;
            oFormat.video_signal_description.color_primaries = 2;
            oFormat.video_signal_description.transfer_characteristics = 2;
            oFormat.video_signal_description.matrix_coefficients = 2;

			std::auto_ptr<VideoDecoder> apVideoDecoder(new VideoDecoder(oFormat, g_oContext, g_eVideoCreateFlags, g_CtxLock));
			std::auto_ptr<VideoParser> apVideoParser(new VideoParser(apVideoDecoder.get(), g_pFrameQueue));
            
			g_pVideoParser = apVideoParser.release();
			g_pVideoDecoder = apVideoDecoder.release();

            // Create a Stream ID for handling Readback
			checkCudaErrors(cuStreamCreate(&g_ReadbackSID, 0));
			checkCudaErrors(cuStreamCreate(&g_KernelSID, 0));
			printf("> initCudaVideo()\n");
			printf("  CUDA Streams (%s) <g_ReadbackSID = %p>\n", ((g_ReadbackSID == 0) ? "Disabled" : "Enabled"), g_ReadbackSID);
			printf("  CUDA Streams (%s) <g_KernelSID   = %p>\n", ((g_KernelSID == 0) ? "Disabled" : "Enabled"), g_KernelSID);
		}

		void CNvDecodeD3D9::freeCudaResources(bool bDestroyContext)
		{
			if (g_pVideoParser)
			{
				delete g_pVideoParser;
			}

			if (g_pVideoDecoder)
			{
				delete g_pVideoDecoder;
			}

			if (g_pFrameQueue)
			{
				delete g_pFrameQueue;
			}

			if (g_ReadbackSID)
			{
				cuStreamDestroy(g_ReadbackSID);
			}

			if (g_KernelSID)
			{
				cuStreamDestroy(g_KernelSID);
			}

			if (g_CtxLock)
			{
				checkCudaErrors(cuvidCtxLockDestroy(g_CtxLock));
			}

			if (g_oContext && bDestroyContext)
			{
				checkCudaErrors(cuCtxDestroy(g_oContext));
				g_oContext = NULL;
			}
		}

		bool CNvDecodeD3D9::copyDecodedFrameToTexture(unsigned int &nRepeats, int *pbIsProgressive)
		{
			CUVIDPARSERDISPINFO oDisplayInfo;

			if (g_pFrameQueue->dequeue(&oDisplayInfo))
			{
				CCtxAutoLock lck(g_CtxLock);
				// Push the current CUDA context (only if we are using CUDA decoding path)
                checkCudaErrors(cuCtxPushCurrent(g_oContext));

				CUdeviceptr  pDecodedFrame[2] = { 0, 0 };
				CUdeviceptr  pInteropFrame[2] = { 0, 0 };

				*pbIsProgressive = oDisplayInfo.progressive_frame;
				g_bIsProgressive = oDisplayInfo.progressive_frame ? true : false;

				int distinct_fields = 1;
				if (!oDisplayInfo.progressive_frame && oDisplayInfo.repeat_first_field <= 1) {
					distinct_fields = 2;
				}
				nRepeats = oDisplayInfo.repeat_first_field;

				for (int active_field = 0; active_field < distinct_fields; active_field++)
				{
					CUVIDPROCPARAMS oVideoProcessingParameters;
					memset(&oVideoProcessingParameters, 0, sizeof(CUVIDPROCPARAMS));

					oVideoProcessingParameters.progressive_frame = oDisplayInfo.progressive_frame;
					oVideoProcessingParameters.second_field = active_field;
					oVideoProcessingParameters.top_field_first = oDisplayInfo.top_field_first;
					oVideoProcessingParameters.unpaired_field = (distinct_fields == 1);

					unsigned int nDecodedPitch = 0;
					unsigned int nWidth = 0;
					unsigned int nHeight = 0;

					// map decoded video frame to CUDA surfae
					g_pVideoDecoder->mapFrame(oDisplayInfo.picture_index, &pDecodedFrame[active_field], &nDecodedPitch, &oVideoProcessingParameters);
					nWidth = g_pVideoDecoder->targetWidth();
					nHeight = g_pVideoDecoder->targetHeight();

					// map DirectX texture to CUDA surface
					size_t nTexturePitch = 0;

					// If we are Encoding and this is the 1st Frame, we make sure we allocate system memory for readbacks
					if (g_bFirstFrame && g_ReadbackSID)
					{
						CUresult result;
                        checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV, (nWidth * nHeight * 3 / 2)));

						g_bFirstFrame = false;

						if (result != CUDA_SUCCESS)
						{
							printf("cuMemAllocHost returned %d\n", (int)result);
							checkCudaErrors(result);
						}
					}

					// If streams are enabled, we can perform the readback to the host while the kernel is executing
					if (g_ReadbackSID)
					{
                        CUresult result = cuMemcpyDtoHAsync(g_pFrameYUV, pDecodedFrame[active_field], (nWidth * nHeight * 3 / 2), g_ReadbackSID);

						if (result != CUDA_SUCCESS)
						{
							printf("cuMemAllocHost returned %d\n", (int)result);
							checkCudaErrors(result);
						}
					}

#if ENABLE_DEBUG_OUT
					printf("%s = %02d, PicIndex = %02d, OutputPTS = %08d\n",
						(oDisplayInfo.progressive_frame ? "Frame" : "Field"),
						g_DecodeFrameCount, oDisplayInfo.picture_index, oDisplayInfo.timestamp);
#endif

					//pInteropFrame[active_field] = g_pInteropFrame[active_field];
					//nTexturePitch = g_pVideoDecoder->targetWidth() * 2;

					// perform post processing on the CUDA surface (performs colors space conversion and post processing)
					// comment this out if we inclue the line of code seen above
					//cudaPostProcessFrame(&pDecodedFrame[active_field], nDecodedPitch, &pInteropFrame[active_field],
					//	nTexturePitch, g_pCudaModule->getModule(), g_kernelNV12toARGB, g_KernelSID);
                    
					// unmap video frame
					// unmapFrame() synchronizes with the VideoDecode API (ensures the frame has finished decoding)
					g_pVideoDecoder->unmapFrame(pDecodedFrame[active_field]);
					// release the frame, so it can be re-used in decoder
					g_pFrameQueue->releaseFrame(&oDisplayInfo);
					g_DecodeFrameCount++;

				}

				// Detach from the Current thread
				checkCudaErrors(cuCtxPopCurrent(NULL));
			}
			else
			{
				// Frame Queue has no frames, we don't compute FPS until we start
				return false;
			}

			return true;
		}

		HRESULT CNvDecodeD3D9::cleanup(bool bDestroyContext)
		{
			//if (bDestroyContext)
			//{
			//	// Attach the CUDA Context (so we may properly free memroy)
			//	checkCudaErrors(cuCtxPushCurrent(g_oContext));

			//	if (g_pInteropFrame[0])
			//	{
			//		checkCudaErrors(cuMemFree(g_pInteropFrame[0]));
			//	}

			//	if (g_pInteropFrame[1])
			//	{
			//		checkCudaErrors(cuMemFree(g_pInteropFrame[1]));
			//	}

			//	// Detach from the Current thread
			//	checkCudaErrors(cuCtxPopCurrent(NULL));
			//}

			freeCudaResources(bDestroyContext);

			return S_OK;
		}

		void CNvDecodeD3D9::cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
			CUdeviceptr *ppTextureData, size_t nTexturePitch,
			CUmodule cuModNV12toARGB,
			CUfunction fpCudaKernel, CUstream streamID)
		{
			uint32 nWidth = g_pVideoDecoder->targetWidth();
			uint32 nHeight = g_pVideoDecoder->targetHeight();

			// Upload the Color Space Conversion Matrices
			if (g_bUpdateCSC)
			{
				// CCIR 601/709
				float hueColorSpaceMat[9];
				setColorSpaceMatrix(g_eColorSpace, hueColorSpaceMat, g_nHue);
				updateConstantMemory_drvapi(cuModNV12toARGB, hueColorSpaceMat);

				if (!g_bUpdateAll)
				{
					g_bUpdateCSC = false;
				}
			}

			// TODO: Stage for handling video post processing

			// Final Stage: NV12toARGB color space conversion
			CUresult eResult;
			eResult = cudaLaunchNV12toARGBDrv(*ppDecodedFrame, nDecodedPitch,
				*ppTextureData, nTexturePitch,
				nWidth, nHeight, fpCudaKernel, streamID);
		}

		// Launches the CUDA kernels to fill in the texture data
		void CNvDecodeD3D9::renderVideoFrame()
		{
			static unsigned int nRepeatFrame = 0;
            int bIsProgressive = 1;
			bool bFramesDecoded = false;

			if (0 != g_pFrameQueue)
			{
				bFramesDecoded = copyDecodedFrameToTexture(nRepeatFrame, &bIsProgressive);
			}
			else
			{
				return;
			}
		}

		HRESULT CNvDecodeD3D9::initCudaResources()
		{
			CUdevice cuda_device;

			cuda_device = gpuGetMaxGflopsDeviceIdDRV();
			checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));

			// get compute capabilities and the devicename
			int major, minor;
			size_t totalGlobalMem;
			char deviceName[256];
			checkCudaErrors(cuDeviceComputeCapability(&major, &minor, g_oDevice));
			checkCudaErrors(cuDeviceGetName(deviceName, 256, g_oDevice));
			printf("> Using GPU Device %d: %s has SM %d.%d compute capability\n", cuda_device, deviceName, major, minor);

			checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, g_oDevice));
			printf("  Total amount of global memory:     %4.4f MB\n", (float)totalGlobalMem / (1024 * 1024));

    		checkCudaErrors(cuCtxCreate(&g_oContext, CU_CTX_BLOCKING_SYNC, g_oDevice));

			try
			{
				// Initialize CUDA releated Driver API (32-bit or 64-bit), depending the platform running
				if (sizeof(void *) == 4)
				{
                    g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_Win32.ptx", ".", 2, 2, 2);
				}
				else
				{
                    g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_x64.ptx", "C:\\Projekte\\RemoteStreamClient\\build\\x64\\Debug\\", 2, 2, 2);
				}
			}
			catch (char const *p_file)
			{
				// If the CUmoduleManager constructor fails to load the PTX file, it will throw an exception
				printf("\n>> CUmoduleManager::Exception!  %s not found!\n", p_file);
				printf(">> Please rebuild NV12ToARGB_drvapi.cu or re-install this sample.\n");
				return E_FAIL;
			}

			g_pCudaModule->GetCudaFunction("NV12ToARGB_drvapi", &g_kernelNV12toARGB);
			g_pCudaModule->GetCudaFunction("Passthru_drvapi", &g_kernelPassThru);

            /////////////////Change///////////////////////////
            // Now we create the CUDA resources and the CUDA decoder context
            initCudaVideo();

            //checkCudaErrors(cuMemAlloc(&g_pInteropFrame[0], g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 2));
            //checkCudaErrors(cuMemAlloc(&g_pInteropFrame[1], g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 2));

            CUcontext cuCurrent = NULL;
            CUresult result = cuCtxPopCurrent(&cuCurrent);

            if (result != CUDA_SUCCESS)
            {
                printf("cuCtxPopCurrent: %d\n", result);
                assert(0);
            }

            /////////////////////////////////////////
            return ((g_pCudaModule && g_pVideoDecoder) ? S_OK : E_FAIL);
        }
	}
}