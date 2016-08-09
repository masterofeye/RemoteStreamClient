#include "DEC_NvDecodeD3D9.hpp"
#include "..\VPL_QT\VPL_FrameProcessor.hpp"

namespace RW
{
	namespace DEC
	{
        CNvDecodeD3D9::CNvDecodeD3D9(std::shared_ptr<spdlog::logger> Logger) :
            RW::CORE::AbstractModule(Logger)
		{
			g_DeviceID = 0;
			g_bDeviceLost = false;
			g_bUseVsync = false;
			g_bFrameRepeat = false;
			g_bFrameStep = false;
			g_bQAReadback = false;
			g_bFirstFrame = true;
			g_bLoop = false;
			g_bUpdateCSC = true;
			g_bUpdateAll = false;
			g_bUseDisplay = true; 
			g_bUseInterop = true;
			g_bIsProgressive = true;
			g_bException = false;
			g_bWaived = false;

			g_iRepeatFactor = 1; 

			g_CtxLock = NULL;
			total_time = 0.0f;

			cuModNV12toARGB = 0;
			g_kernelNV12toARGB = 0;
			g_kernelPassThru = 0;
			g_oContext = 0;
			g_oDevice = 0;
			g_ReadbackSID = 0, g_KernelSID = 0;
			g_eColorSpace = ITU601;
			g_nHue = 0.0f;

			g_pFrameYUV[0] = { 0 };
			g_pFrameYUV[1] = { 0 };
			g_pFrameYUV[2] = { 0 };
			g_pFrameYUV[3] = { 0 };

			g_pFrameQueue = 0;
			g_pVideoParser = 0;
			g_pVideoDecoder = 0;
			g_pImageDX = 0;

			g_pInteropFrame[0] = { 0 };
			g_pInteropFrame[1] = { 1 };

			g_nWindowWidth = 0;
			g_nWindowHeight = 0;

			g_nVideoWidth = 0;
			g_nVideoHeight = 0;

			g_FrameCount = 0;
			g_DecodeFrameCount = 0;
			g_fpsCount = 0;      
			g_fpsLimit = 16;   

		}

		CNvDecodeD3D9::~CNvDecodeD3D9()
		{
		}


        CORE::tstModuleVersion CNvDecodeD3D9::ModulVersion() {
            CORE::tstModuleVersion version = { 0, 1 };
            return version;
        }

        CORE::tenSubModule CNvDecodeD3D9::SubModulType()
        {
            return CORE::tenSubModule::nenDecoder_INTEL;
        }

        tenStatus CNvDecodeD3D9::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
        {

            tenStatus enStatus = tenStatus::nenSuccess;
            m_Logger->debug("Initialise nenDecoder_NVENC");
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

            // Find out the video size
            //g_bIsProgressive = loadVideoSource(sFileName.c_str(),
            //	g_nVideoWidth, g_nVideoHeight,
            //	g_nWindowWidth, g_nWindowHeight);

            int bTCC = 0;

            if (g_bUseInterop)
            {
                // Initialize Direct3D
                if (initD3D9(&bTCC) == false)
                {
                    g_bWaived = true;
                    m_Logger->error("initD3D9 failed!");
                    return tenStatus::nenError;
                }
            }

            if (!g_bUseInterop && g_eVideoCreateFlags == cudaVideoCreate_PreferDXVA)
            {
                // preferDXVA will not work with -nointerop mode. Overwrite it.
                g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
            }

            // If we are using TCC driver, then graphics interop must be disabled
            if (bTCC)
            {
                g_bUseInterop = false;
            }

            // Initialize CUDA/D3D9 context and other video memory resources
            if (initCudaResources(g_bUseInterop, bTCC) == E_FAIL)
            {
                g_bException = true;
                g_bWaived = true;
                m_Logger->error("initCudaResources failed!");
                return tenStatus::nenError;
            }

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "DEC_CudaInterop::Initialise: Time to Initialise for nenDecoder_INTEL module: " << (RW::CORE::HighResClock::diffMilli(t1, t2).count()) << "ms.";
#endif
            return enStatus;
        }

        tenStatus CNvDecodeD3D9::DoRender(CORE::tstControlStruct * ControlStruct)
        {
            tenStatus enStatus = tenStatus::nenSuccess;

            m_Logger->debug("DoRender nenDecoder_NVENC");
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

            /////////////////Change///////////////////////////
            // Now we create the CUDA resources and the CUDA decoder context
            initCudaVideo(data->pstEncodedStream);

            if (g_bUseInterop)
            {
                initD3D9Surface(g_pVideoDecoder->targetWidth(),
                    g_pVideoDecoder->targetHeight());
            }
            else
            {
                checkCudaErrors(cuMemAlloc(&g_pInteropFrame[0], g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 2));
                checkCudaErrors(cuMemAlloc(&g_pInteropFrame[1], g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 2));
            }

            CUcontext cuCurrent = NULL;
            CUresult result = cuCtxPopCurrent(&cuCurrent);

            if (result != CUDA_SUCCESS)
            {
                printf("cuCtxPopCurrent: %d\n", result);
                assert(0);
            }

            /////////////////////////////////////////
            enStatus = (g_pCudaModule && g_pVideoDecoder && (g_pImageDX || g_pInteropFrame[0])) ? tenStatus::nenSuccess : tenStatus::nenError;

            // On this case we drive the display with a while loop (no openGL calls)
            while (!g_bDone)
            {
                renderVideoFrame(g_bUseInterop);
            }

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "DEC_CudaInterop::DoRender: Time to DoRender for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return enStatus;
        }

        tenStatus CNvDecodeD3D9::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
        {
            m_Logger->debug("Deinitialise nenDecoder_NVENC");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

            g_pFrameQueue->endDecode();
            //g_pVideoSource->stop();

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "DEC_CudaInterop::Deinitialise: Time to Deinitialise for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return tenStatus::nenSuccess;
        }

        bool CNvDecodeD3D9::initD3D9(int *pbTCC)
		{
			int dev, device_count = 0;
			bool bSpecifyDevice = false;
			char device_name[256];

			// Check for a min spec of Compute 1.1 capability before running
			checkCudaErrors(cuDeviceGetCount(&device_count));

			// If deviceID == 0, and there is more than 1 device, let's find the first available graphics GPU
			if (!bSpecifyDevice && device_count > 0)
			{
				for (int i = 0; i < device_count; i++)
				{
					checkCudaErrors(cuDeviceGet(&dev, i));
					checkCudaErrors(cuDeviceGetName(device_name, 256, dev));

					int bSupported = checkCudaCapabilitiesDRV(1, 1, i);

					if (!bSupported)
					{
						printf("  -> GPU: \"%s\" does not meet the minimum spec of SM 1.1\n", device_name);
						printf("  -> A GPU with a minimum compute capability of SM 1.1 or higher is required.\n");
						return false;
					}

					checkCudaErrors(cuDeviceGetAttribute(pbTCC, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev));
					printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");

					if (*pbTCC)
					{
						g_bUseInterop = false;
						continue;
					}
					else
					{
						g_DeviceID = i; // we choose an available WDDM display device
					}

					printf("\n");
				}
			}
			else
			{
				if ((g_DeviceID > (device_count - 1)) || (g_DeviceID < 0))
				{
					printf(" >>> Invalid GPU Device ID=%d specified, only %d GPU device(s) are available.<<<\n", g_DeviceID, device_count);
					printf(" >>> Valid GPU ID (n) range is between [%d,%d]...  Exiting... <<<\n", 0, device_count - 1);
					return false;
				}

				// We are specifying a GPU device, check to see if it is TCC or not
				checkCudaErrors(cuDeviceGet(&dev, g_DeviceID));
				checkCudaErrors(cuDeviceGetName(device_name, 256, dev));

				checkCudaErrors(cuDeviceGetAttribute(pbTCC, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev));
				printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");

				if (*pbTCC)
				{
					g_bUseInterop = false;
				}
			}

			HRESULT eResult = S_OK;

			if (g_bUseInterop)
			{
				// Create the D3D object.
				if (NULL == (g_pD3D = Direct3DCreate9(D3D_SDK_VERSION)))
				{
					return false;
				}

				// Get primary display identifier
				D3DADAPTER_IDENTIFIER9 adapterId;
				bool bDeviceFound = false;
				int device;

				// Find the first CUDA capable device
				CUresult cuStatus;

				for (unsigned int g_iAdapter = 0; g_iAdapter < g_pD3D->GetAdapterCount(); g_iAdapter++)
				{
					HRESULT hr = g_pD3D->GetAdapterIdentifier(g_iAdapter, 0, &adapterId);

					if (FAILED(hr))
					{
						continue;
					}

					cuStatus = cuD3D9GetDevice(&device, adapterId.DeviceName);
					printf("> Display Device: \"%s\" %s Direct3D9\n",
						adapterId.Description,
						(cuStatus == cudaSuccess) ? "supports" : "does not support");

					if (cudaSuccess == cuStatus)
					{
						bDeviceFound = true;
						break;
					}
				}

				// we check to make sure we have found a cuda-compatible D3D device to work on
				if (!bDeviceFound)
				{
					printf("\n");
					printf("  No CUDA-compatible Direct3D9 device available\n");
					// destroy the D3D device
					g_pD3D->Release();
					g_pD3D = NULL;
					return false;
				}

				// Create the D3D Display Device

				g_pD3D->GetAdapterDisplayMode(D3DADAPTER_DEFAULT, &g_d3ddm);
				ZeroMemory(&g_d3dpp, sizeof(g_d3dpp));

				g_d3dpp.Windowed = (g_bQAReadback ? TRUE : g_bWindowed); // fullscreen or windowed?

				g_d3dpp.BackBufferCount = 1;
				g_d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
				g_d3dpp.hDeviceWindow = NULL;
                g_d3dpp.BackBufferWidth = g_nVideoWidth;// rc.right - rc.left;
                g_d3dpp.BackBufferHeight = g_nVideoHeight;// rc.bottom - rc.top;
				g_d3dpp.BackBufferFormat = g_d3ddm.Format;
				g_d3dpp.FullScreen_RefreshRateInHz = 0; // set to 60 for fullscreen, and also don't forget to set Windowed to FALSE
				g_d3dpp.PresentationInterval = (g_bUseVsync ? D3DPRESENT_INTERVAL_ONE : D3DPRESENT_INTERVAL_IMMEDIATE);

				if (g_bQAReadback)
				{
					g_d3dpp.Flags = D3DPRESENTFLAG_VIDEO;    // turn off vsync
				}

				eResult = g_pD3D->CreateDevice(0, D3DDEVTYPE_HAL, NULL,
					D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
					&g_d3dpp, &g_pD3DDevice);
			}
			else
			{
				fprintf(stderr, "> NVDecodeD3D9 is decoding w/o visualization\n");
				eResult = S_OK;
			}

			return (eResult == S_OK);
		}

		HRESULT CNvDecodeD3D9::initD3D9Surface(unsigned int nWidth, unsigned int nHeight)
		{
			g_pImageDX = new ImageDX(g_pD3DDevice,
				nWidth, nHeight,
				nWidth, nHeight,
				g_bIsProgressive,
				ImageDX::BGRA_PIXEL_FORMAT); // ImageDX::LUMINANCE_PIXEL_FORMAT
			g_pImageDX->clear(0x80);

			g_pImageDX->setCUDAcontext(g_oContext);
			g_pImageDX->setCUDAdevice(g_oDevice);

			return S_OK;
		}

		HRESULT CNvDecodeD3D9::freeDestSurface()
		{
			if (g_pImageDX)
			{
				delete g_pImageDX;
				g_pImageDX = NULL;
			}

			return S_OK;
		}

		void CNvDecodeD3D9::initCudaVideo(RW::tstBitStream *encodedStream)
		{
			// bind the context lock to the CUDA context
			CUresult result = cuvidCtxLockCreate(&g_CtxLock, g_oContext);

			if (result != CUDA_SUCCESS)
			{
				printf("cuvidCtxLockCreate failed: %d\n", result);
				assert(0);
			}

            // Create CUVIDEOFORMAT oFormat manually (needs to be filled by config inputs ...):
            CUVIDEOFORMAT oFormat;
            oFormat.codec = cudaVideoCodec_enum::cudaVideoCodec_H264;
            oFormat.frame_rate.numerator = 30000;
            oFormat.frame_rate.denominator = 1000;
            oFormat.progressive_sequence = 1;
            oFormat.coded_width = g_nVideoWidth;
            oFormat.coded_height = g_nVideoHeight;
            oFormat.display_area.left = 0;
            oFormat.display_area.top = 0;
            oFormat.display_area.right = g_nVideoWidth;
            oFormat.display_area.bottom = g_nVideoHeight;
            oFormat.chroma_format = cudaVideoChromaFormat_420;
            oFormat.display_aspect_ratio.x = g_nWindowWidth;
            oFormat.display_aspect_ratio.y = g_nWindowHeight;
            oFormat.video_signal_description.video_format = 5;
            oFormat.video_signal_description.color_primaries = 2;
            oFormat.video_signal_description.transfer_characteristics = 2;
            oFormat.video_signal_description.matrix_coefficients = 2;

			std::auto_ptr<VideoDecoder> apVideoDecoder(new VideoDecoder(oFormat, g_oContext, g_eVideoCreateFlags, g_CtxLock));

            //instead of cuvidCreateVideoSource in loadVideoSource use cuvidParseVideoData: 
            CUVIDSOURCEDATAPACKET packet = {};
            packet.payload_size = encodedStream->u32Size;
            packet.payload = (unsigned char*)encodedStream->pBuffer;
            cuvidParseVideoData(g_pVideoParser, &packet);

			std::auto_ptr<VideoParser> apVideoParser(new VideoParser(apVideoDecoder.get(), g_pFrameQueue));
            //g_pVideoSource->setParser(*apVideoParser.get()); 

			g_pVideoParser = apVideoParser.release();
			g_pVideoDecoder = apVideoDecoder.release();

			// Create a Stream ID for handling Readback
			if (g_bReadback)
			{
				checkCudaErrors(cuStreamCreate(&g_ReadbackSID, 0));
				checkCudaErrors(cuStreamCreate(&g_KernelSID, 0));
				printf("> initCudaVideo()\n");
				printf("  CUDA Streams (%s) <g_ReadbackSID = %p>\n", ((g_ReadbackSID == 0) ? "Disabled" : "Enabled"), g_ReadbackSID);
				printf("  CUDA Streams (%s) <g_KernelSID   = %p>\n", ((g_KernelSID == 0) ? "Disabled" : "Enabled"), g_KernelSID);
			}
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
				CUresult result = cuCtxPushCurrent(g_oContext);

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
					if (g_bReadback && g_bFirstFrame && g_ReadbackSID)
					{
						CUresult result;
						checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[0], (nDecodedPitch * nHeight + nDecodedPitch*nHeight / 2)));
						checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[1], (nDecodedPitch * nHeight + nDecodedPitch*nHeight / 2)));
						checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[2], (nDecodedPitch * nHeight + nDecodedPitch*nHeight / 2)));
						checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[3], (nDecodedPitch * nHeight + nDecodedPitch*nHeight / 2)));

						g_bFirstFrame = false;

						if (result != CUDA_SUCCESS)
						{
							printf("cuMemAllocHost returned %d\n", (int)result);
							checkCudaErrors(result);
						}
					}

					// If streams are enabled, we can perform the readback to the host while the kernel is executing
					if (g_bReadback && g_ReadbackSID)
					{
						CUresult result = cuMemcpyDtoHAsync(g_pFrameYUV[active_field], pDecodedFrame[active_field], (nDecodedPitch * nHeight * 3 / 2), g_ReadbackSID);

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

					if (g_pImageDX)
					{
						// map the texture surface
						g_pImageDX->map(&pInteropFrame[active_field], &nTexturePitch, active_field);
					}
					else
					{
						pInteropFrame[active_field] = g_pInteropFrame[active_field];
						nTexturePitch = g_pVideoDecoder->targetWidth() * 2;
					}

					// perform post processing on the CUDA surface (performs colors space conversion and post processing)
					// comment this out if we inclue the line of code seen above
					cudaPostProcessFrame(&pDecodedFrame[active_field], nDecodedPitch, &pInteropFrame[active_field],
						nTexturePitch, g_pCudaModule->getModule(), g_kernelNV12toARGB, g_KernelSID);

					if (g_pImageDX)
					{
						// unmap the texture surface
						g_pImageDX->unmap(active_field);
					}

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

			// check if decoding has come to an end.
			// if yes, signal the app to shut down.
			if (g_pFrameQueue->isEndOfDecode() && g_pFrameQueue->isEmpty())
			{
				// Let's free the Frame Data
				if (g_ReadbackSID && g_pFrameYUV)
				{
					cuMemFreeHost((void *)g_pFrameYUV[0]);
					cuMemFreeHost((void *)g_pFrameYUV[1]);
					cuMemFreeHost((void *)g_pFrameYUV[2]);
					cuMemFreeHost((void *)g_pFrameYUV[3]);
					g_pFrameYUV[0] = NULL;
					g_pFrameYUV[1] = NULL;
					g_pFrameYUV[2] = NULL;
					g_pFrameYUV[3] = NULL;
				}
			}

			return true;
		}

		void CNvDecodeD3D9::SaveFrameAsYUV(unsigned char *pdst, const unsigned char *psrc, int width, int height, int pitch)
		{

            FILE *fpWriteYUV = NULL;
            fpWriteYUV = fopen("dummy.yuv", "wb");
            if (fpWriteYUV == NULL)
            {
                printf("Error opening file dummy.yuv\n");
            }

			int x, y, width_2, height_2;
			int xy_offset = width*height;
			int uvoffs = (width / 2)*(height / 2);
			const unsigned char *py = psrc;
			const unsigned char *puv = psrc + height*pitch;

			// luma
			for (y = 0; y<height; y++)
			{
				memcpy(&pdst[y*width], py, width);
				py += pitch;
			}

			// De-interleave chroma
			width_2 = width >> 1;
			height_2 = height >> 1;
			for (y = 0; y<height_2; y++)
			{
				for (x = 0; x<width_2; x++)
				{
					pdst[xy_offset + y*(width_2)+x] = puv[x * 2];
					pdst[xy_offset + uvoffs + y*(width_2)+x] = puv[x * 2 + 1];
				}
				puv += pitch;
			}

			fwrite(pdst, 1, width*height + (width*height) / 2, fpWriteYUV);
		}

		HRESULT CNvDecodeD3D9::cleanup(bool bDestroyContext)
		{
			if (bDestroyContext)
			{
				// Attach the CUDA Context (so we may properly free memroy)
				checkCudaErrors(cuCtxPushCurrent(g_oContext));

				if (g_pInteropFrame[0])
				{
					checkCudaErrors(cuMemFree(g_pInteropFrame[0]));
				}

				if (g_pInteropFrame[1])
				{
					checkCudaErrors(cuMemFree(g_pInteropFrame[1]));
				}

				// Detach from the Current thread
				checkCudaErrors(cuCtxPopCurrent(NULL));
			}

			if (g_pImageDX)
			{
				delete g_pImageDX;
				g_pImageDX = NULL;
			}

			freeCudaResources(bDestroyContext);

			// destroy the D3D device
			if (g_pD3DDevice)
			{
				g_pD3DDevice->Release();
				g_pD3DDevice = NULL;
			}

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

		HRESULT CNvDecodeD3D9::drawScene(int field_num)
		{
			HRESULT hr = S_OK;

			// Begin code to handle case where the D3D gets lost
			if (g_bDeviceLost)
			{
				// test the cooperative level to see if it's okay
				// to render
				if (FAILED(hr = g_pD3DDevice->TestCooperativeLevel()))
				{
					fprintf(stderr, "TestCooperativeLevel = %08x failed, will attempt to reset\n", hr);

					// if the device was truly lost, (i.e., a fullscreen device just lost focus), wait
					// until we get it back

					if (hr == D3DERR_DEVICELOST)
					{
						fprintf(stderr, "TestCooperativeLevel = %08x DeviceLost, will retry next call\n", hr);
						return S_OK;
					}

					// eventually, we will get this return value,
					// indicating that we can now reset the device
					if (hr == D3DERR_DEVICENOTRESET)
					{
						fprintf(stderr, "TestCooperativeLevel = %08x will try to RESET the device\n", hr);
						// if we are windowed, read the desktop mode and use the same format for
						// the back buffer; this effectively turns off color conversion

						if (g_bWindowed)
						{
							g_pD3D->GetAdapterDisplayMode(D3DADAPTER_DEFAULT, &g_d3ddm);
							g_d3dpp.BackBufferFormat = g_d3ddm.Format;
						}

						// now try to reset the device
						if (FAILED(hr = g_pD3DDevice->Reset(&g_d3dpp)))
						{
							fprintf(stderr, "TestCooperativeLevel = %08x RESET device FAILED\n", hr);
							return hr;
						}
						else
						{
							fprintf(stderr, "TestCooperativeLevel = %08x RESET device SUCCESS!\n", hr);
							// Reinit All other resources including CUDA contexts
							initCudaResources(true, false);
							fprintf(stderr, "TestCooperativeLevel = %08x INIT device SUCCESS!\n", hr);

							// we have acquired the device
							g_bDeviceLost = false;
						}
					}
				}
			}

			// Normal D3D9 rendering code
			if (!g_bDeviceLost)
			{

				// init the scene
				if (g_bUseDisplay)
				{
					g_pD3DDevice->BeginScene();
					g_pD3DDevice->Clear(0, NULL, D3DCLEAR_TARGET, 0, 1.0f, 0);
					// render image
					g_pImageDX->render(field_num);
					// end the scene
					g_pD3DDevice->EndScene();
				}

				hr = g_pD3DDevice->Present(NULL, NULL, NULL, NULL);

				if (hr == D3DERR_DEVICELOST)
				{
					fprintf(stderr, "drawScene Present = %08x detected D3D DeviceLost\n", hr);
					g_bDeviceLost = true;

					// We check for DeviceLost, if that happens we want to release resources
					g_pFrameQueue->endDecode();

					freeCudaResources(false);
				}
			}
			else
			{
				fprintf(stderr, "drawScene (DeviceLost == TRUE), waiting\n");
			}

			return S_OK;
		}

		// Launches the CUDA kernels to fill in the texture data
		void CNvDecodeD3D9::renderVideoFrame(bool bUseInterop)
		{
			static unsigned int nRepeatFrame = 0;
			int repeatFactor = g_iRepeatFactor;
			int bIsProgressive = 1, bFPSComputed = 0;
			bool bFramesDecoded = false;

			if (0 != g_pFrameQueue)
			{
				// if not running, we simply don't copy new frames from the decoder
				if (!g_bDeviceLost)
				{
					bFramesDecoded = copyDecodedFrameToTexture(nRepeatFrame, &bIsProgressive);
				}
			}
			else
			{
				return;
			}
		}

		HRESULT CNvDecodeD3D9::initCudaResources(int bUseInterop, int bTCC)
		{
			HRESULT hr = S_OK;

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

			// Create CUDA Device w/ D3D9 interop (if WDDM), otherwise CUDA w/o interop (if TCC)
			// (use CU_CTX_BLOCKING_SYNC for better CPU synchronization)
			if (bUseInterop)
			{
				checkCudaErrors(cuD3D9CtxCreate(&g_oContext, &g_oDevice, CU_CTX_BLOCKING_SYNC, g_pD3DDevice));
			}
			else
			{
				checkCudaErrors(cuCtxCreate(&g_oContext, CU_CTX_BLOCKING_SYNC, g_oDevice));
			}

			try
			{
				// Initialize CUDA releated Driver API (32-bit or 64-bit), depending the platform running
				if (sizeof(void *) == 4)
				{
					g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_Win32.ptx", exec_path, 2, 2, 2);
				}
				else
				{
					g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_x64.ptx", exec_path, 2, 2, 2);
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

			/////////////////////////////////////////
			return (g_pCudaModule ? S_OK : E_FAIL);
		}
	}
}