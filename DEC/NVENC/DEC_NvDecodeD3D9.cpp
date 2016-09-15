#include "DEC_NvDecodeD3D9.hpp"
#include "..\..\IMP\ConvColor_NV12toRGB\IMP_ConvColorFramesNV12ToRGB.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#endif
#include <..\..\..\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Include\d3dx9.h>

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
        namespace NVENC
        {
            CNvDecodeD3D9::CNvDecodeD3D9(std::shared_ptr<spdlog::logger> Logger) :
                RW::CORE::AbstractModule(Logger)
            {
                g_bFirstFrame = true;
                g_bException = false;
                g_bWaived = false;

                g_CtxLock = NULL;

                cuModNV12toARGB = 0;
                g_kernelNV12toARGB = 0;
                g_kernelPassThru = 0;
                g_oContext = 0;
                g_oDevice = 0;
                //g_ReadbackSID = 0;
                //g_KernelSID = 0;

                g_pFrameYUV = { 0 };

                g_pFrameQueue = 0;
                g_pVideoParser = 0;
                g_pVideoDecoder = 0;

                g_nVideoWidth = 0;
                g_nVideoHeight = 0;

                g_DecodeFrameCount = 0;

                //g_pInteropFrame[0] = 0;
                //g_pInteropFrame[1] = 0;
            }

            CNvDecodeD3D9::~CNvDecodeD3D9()
            {
            }

            void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
            {
                switch (SubModuleType)
                {
                case RW::CORE::tenSubModule::nenGraphic_ColorNV12ToRGB:
                {
                    RW::IMP::COLOR_NV12TORGB::tstMyControlStruct *data = static_cast<RW::IMP::COLOR_NV12TORGB::tstMyControlStruct*>(*Data);
                    data->pInput = new RW::IMP::tstInputOutput;
                    data->pInput->cuDevice = this->pOutput;
                    data->pInput->pBitstream = nullptr;
                    data->pPayload = this->pPayload;
                    break;
                }
                default:
                    break;
                }
                SAFE_DELETE_ARRAY(this->pstEncodedStream->pBuffer);
                SAFE_DELETE(this->pstEncodedStream);
                this->pOutput = 0;
                this->pPayload = nullptr;
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

                g_nVideoWidth = data->inputParams->nWidth;
                g_nVideoHeight = data->inputParams->nHeight;
                g_codec = data->inputParams->codec;

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
                if (!data->pstEncodedStream)
                {
                    m_Logger->error("DEC_CudaInterop::DoRender: pstEncodedStream of stMyControlStruct is empty!");
                    return tenStatus::nenError;
                }
                if (!data->pstEncodedStream->pBuffer)
                {
                    m_Logger->error("DEC_CudaInterop::DoRender: pstEncodedStream->pBuffer of stMyControlStruct is empty!");
                    return tenStatus::nenError;
                }

                // instead of cuvidCreateVideoSource in loadVideoSource use cuvidParseVideoData: 
                CUVIDSOURCEDATAPACKET packet;
                memset(&packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
                packet.payload_size = data->pstEncodedStream->u32Size;
                packet.payload = (unsigned char*)data->pstEncodedStream->pBuffer;
                packet.flags = CUVID_PKT_ENDOFSTREAM;
                CUresult oResult = cuvidParseVideoData(g_pVideoParser->hParser_, &packet);
                if ((packet.flags & CUVID_PKT_ENDOFSTREAM) || (oResult != CUDA_SUCCESS))
                    g_pFrameQueue->endDecode();

                data->pPayload = new tstBitStream;
                data->pPayload->u32Size = packet.payload_size;
                data->pPayload->pBuffer = new uint8[data->pPayload->u32Size];
                memcpy(data->pPayload->pBuffer, packet.payload, data->pPayload->u32Size);

                /////////////////////////////////////////
                enStatus = (g_pVideoDecoder) ? tenStatus::nenSuccess : tenStatus::nenError;

                // On this case we drive the display with a while loop (no openGL calls)
                while (!g_pFrameQueue->isEmpty())
                {
                    renderVideoFrame();
                }

                data->pOutput = g_pFrameYUV;

                //FILE *pFile;
                //pFile = fopen("c:\\dummy\\decoded.raw", "wb");
                //fwrite(data->pOutput->pBuffer, 1, data->pOutput->u32Size, pFile);
                //fclose(pFile);

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "DEC_CudaInterop::DoRender: Time to DoRender for nenDecoder_NVIDIA module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                //cuMemFreeHost((void*)argbArr);

                return enStatus;
            }

            tenStatus CNvDecodeD3D9::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenDecoder_NVIDIA");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

                g_pFrameQueue->endDecode();

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
                oFormat.codec = g_codec;
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
                //checkCudaErrors(cuStreamCreate(&g_ReadbackSID, 0));
                //checkCudaErrors(cuStreamCreate(&g_KernelSID, 0));
                //printf("> initCudaVideo()\n");
                //printf("  CUDA Streams (%s) <g_ReadbackSID = %p>\n", ((g_ReadbackSID == 0) ? "Disabled" : "Enabled"), g_ReadbackSID);
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

                //if (g_ReadbackSID)
                //{
                //	cuStreamDestroy(g_ReadbackSID);
                //}

                //if (g_KernelSID)
                //{
                //    cuStreamDestroy(g_KernelSID);
                //}

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
                    //CUdeviceptr  pInteropFrame[2] = { 0, 0 };

                    *pbIsProgressive = oDisplayInfo.progressive_frame;

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

                        // map decoded video frame to CUDA surfae
                        g_pVideoDecoder->mapFrame(oDisplayInfo.picture_index, &pDecodedFrame[active_field], &g_nDecodedPitch, &oVideoProcessingParameters);

                        // map DirectX texture to CUDA surface
                        //size_t nTexturePitch = 0;
                        size_t pitch;
                        CUresult result = cuMemAllocPitch(&g_pFrameYUV, &pitch, g_nVideoWidth, g_nVideoHeight * 3 / 2, 8);
                        if (result != cudaSuccess)
                        {
                            printf("CNvDecodeD3D9::copyDecodedFrameToTexture: cuMemcpyDtoD failed!");
                            return false;
                        }
                        if (g_nDecodedPitch != pitch)
                        {
                            printf("CNvDecodeD3D9::copyDecodedFrameToTexture: Pitch from dest array and from decoded array are not equal!");
                            return false;
                        }

                        result = cuMemcpyDtoD(g_pFrameYUV, pDecodedFrame[active_field], (g_nDecodedPitch * g_nVideoHeight * 3 / 2));
                        if (result != cudaSuccess)
                        {
                            printf("CNvDecodeD3D9::copyDecodedFrameToTexture: cuMemcpyDtoD failed!");
                            return false;
                        }
                        //g_pFrameYUV = pDecodedFrame[active_field];

#if ENABLE_DEBUG_OUT
                        printf("%s = %02d, PicIndex = %02d, OutputPTS = %08d\n",
                            (oDisplayInfo.progressive_frame ? "Frame" : "Field"),
                            g_DecodeFrameCount, oDisplayInfo.picture_index, oDisplayInfo.timestamp);
#endif

                        ////pInteropFrame[active_field] = g_pInteropFrame[active_field];
                        //nTexturePitch = g_pVideoDecoder->targetWidth() * 2;

                        //// perform post processing on the CUDA surface (performs colors space conversion and post processing)
                        //CUresult eResult = cudaLaunchNV12toARGBDrv(pDecodedFrame[active_field], g_nDecodedPitch,
                        //    g_pInteropFrame[active_field], nTexturePitch,
                        //    nWidth, nHeight, g_kernelNV12toARGB, g_KernelSID);

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
                freeCudaResources(bDestroyContext);

                return S_OK;
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

                //try
                //{
                //	// Initialize CUDA releated Driver API (32-bit or 64-bit), depending the platform running
                //	if (sizeof(void *) == 4)
                //	{
                //                 g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_Win32.ptx", ".", 2, 2, 2);
                //	}
                //	else
                //	{
                //                 g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_x64.ptx", "C:\\Projekte\\RemoteStreamClient\\build\\x64\\Debug\\", 2, 2, 2);
                //	}
                //}
                //catch (char const *p_file)
                //{
                //	// If the CUmoduleManager constructor fails to load the PTX file, it will throw an exception
                //	printf("\n>> CUmoduleManager::Exception!  %s not found!\n", p_file);
                //	printf(">> Please rebuild NV12ToARGB_drvapi.cu or re-install this sample.\n");
                //	return E_FAIL;
                //}

                //g_pCudaModule->GetCudaFunction("NV12ToARGB_drvapi", &g_kernelNV12toARGB);
                //g_pCudaModule->GetCudaFunction("Passthru_drvapi", &g_kernelPassThru);

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
                return ((g_pVideoDecoder) ? S_OK : E_FAIL);
            }
        }
	}
}