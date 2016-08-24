#pragma once

#include "DEC_NVENC_inputs.h"
#include "AbstractModule.hpp"

#include "d3d9types.h"
#include "cudaProcessFrame.h"

struct IDirect3D9;
struct IDirect3DDevice9;
class CUmoduleManager;
class FrameQueue;
class VideoParser;
class VideoDecoder;
//class ImageDX;

namespace RW
{
	namespace DEC
	{
        namespace NVENC
        {

            typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
            {
                tstInputParams *inputParams;
            }tstMyInitialiseControlStruct;

            typedef struct stMyControlStruct : public CORE::tstControlStruct
            {
                CUdeviceptr pOutput;
                tstBitStream *pstEncodedStream;
                tstBitStream *pPayload;
                REMOTE_API void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
            }tstMyControlStruct;

            typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
            {
            }tstMyDeinitialiseControlStruct;


            class CNvDecodeD3D9 : public RW::CORE::AbstractModule
            {
                Q_OBJECT

            private:

                //#define VIDEO_SOURCE_FILE "plush1_720p_10s.m2v"

#ifdef _DEBUG
#define ENABLE_DEBUG_OUT    0
#else
#define ENABLE_DEBUG_OUT    0
#endif

                cudaVideoCodec      g_codec;
                bool                g_bFirstFrame;
                bool                g_bException;
                bool                g_bWaived;

                cudaVideoCreateFlags g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
                CUvideoctxlock       g_CtxLock;

                // These are CUDA function pointers to the CUDA kernels
                //CUmoduleManager   *g_pCudaModule;

                CUmodule           cuModNV12toARGB;
                CUfunction         g_kernelNV12toARGB;
                CUfunction         g_kernelPassThru;

                CUcontext          g_oContext;
                CUdevice           g_oDevice;

                //CUstream           g_ReadbackSID;
                //CUstream           g_KernelSID;

                // System Memory surface we want to readback to
                CUdeviceptr        g_pFrameYUV;

                FrameQueue    *g_pFrameQueue;
                VideoParser   *g_pVideoParser;
                VideoDecoder  *g_pVideoDecoder;
                //CUdeviceptr    g_pInteropFrame[2];

                unsigned int g_nVideoWidth;
                unsigned int g_nVideoHeight;
                unsigned int g_nDecodedPitch = 0;

                unsigned int g_DecodeFrameCount;

                void initCudaVideo();

                void freeCudaResources(bool bDestroyContext);

                bool copyDecodedFrameToTexture(unsigned int &nRepeats, int *pbIsProgressive);
                HRESULT cleanup(bool bDestroyContext);
                HRESULT initCudaResources();

                void renderVideoFrame();

            public:

                explicit CNvDecodeD3D9(std::shared_ptr<spdlog::logger> Logger);
                ~CNvDecodeD3D9();

                virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
                virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
                virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;
            };
        }
	}
}