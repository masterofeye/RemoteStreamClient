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

        typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
        {
            tstInputParams *inputParams;
        }tstMyInitialiseControlStruct;

        typedef struct stMyControlStruct : public CORE::tstControlStruct
        {
            tstBitStream *pOutput;
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

			bool                g_bFirstFrame;
			bool                g_bUpdateCSC;
			bool                g_bUpdateAll;
			bool                g_bUseDisplay; // this flag enables/disables video on the window
			bool                g_bReadback; // this flag enables/disables reading back of a video from a window
			bool                g_bWriteFile; // this flag enables/disables writing of a file
			bool                g_bIsProgressive; // assume it is progressive, unless otherwise noted
			bool                g_bException;
			bool                g_bWaived;

			int                 g_iRepeatFactor = 1; // 1:1 assumes no frame repeats

			cudaVideoCreateFlags g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
			CUvideoctxlock       g_CtxLock;

			// These are CUDA function pointers to the CUDA kernels
			CUmoduleManager   *g_pCudaModule;

			CUmodule           cuModNV12toARGB;
			CUfunction         g_kernelNV12toARGB;
			CUfunction         g_kernelPassThru;

			CUcontext          g_oContext;
			CUdevice           g_oDevice;

			CUstream           g_ReadbackSID, g_KernelSID;

			eColorSpace        g_eColorSpace;
			float              g_nHue;

			// System Memory surface we want to readback to
			BYTE          *g_pFrameYUV[4];
			FrameQueue    *g_pFrameQueue;
			//VideoSource   *g_pVideoSource;
			VideoParser   *g_pVideoParser;
			VideoDecoder  *g_pVideoDecoder;

			//ImageDX       *g_pImageDX = 0;
			CUdeviceptr    g_pInteropFrame[2]; // if we're using CUDA malloc

			unsigned int g_nVideoWidth;
			unsigned int g_nVideoHeight;

			unsigned int g_DecodeFrameCount;

			void initCudaVideo();

			void freeCudaResources(bool bDestroyContext);

			bool copyDecodedFrameToTexture(unsigned int &nRepeats, int *pbIsProgressive);
			void cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
				CUdeviceptr *ppTextureData, size_t nTexturePitch,
				CUmodule cuModNV12toARGB,
				CUfunction fpCudaKernel, CUstream streamID);
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