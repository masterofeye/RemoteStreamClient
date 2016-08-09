#pragma once


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
#include "VideoSource.h"
#include "VideoParser.h"
#include "VideoDecoder.h"
#include "ImageDX.h"

#include "cudaProcessFrame.h"
#include "cudaModuleMgr.h"

#include "DEC_NVENC_inputs.h"
#include "AbstractModule.hpp"

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

			//StopWatchInterface *frame_timer;
			//StopWatchInterface *global_timer;

			int                 g_DeviceID;
			bool                g_bWindowed;
			bool                g_bDeviceLost;
			bool                g_bUseVsync;
			bool                g_bFrameRepeat;
			bool                g_bFrameStep;
			bool                g_bQAReadback;
			bool                g_bFirstFrame;
            bool                g_bDone;
			bool                g_bLoop;
			bool                g_bUpdateCSC;
			bool                g_bUpdateAll;
			bool                g_bUseDisplay; // this flag enables/disables video on the window
			bool                g_bUseInterop;
			bool                g_bReadback; // this flag enables/disables reading back of a video from a window
			bool                g_bWriteFile; // this flag enables/disables writing of a file
			bool                g_bIsProgressive; // assume it is progressive, unless otherwise noted
			bool                g_bException;
			bool                g_bWaived;

			int                 g_iRepeatFactor = 1; // 1:1 assumes no frame repeats

			cudaVideoCreateFlags g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
			CUvideoctxlock       g_CtxLock;

			float present_fps, decoded_fps, total_time;

			D3DDISPLAYMODE        g_d3ddm;
			D3DPRESENT_PARAMETERS g_d3dpp;

			IDirect3D9        *g_pD3D; // Used to create the D3DDevice
			IDirect3DDevice9  *g_pD3DDevice;

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

			ImageDX       *g_pImageDX = 0;
			CUdeviceptr    g_pInteropFrame[2]; // if we're using CUDA malloc

			char exec_path[256];

			unsigned int g_nWindowWidth;
			unsigned int g_nWindowHeight;

			unsigned int g_nVideoWidth;
			unsigned int g_nVideoHeight;

			unsigned int g_FrameCount;
			unsigned int g_DecodeFrameCount;
			unsigned int g_fpsCount;      // FPS count for averaging
			unsigned int g_fpsLimit;     // FPS limit for sampling timer;

			// Forward declarations
			bool    initD3D9(int *pbTCC);
			HRESULT initD3D9Surface(unsigned int nWidth, unsigned int nHeight);
			HRESULT freeDestSurface();

			void initCudaVideo(RW::tstBitStream *encodedStream);

			void freeCudaResources(bool bDestroyContext);

			bool copyDecodedFrameToTexture(unsigned int &nRepeats, int *pbIsProgressive);
			void cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
				CUdeviceptr *ppTextureData, size_t nTexturePitch,
				CUmodule cuModNV12toARGB,
				CUfunction fpCudaKernel, CUstream streamID);
			HRESULT drawScene(int field_num);
			HRESULT cleanup(bool bDestroyContext);
			HRESULT initCudaResources(int bUseInterop, int bTCC);

			void renderVideoFrame(bool bUseInterop);
			void SaveFrameAsYUV(unsigned char *pdst, const unsigned char *psrc, int width, int height, int pitch);

            public:
			CNvDecodeD3D9(std::shared_ptr<spdlog::logger> m_Logger);
			virtual ~CNvDecodeD3D9();

            virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
            virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
            virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;


			protected:
				std::shared_ptr<spdlog::logger> m_Logger;
		};
	}
}