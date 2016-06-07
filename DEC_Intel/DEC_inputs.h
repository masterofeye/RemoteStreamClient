#pragma once

#include "sample_params.h"

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

namespace RW{
	namespace DEC{

		enum MemType {
			SYSTEM_MEMORY = 0x00,
			D3D9_MEMORY = 0x01,
			D3D11_MEMORY = 0x02,
		};

		enum eWorkMode {
			MODE_PERFORMANCE,
			MODE_RENDERING,
			MODE_FILE_DUMP
		};

		typedef struct stInputParams
		{
			mfxU32 videoType; // MFX_CODEC_AVC for h264, MFX_CODEC_HEVC for h265
			eWorkMode mode;
			MemType memType;   // SYSTEM_MEMORY or D3D9_MEMORY or D3D11_MEMORY
			bool    bUseHWLib; // true if application wants to use HW mfx library (platform specific SDK implementation)
			//bool    bIsMVC; // true if Multi-View Codec is in use. Stereoscopic Video Coding 
			bool    bLowLat; // low latency mode
			bool    bCalLat; // latency calculation
			mfxU32  nMaxFPS; //rendering limited by certain fps

			mfxU32  nWallW; //number of windows located in each row
			mfxU32  nWallH; //number of windows located in each column
			mfxU32  nWallCell;    //order of video window in table that will be rendered
			mfxU32  nWallMonitor; //monitor id, 0,1,.. etc
			bool    bWallNoTitle; //whether to show title for each window with fps value
			mfxU32  nWallTimeout; //timeout for -wall option

			mfxU32  numViews; // number of views for Multi-View Codec
			//mfxU32  nRotation; // rotation for Motion JPEG Codec

			mfxU16  nAsyncDepth; // depth of asynchronous pipeline. default value is 4. must be between 1 and 20
			mfxU16  gpuCopy; // GPU Copy mode (three-state option): MFX_GPUCOPY_DEFAULT or MFX_GPUCOPY_ON or MFX_GPUCOPY_OFF

			//MultiThreading only on Win32 or Win64
			mfxU16  nThreadsNum;  //number of mediasdk task threads
			mfxI32  SchedulingType;  //scheduling type of mediasdk task threads
			mfxI32  Priority;  //priority of mediasdk task threads

			mfxU16  scrWidth;  //screen resolution width
			mfxU16  scrHeight;  //screen resolution height

			mfxU16  Width;  //output width
			mfxU16  Height;  //output height

			mfxU32  fourcc;  //Output format parameters: MFX_FOURCC_NV12 or MFX_FOURCC_RGB4 or MFX_FOURCC_P010 or MFX_FOURCC_A2RGB10
			mfxU32  nFrames;
			mfxU16  eDeinterlace;  //enable deinterlacing BOB/ADI; MFX_DEINTERLACING_BOB or MFX_DEINTERLACING_ADVANCED; 

			bool    bRenderWin;

#if defined(LIBVA_SUPPORT) 
			mfxI32  libvaBackend;  //MFX_LIBVA_X11 or MFX_LIBVA_WAYLAND or MFX_LIBVA_DRM_MODESET; only works with memType = D3D9_MEMORY and mode = MODE_RENDERING
			mfxU32  nRenderWinX;
			mfxU32  nRenderWinY;

			mfxI32  monitorType;  //has to be below MFX_MONITOR_MAXNUMBER
			bool    bPerfMode;
#endif // defined(MFX_LIBVA_SUPPORT)

			msdk_char     strSrcFile[MSDK_MAX_FILENAME_LEN];
			msdk_char     strDstFile[MSDK_MAX_FILENAME_LEN];
			sPluginParams pluginParams;

			//stInputParams()
			//{
			//    MSDK_ZERO_MEMORY(*this);
			//}
			stInputParams() : videoType(MFX_CODEC_AVC), mode(MODE_PERFORMANCE), memType(D3D9_MEMORY), bUseHWLib(true), bLowLat(true), bCalLat(true),
                nThreadsNum(1), SchedulingType(NORMAL_PRIORITY_CLASS), Priority(THREAD_PRIORITY_NORMAL),
				nAsyncDepth(4), gpuCopy(MFX_GPUCOPY_DEFAULT), fourcc(MFX_FOURCC_RGB4), nFrames(0), eDeinterlace(MFX_DEINTERLACING_BOB), bRenderWin(false)
			{}
			~stInputParams(){}
		}tstInputParams;
	}
}