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
            mfxU32   videoType;       // MFX_CODEC_AVC for h264, MFX_CODEC_HEVC for h265
            eWorkMode mode;
            MemType  memType;         // SYSTEM_MEMORY or D3D9_MEMORY or D3D11_MEMORY
            bool     bUseHWLib;       // true if application wants to use HW mfx library (platform specific SDK implementation)
            bool     bLowLat;         // low latency mode
            bool     bCalLat;         // latency calculation
            mfxU32   nMaxFPS;         //rendering limited by certain fps
            uint32_t uBitstreamBufferSize; 

            mfxU16  nAsyncDepth;      // depth of asynchronous pipeline. default value is 4. must be between 1 and 20
            mfxU16  gpuCopy;          // GPU Copy mode (three-state option): MFX_GPUCOPY_DEFAULT or MFX_GPUCOPY_ON or MFX_GPUCOPY_OFF

            //MultiThreading only if not Win32 or Win64
            mfxU16  nThreadsNum;      //number of mediasdk task threads
            mfxI32  SchedulingType;   //scheduling type of mediasdk task threads
            mfxI32  Priority;         //priority of mediasdk task threads

            mfxU16  Width;            //output width
            mfxU16  Height;           //output height

            mfxU32  fourcc;           //Output format parameters: MFX_FOURCC_NV12 or MFX_FOURCC_RGB4 or MFX_FOURCC_P010 or MFX_FOURCC_A2RGB10
            mfxU32  nFrames;          //
            mfxU16  eDeinterlace;     //Deinterlace modus

            mfxU16  scrWidth;         //DON'T USE: Screen Width
            mfxU16  scrHeight;        //DON'T USE: Screen Height

            bool    bIsMVC;           //DON'T USE: Multi View
            mfxU32  nWallCell;        //DON'T USE  
            mfxU32  nWallW;           //DON'T USE: number of windows located in each row
            mfxU32  nWallH;           //DON'T USE: number of windows located in each column
            mfxU32  nWallMonitor;     //DON'T USE: monitor id, 0,1,.. etc
            bool    bWallNoTitle;     //DON'T USE: whether to show title for each window with fps value
            mfxU32  nWallTimeout;     //DON'T USE: timeout for -wall option
            mfxU32  numViews;         //DON'T USE: number of views for Multi-View Codec
            mfxU32  nRotation;        //DON'T USE: rotation for Motion JPEG Codec
            bool    bPerfMode;        //DON'T USE  
            bool    bRenderWin;       //DON'T USE  
            mfxU32  nRenderWinX;      //DON'T USE  
            mfxU32  nRenderWinY;      //DON'T USE  

            mfxI32  monitorType;      //DON'T USE: only for LIBVA
#if defined(LIBVA_SUPPORT)
            mfxI32  libvaBackend;
#endif // defined(MFX_LIBVA_SUPPORT)

            msdk_char     strSrcFile[MSDK_MAX_FILENAME_LEN];    //DON'T USE: only for reading data out of a file
            msdk_char     strDstFile[MSDK_MAX_FILENAME_LEN];    //DON'T USE: only for writing data into a file
            sPluginParams pluginParams;

            stInputParams() : videoType(MFX_CODEC_AVC), memType(SYSTEM_MEMORY), bUseHWLib(true), bLowLat(true), bCalLat(true),
                nThreadsNum(0), SchedulingType(0), Priority(0), mode(MODE_PERFORMANCE),
                numViews(1), bIsMVC(false), eDeinterlace(0), bRenderWin(false), nWallCell(0), nWallW(0), nWallH(0), nWallMonitor(0), bWallNoTitle(true), nWallTimeout(0), nRotation(0),
                nMaxFPS(30), uBitstreamBufferSize(2*1024*1024),
                nAsyncDepth(4), gpuCopy(MFX_GPUCOPY_DEFAULT), nFrames(1), Width(0), Height(0), fourcc(MFX_FOURCC_RGB4)
            {}
            ~stInputParams(){}
        }tstInputParams;
    }
}