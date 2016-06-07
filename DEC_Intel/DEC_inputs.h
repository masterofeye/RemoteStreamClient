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

        typedef struct stInputParams
        {
            mfxU32 videoType; // MFX_CODEC_AVC for h264, MFX_CODEC_HEVC for h265
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

            mfxU16  Width;  //output width
            mfxU16  Height;  //output height

            mfxU32  fourcc;  //Output format parameters: MFX_FOURCC_NV12 or MFX_FOURCC_RGB4 or MFX_FOURCC_P010 or MFX_FOURCC_A2RGB10
            mfxU32  nFrames;

            sPluginParams pluginParams;

            //stInputParams()
            //{
            //    MSDK_ZERO_MEMORY(*this);
            //}
            stInputParams() : videoType(MFX_CODEC_AVC), memType(SYSTEM_MEMORY), bUseHWLib(true), bLowLat(true), bCalLat(true),
                nThreadsNum(0), SchedulingType(0), Priority(0), nMaxFPS(0), nWallCell(0), nWallW(0), nWallH(0), nWallMonitor(0), bWallNoTitle(false), numViews(0), 
                nAsyncDepth(4), gpuCopy(MFX_GPUCOPY_DEFAULT), nFrames(0), Width(0), Height(0), fourcc(0)
            {}
            ~stInputParams(){}
        }tstInputParams;
    }
}