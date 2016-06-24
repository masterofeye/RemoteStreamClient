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
            mfxU32   videoType;       // MFX_CODEC_AVC for h264, MFX_CODEC_HEVC for h265
            MemType  memType;         // SYSTEM_MEMORY or D3D9_MEMORY or D3D11_MEMORY
            bool     bUseHWLib;       // true if application wants to use HW mfx library (platform specific SDK implementation)
            bool     bLowLat;         // low latency mode
            bool     bCalLat;         // latency calculation
            mfxU32   nMaxFPS;         //rendering limited by certain fps
            uint32_t uBitstreamBufferSize; 

            mfxU16  nAsyncDepth;      // depth of asynchronous pipeline. default value is 4. must be between 1 and 20
            mfxU16  gpuCopy;          // GPU Copy mode (three-state option): MFX_GPUCOPY_DEFAULT or MFX_GPUCOPY_ON or MFX_GPUCOPY_OFF

            //MultiThreading only on Win32 or Win64
            mfxU16  nThreadsNum;      //number of mediasdk task threads
            mfxI32  SchedulingType;   //scheduling type of mediasdk task threads
            mfxI32  Priority;         //priority of mediasdk task threads

            mfxU16  Width;            //output width
            mfxU16  Height;           //output height

            mfxU32  fourcc;           //Output format parameters: MFX_FOURCC_NV12 or MFX_FOURCC_RGB4 or MFX_FOURCC_P010 or MFX_FOURCC_A2RGB10
            mfxU32  nFrames;

            sPluginParams pluginParams;

            stInputParams() : videoType(MFX_CODEC_AVC), memType(SYSTEM_MEMORY), bUseHWLib(true), bLowLat(true), bCalLat(true),
                nThreadsNum(0), SchedulingType(0), Priority(0), nMaxFPS(30), uBitstreamBufferSize(2*1024*1024),
                nAsyncDepth(4), gpuCopy(MFX_GPUCOPY_DEFAULT), nFrames(1), Width(0), Height(0), fourcc(MFX_FOURCC_RGB4)
            {}
            ~stInputParams(){}
        }tstInputParams;
    }
}