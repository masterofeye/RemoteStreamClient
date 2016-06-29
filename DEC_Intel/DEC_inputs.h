#pragma once

#include "sample_params.h"

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

namespace RW{
    namespace DEC{

        typedef struct stInputParams
        {
            mfxU32   videoType;       // MFX_CODEC_AVC for h264, MFX_CODEC_HEVC for h265
            bool     bLowLat;         // low latency mode
            bool     bCalLat;         // latency calculation
            mfxU32   nMaxFPS;         //rendering limited by certain fps

            mfxU16  nAsyncDepth;      // depth of asynchronous pipeline. default value is 4. must be between 1 and 20
            mfxU16  gpuCopy;          // GPU Copy mode (three-state option): MFX_GPUCOPY_DEFAULT or MFX_GPUCOPY_ON or MFX_GPUCOPY_OFF

            mfxU16  Width;            //output width
            mfxU16  Height;           //output height

            mfxU32  fourcc;           //Output format parameters: MFX_FOURCC_NV12 or MFX_FOURCC_RGB4 or MFX_FOURCC_P010 or MFX_FOURCC_A2RGB10
            mfxU32  nFrames;          //
            mfxU16  eDeinterlace;     //Deinterlace modus

            mfxU16  scrWidth;         //DON'T USE: Screen Width
            mfxU16  scrHeight;        //DON'T USE: Screen Height

            sPluginParams pluginParams;

            stInputParams() : videoType(MFX_CODEC_AVC), bLowLat(true), bCalLat(true),
                eDeinterlace(0), 
                nMaxFPS(30), 
                nAsyncDepth(4), gpuCopy(MFX_GPUCOPY_DEFAULT), nFrames(1), Width(0), Height(0), fourcc(MFX_FOURCC_RGB4)
            {}
            ~stInputParams(){}
        }tstInputParams;
    }
}