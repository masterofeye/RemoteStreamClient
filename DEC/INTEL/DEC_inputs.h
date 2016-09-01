#pragma once

#include "sample_params.h"

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

namespace RW{
    namespace DEC{
        namespace INTEL{

            enum MemType {
                SYSTEM_MEMORY = 0x00,
                D3D9_MEMORY = 0x01,
                D3D11_MEMORY = 0x02,
            };

            typedef struct stInputParams
            {
                mfxU32   videoType;       // Default: MFX_CODEC_AVC. (MFX_CODEC_AVC for h264, MFX_CODEC_HEVC for h265)
                bool     bLowLat;         // Default: true. Low latency mode
                bool     bCalLat;         // Default: true. Latency calculation

                mfxU16  nAsyncDepth;      // Default: 4. Depth of asynchronous pipeline. Value must be between 1 and 20
                mfxU16  gpuCopy;          // Default: MFX_GPUCOPY_DEFAULT. GPU Copy mode (MFX_GPUCOPY_DEFAULT or MFX_GPUCOPY_ON or MFX_GPUCOPY_OFF)

                mfxU16  Width;            // NEEDS TO BE SET: output width
                mfxU16  Height;           // NEEDS TO BE SET: output height

                MemType memType;          // Default: SYSTEM_MEMORY. (SYSTEM_MEMORY, D3D9_MEMORY or D3D11_MEMORY)

                bool    bUseHWLib;        // Default: true. Use hardware acceleration lib

                mfxU32  fourcc;           // Default: MFX_FOURCC_RGB4. Output format parameters (MFX_FOURCC_NV12 or MFX_FOURCC_RGB4 or MFX_FOURCC_P010 or MFX_FOURCC_A2RGB10)
                sPluginParams pluginParams;

                stInputParams() : videoType(MFX_CODEC_AVC), bLowLat(true), bCalLat(true), bUseHWLib(true), memType(D3D9_MEMORY),
                    nAsyncDepth(4), gpuCopy(MFX_GPUCOPY_OFF), Width(0), Height(0), fourcc(MFX_FOURCC_RGB4)
                {}
                ~stInputParams(){}
            }tstInputParams;
        }
    }
}