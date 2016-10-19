#pragma once

#include "mfxdefs.h"
#include "sample_params.h"
#include "vm\strings_defs.h"

namespace RW{
    namespace ENC{
        namespace INTEL{

            enum {
                MVC_DISABLED = 0x0,
                MVC_ENABLED = 0x1,
                MVC_VIEWOUTPUT = 0x2,    // 2 output bitstreams
            };

            enum MemType {
                SYSTEM_MEMORY = 0x00,
                D3D9_MEMORY = 0x01,
                D3D11_MEMORY = 0x02,
            };

            struct sInputParams
            {
                mfxU16 nTargetUsage;  //supported only for H.264, MPEG2 and MVC encoders
                mfxU32 CodecId;
                mfxU32 ColorFormat;
                mfxU16 nPicStruct;
                mfxU16 nWidth; // source picture width
                mfxU16 nHeight; // source picture height
                mfxF64 dFrameRate;
                mfxU16 nBitRate;  //supported only for H.264, MPEG2 and MVC encoders
                mfxU16 nGopPicSize;
                mfxU16 nGopRefDist;
                mfxU16 nNumRefFrame;
                mfxU16 nBRefType;
                mfxU16 nIdrInterval;
                mfxU16 reserved[4];

                mfxU16 nDstWidth; // destination picture width, specified if resizing required
                mfxU16 nDstHeight; // destination picture height, specified if resizing required

                MemType memType;
                bool bUseHWLib; // true if application wants to use HW MSDK library

                //msdk_char strSrcFile[MSDK_MAX_FILENAME_LEN];

                sPluginParams pluginParams;

                //std::vector<msdk_char*> srcFileBuff;
                //std::vector<msdk_char*> dstFileBuff;

                mfxU32  HEVCPluginVersion;
                msdk_char strPluginDLLPath[MSDK_MAX_FILENAME_LEN]; // plugin dll path and name

                mfxU16 nAsyncDepth; // depth of asynchronous pipeline, this number can be tuned to achieve better performance
                mfxU16 gpuCopy; // GPU Copy mode (three-state option)

                mfxU16 nRateControlMethod;  // Look ahead BRC is supported only with -hw option
                mfxU16 nLADepth; // depth of the look ahead bitrate control  algorithm
                mfxU16 nMaxSliceSize; //maximum size of slice. MaxSliceSize option is supported only with -hw option
                mfxU16 nQPI;
                mfxU16 nQPP;
                mfxU16 nQPB;

                sInputParams() : nLADepth(100), nMaxSliceSize(0), bUseHWLib(false), CodecId(MFX_CODEC_AVC), ColorFormat(MFX_FOURCC_YV12), 
                    nPicStruct(MFX_PICSTRUCT_PROGRESSIVE), memType(SYSTEM_MEMORY), nDstHeight(0), nDstWidth(0), nBitRate(0), dFrameRate(0),
                    nTargetUsage(MFX_TARGETUSAGE_BALANCED), nAsyncDepth(0), nRateControlMethod(0), gpuCopy(0), 
                    nGopPicSize(0), nGopRefDist(0), nNumRefFrame(0), nBRefType(0), nIdrInterval(0), 
                    nQPI(0), nQPP(0), nQPB(0)
                {}
            };
        }
    }
}
