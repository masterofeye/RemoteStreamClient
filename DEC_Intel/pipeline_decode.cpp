/******************************************************************************\
Copyright (c) 2005-2016, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This sample was distributed or derived from the Intel's Media Samples package.
The original version of this sample may be obtained from https://software.intel.com/en-us/intel-media-server-studio
or https://software.intel.com/en-us/media-client-solutions-support.
\**********************************************************************************/

#include "mfx_samples_config.h"
#include "sample_defs.h"

#if defined(_WIN32) || defined(_WIN64)
#include <tchar.h>
#include <windows.h>
#endif

#include "sample_params.h"

#include "sysmem_allocator.h"
#include "DEC_inputs.h"
#include "pipeline_decode.h"

#if defined(_WIN32) || defined(_WIN64)
#include "d3d_allocator.h"
#include "d3d11_allocator.h"
#include "d3d_device.h"
#include "d3d11_device.h"
#endif

#if defined LIBVA_SUPPORT
#include "vaapi_allocator.h"
#include "vaapi_device.h"
#include "vaapi_utils.h"
#endif

#if defined(LIBVA_WAYLAND_SUPPORT)
#include "class_wayland.h"
#endif

#pragma warning(disable : 4100)

#define __SYNC_WA // avoid sync issue on Media SDK side

namespace RW{
    namespace DEC{
        CDecodingPipeline::CDecodingPipeline(std::shared_ptr<spdlog::logger> Logger) :
            m_Logger(Logger)
        {
            m_bVppIsUsed = false;
            MSDK_ZERO_MEMORY(m_mfxBS);

            m_pmfxDEC = NULL;
            m_pmfxVPP = NULL;
            m_impl = 0;

            MSDK_ZERO_MEMORY(m_mfxVideoParams);
            MSDK_ZERO_MEMORY(m_mfxVppVideoParams);

            m_pGeneralAllocator = NULL;
            m_pmfxAllocatorParams = NULL;
            m_bExternalAlloc = false;
            m_bDecOutSysmem = false;
            MSDK_ZERO_MEMORY(m_mfxResponse);
            MSDK_ZERO_MEMORY(m_mfxVppResponse);

            m_pCurrentFreeSurface = NULL;
            m_pCurrentFreeVppSurface = NULL;
            m_pCurrentFreeOutputSurface = NULL;
            m_pCurrentOutputSurface = NULL;

            m_bIsExtBuffers = false;
            m_bIsCompleteFrame = false;
            m_bPrintLatency = false;
            m_fourcc = 0;

            m_bFirstFrameInitialized = false;

            m_nTimeout = 0;
            m_nMaxFps = 0;

            m_diMode = 0;
            m_vppOutWidth = 0;
            m_vppOutHeight = 0;

            m_vLatency.reserve(1000); // reserve some space to reduce dynamic reallocation impact on pipeline execution

            MSDK_ZERO_MEMORY(m_VppDoNotUse);
            m_VppDoNotUse.Header.BufferId = MFX_EXTBUFF_VPP_DONOTUSE;
            m_VppDoNotUse.Header.BufferSz = sizeof(m_VppDoNotUse);

            m_VppDeinterlacing.Header.BufferId = MFX_EXTBUFF_VPP_DEINTERLACING;
            m_VppDeinterlacing.Header.BufferSz = sizeof(m_VppDeinterlacing);
        }

        CDecodingPipeline::~CDecodingPipeline()
        {
            Close();
        }

        mfxStatus CDecodingPipeline::Init(tstInputParams *pParams)
        {
            MSDK_CHECK_POINTER(pParams, MFX_ERR_NULL_PTR);

            mfxStatus sts = MFX_ERR_NONE;

            m_pInputParams = pParams;

                // prepare input stream file reader
                // for VP8 complete and single frame reader is a requirement
                // create reader that supports completeframe mode for latency oriented scenarios
                if (pParams->bLowLat || pParams->bCalLat)
                {
                    switch (pParams->videoType)
                    {
                    case MFX_CODEC_HEVC:
                    case MFX_CODEC_AVC:
                    case MFX_CODEC_JPEG:
                    case CODEC_VP8:
                        m_bIsCompleteFrame = true;
                        m_bPrintLatency = pParams->bCalLat;
                        break;
                    default:
                        return MFX_ERR_UNSUPPORTED; // latency mode is supported only for H.264 and JPEG codecs
                    }
                }
                if (pParams->fourcc)
                m_fourcc = pParams->fourcc;

            if (pParams->Width)
                m_vppOutWidth = pParams->Width;
            if (pParams->Height)
                m_vppOutHeight = pParams->Height;

            m_nMaxFps = pParams->nMaxFPS;
            m_nFrames = pParams->nFrames ? pParams->nFrames : MFX_INFINITE;

            mfxInitParam initPar;
            mfxExtThreadsParam threadsPar;
            mfxExtBuffer* extBufs[1];
            mfxVersion version;     // real API version with which library is initialized

            MSDK_ZERO_MEMORY(initPar);
            MSDK_ZERO_MEMORY(threadsPar);

            // we set version to 1.0 and later we will query actual version of the library which will got leaded
            initPar.Version.Major = 1;
            initPar.Version.Minor = 0;

            initPar.GPUCopy = pParams->gpuCopy;

            init_ext_buffer(threadsPar);

            bool needInitExtPar = false;

            if (pParams->eDeinterlace)
            {
                m_diMode = pParams->eDeinterlace;
            }

            if (needInitExtPar) {
                extBufs[0] = (mfxExtBuffer*)&threadsPar;
                initPar.ExtParam = extBufs;
                initPar.NumExtParam = 1;
            }

            initPar.Implementation = MFX_IMPL_SOFTWARE;
            sts = m_mfxSession.InitEx(initPar);

            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = m_mfxSession.QueryVersion(&version); // get real API version of the loaded library
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = m_mfxSession.QueryIMPL(&m_impl); // get actual library implementation
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            if ((pParams->videoType == MFX_CODEC_JPEG) && !CheckVersion(&version, MSDK_FEATURE_JPEG_DECODE)) {
                m_Logger->error("CDecodingPipeline::Init: Jpeg is not supported in the API version ")
                    << version.Major << "." << version.Minor;
                return MFX_ERR_UNSUPPORTED;
            }
            if (pParams->bLowLat && !CheckVersion(&version, MSDK_FEATURE_LOW_LATENCY)) {
                m_Logger->error("CDecodingPipeline::Init: Low Latency mode is not supported in the API version ")
                    << version.Major << "." << version.Minor;
                return MFX_ERR_UNSUPPORTED;
            }

            if (pParams->eDeinterlace &&
                (pParams->eDeinterlace != MFX_DEINTERLACING_ADVANCED) &&
                (pParams->eDeinterlace != MFX_DEINTERLACING_BOB))
            {
                m_Logger->error("CDecodingPipeline::Init: Unsupported deinterlace value: ") << pParams->eDeinterlace;
                return MFX_ERR_UNSUPPORTED;
            }

            // create decoder
            m_pmfxDEC = new MFXVideoDECODE(m_mfxSession);
            MSDK_CHECK_POINTER(m_pmfxDEC, MFX_ERR_MEMORY_ALLOC);

            // set video type in parameters
            m_mfxVideoParams.mfx.CodecId = pParams->videoType;

            if (CheckVersion(&version, MSDK_FEATURE_PLUGIN_API)) {
                /* Here we actually define the following codec initialization scheme:
                *  1. If plugin path or guid is specified: we load user-defined plugin (example: VP8 sample decoder plugin)
                *  2. If plugin path not specified:
                *    2.a) we check if codec is distributed as a mediasdk plugin and load it if yes
                *    2.b) if codec is not in the list of mediasdk plugins, we assume, that it is supported inside mediasdk library
                */
                // Load user plug-in, should go after CreateAllocator function (when all callbacks were initialized)
                if (pParams->pluginParams.type == MFX_PLUGINLOAD_TYPE_FILE && strlen(pParams->pluginParams.strPluginPath))
                {
                    m_pUserModule.reset(new MFXVideoUSER(m_mfxSession));
                    if (pParams->videoType == CODEC_VP8 || pParams->videoType == MFX_CODEC_HEVC)
                    {
                        m_pPlugin.reset(LoadPlugin(MFX_PLUGINTYPE_VIDEO_DECODE, m_mfxSession, pParams->pluginParams.pluginGuid, 1, pParams->pluginParams.strPluginPath, (mfxU32)strlen(pParams->pluginParams.strPluginPath)));
                    }
                    if (m_pPlugin.get() == NULL) sts = MFX_ERR_UNSUPPORTED;
                }
                else
                {
                    if (AreGuidsEqual(pParams->pluginParams.pluginGuid, MSDK_PLUGINGUID_NULL))
                    {
                        mfxIMPL impl = MFX_IMPL_SOFTWARE;
                        pParams->pluginParams.pluginGuid = msdkGetPluginUID(impl, MSDK_VDECODE, pParams->videoType);
                    }
                    if (!AreGuidsEqual(pParams->pluginParams.pluginGuid, MSDK_PLUGINGUID_NULL))
                    {
                        m_pPlugin.reset(LoadPlugin(MFX_PLUGINTYPE_VIDEO_DECODE, m_mfxSession, pParams->pluginParams.pluginGuid, 1));
                        if (m_pPlugin.get() == NULL) sts = MFX_ERR_UNSUPPORTED;
                    }
                    if (sts == MFX_ERR_UNSUPPORTED)
                    {
                        m_Logger->error("CDecodingPipeline::Init: Default plugin cannot be loaded (possibly you have to define plugin explicitly)");
                    }
                }
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
            }
            return sts;
        }

        mfxStatus CDecodingPipeline::InitForFirstFrame()
        {
            mfxStatus sts = MFX_ERR_NONE;

            // prepare bit stream
            if (MFX_CODEC_CAPTURE != m_pInputParams->videoType)
            {
                sts = InitMfxBitstream(&m_mfxBS, m_mfxBS.MaxLength);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
            }

            // Populate parameters. Involves DecodeHeader call
            sts = InitMfxParams(m_pInputParams);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            m_bDecOutSysmem = true;

            if (m_bVppIsUsed)
            {
                m_pmfxVPP = new MFXVideoVPP(m_mfxSession);
                if (!m_pmfxVPP) return MFX_ERR_MEMORY_ALLOC;
            }

            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // create device and allocator
            sts = CreateAllocator();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // in case of HW accelerated decode frames must be allocated prior to decoder initialization
            sts = AllocFrames();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = m_pmfxDEC->Init(&m_mfxVideoParams);
            if (MFX_WRN_PARTIAL_ACCELERATION == sts)
            {
                m_Logger->warn("CDecodingPipeline::InitForFirstFrame: partial acceleration");
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
            }
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            if (m_bVppIsUsed)
            {
                if (m_diMode)
                    m_mfxVppVideoParams.vpp.Out.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;

                sts = m_pmfxVPP->Init(&m_mfxVppVideoParams);
                if (MFX_WRN_PARTIAL_ACCELERATION == sts)
                {
                    m_Logger->warn("CDecodingPipeline::InitForFirstFrame: partial acceleration");
                    MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
                }
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
            }

            sts = m_pmfxDEC->GetVideoParam(&m_mfxVideoParams);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            return sts;
        }

        bool CDecodingPipeline::IsVppRequired(tstInputParams *pParams)
        {
            bool bVppIsUsed = false;
            // JPEG and Capture decoders can provide output in nv12 and rgb4 formats
            if ((pParams->videoType == MFX_CODEC_JPEG) ||
                ((pParams->videoType == MFX_CODEC_CAPTURE)))
            {
                bVppIsUsed = m_fourcc && (m_fourcc != MFX_FOURCC_NV12) && (m_fourcc != MFX_FOURCC_RGB4);
            }
            else
            {
                bVppIsUsed = m_fourcc && (m_fourcc != m_mfxVideoParams.mfx.FrameInfo.FourCC);
            }

            if ((m_mfxVideoParams.mfx.FrameInfo.CropW != pParams->Width) ||
                (m_mfxVideoParams.mfx.FrameInfo.CropH != pParams->Height))
            {
                bVppIsUsed |= pParams->Width && pParams->Height;
            }

            if (pParams->eDeinterlace)
            {
                bVppIsUsed = true;
            }
            return bVppIsUsed;
        }

        void CDecodingPipeline::Close()
        {
            WipeMfxBitstream(&m_mfxBS);
            MSDK_SAFE_DELETE(m_pmfxDEC);
            MSDK_SAFE_DELETE(m_pmfxVPP);

            DeleteFrames();

            if (m_bIsExtBuffers)
            {
                DeallocateExtMVCBuffers();
                DeleteExtBuffers();
            }

            m_pPlugin.reset();
            m_mfxSession.Close();
            MSDK_SAFE_DELETE(m_pInputParams);
            MSDK_SAFE_DELETE_ARRAY(m_VppDoNotUse.AlgList);

            // allocator if used as external for MediaSDK must be deleted after decoder
            DeleteAllocator();

            return;
        }

        mfxStatus CDecodingPipeline::InitMfxParams(tstInputParams *pParams)
        {
            MSDK_CHECK_POINTER(m_pmfxDEC, MFX_ERR_NULL_PTR);
            mfxStatus sts = MFX_ERR_NONE;
            mfxU32 numViews = 1;

            // try to find a sequence header in the stream
            // if header is not found this function exits with error (e.g. if device was lost and there's no header in the remaining stream)
            if (MFX_CODEC_CAPTURE == pParams->videoType)
            {
                m_mfxVideoParams.mfx.CodecId = MFX_CODEC_CAPTURE;
                m_mfxVideoParams.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
                m_mfxVideoParams.mfx.FrameInfo.Width = MSDK_ALIGN32(pParams->scrWidth);
                m_mfxVideoParams.mfx.FrameInfo.Height = MSDK_ALIGN32(pParams->scrHeight);
                m_mfxVideoParams.mfx.FrameInfo.CropW = pParams->scrWidth;
                m_mfxVideoParams.mfx.FrameInfo.CropH = pParams->scrHeight;
                m_mfxVideoParams.mfx.FrameInfo.FourCC = (m_fourcc == MFX_FOURCC_RGB4) ? MFX_FOURCC_RGB4 : MFX_FOURCC_NV12;

                if (!m_mfxVideoParams.mfx.FrameInfo.ChromaFormat)
                {
                    if (MFX_FOURCC_NV12 == m_mfxVideoParams.mfx.FrameInfo.FourCC)
                        m_mfxVideoParams.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
                    else if (MFX_FOURCC_RGB4 == m_mfxVideoParams.mfx.FrameInfo.FourCC)
                        m_mfxVideoParams.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV444;
                }
                m_bVppIsUsed = IsVppRequired(pParams);
            }

            for (; MFX_CODEC_CAPTURE != pParams->videoType;)
            {
                // trying to find PicStruct information in AVI headers
                if (m_mfxVideoParams.mfx.CodecId == MFX_CODEC_JPEG)
                    MJPEG_AVI_ParsePicStruct(&m_mfxBS);

                // parse bit stream and fill mfx params
                sts = m_pmfxDEC->DecodeHeader(&m_mfxBS, &m_mfxVideoParams);
                if (!sts)
                {
                    m_bVppIsUsed = IsVppRequired(pParams);
                }

                if (!sts &&
                    !(m_impl & MFX_IMPL_SOFTWARE) &&                        // hw lib
                    (m_mfxVideoParams.mfx.FrameInfo.BitDepthLuma == 10) &&  // hevc 10 bit
                    (m_mfxVideoParams.mfx.CodecId == MFX_CODEC_HEVC) &&
                    AreGuidsEqual(pParams->pluginParams.pluginGuid, MFX_PLUGINID_HEVCD_SW) && // sw hevc decoder
                    m_bVppIsUsed)
                {
                    sts = MFX_ERR_UNSUPPORTED;
                    m_Logger->error("CDecodingPipeline::InitMfxParams: Combination of (SW HEVC plugin in 10bit mode + HW lib VPP) isn't supported. Use -sw option.");
                }
                if (m_pPlugin.get() && pParams->videoType == CODEC_VP8 && !sts) {
                    // force set format to nv12 as the vp8 plugin uses yv12
                    m_mfxVideoParams.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
                }
                if (MFX_ERR_MORE_DATA == sts)
                {
                        m_mfxBS.DataFlag |= MFX_BITSTREAM_EOS;
                        sts = MFX_ERR_NONE;
                    continue;
                }
                else
                {
                    break;
                }
            }

            // check DecodeHeader status
            if (MFX_WRN_PARTIAL_ACCELERATION == sts)
            {
                m_Logger->warn("CDecodingPipeline::InitMfxParams: partial acceleration");
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
            }
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            if (!m_mfxVideoParams.mfx.FrameInfo.FrameRateExtN || !m_mfxVideoParams.mfx.FrameInfo.FrameRateExtD) {
                m_Logger->debug("CDecodingPipeline::InitMfxParams: pretending that stream is 30fps one");
                m_mfxVideoParams.mfx.FrameInfo.FrameRateExtN = 30;
                m_mfxVideoParams.mfx.FrameInfo.FrameRateExtD = 1;
            }
            if (!m_mfxVideoParams.mfx.FrameInfo.AspectRatioW || !m_mfxVideoParams.mfx.FrameInfo.AspectRatioH) {
                m_Logger->debug("CDecodingPipeline::InitMfxParams: pretending that aspect ratio is 1:1");
                m_mfxVideoParams.mfx.FrameInfo.AspectRatioW = 1;
                m_mfxVideoParams.mfx.FrameInfo.AspectRatioH = 1;
            }

            // Videoparams for RGB4 JPEG decoder output
            if ((pParams->fourcc == MFX_FOURCC_RGB4) && (pParams->videoType == MFX_CODEC_JPEG))
            {
                m_mfxVideoParams.mfx.FrameInfo.FourCC = MFX_FOURCC_RGB4;
                m_mfxVideoParams.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV444;
            }

            // specify memory type
            m_mfxVideoParams.IOPattern = (mfxU16)MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

            m_mfxVideoParams.AsyncDepth = pParams->nAsyncDepth;

            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::AllocAndInitVppFilters()
        {
            m_VppDoNotUse.NumAlg = 4;

            m_VppDoNotUse.AlgList = new mfxU32[m_VppDoNotUse.NumAlg];
            if (!m_VppDoNotUse.AlgList) return MFX_ERR_NULL_PTR;

            m_VppDoNotUse.AlgList[0] = MFX_EXTBUFF_VPP_DENOISE; // turn off denoising (on by default)
            m_VppDoNotUse.AlgList[1] = MFX_EXTBUFF_VPP_SCENE_ANALYSIS; // turn off scene analysis (on by default)
            m_VppDoNotUse.AlgList[2] = MFX_EXTBUFF_VPP_DETAIL; // turn off detail enhancement (on by default)
            m_VppDoNotUse.AlgList[3] = MFX_EXTBUFF_VPP_PROCAMP; // turn off processing amplified (on by default)

            if (m_diMode)
            {
                m_VppDeinterlacing.Mode = m_diMode;
            }

            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::InitVppParams()
        {
            m_mfxVppVideoParams.IOPattern = (mfxU16)(m_bDecOutSysmem ?
                MFX_IOPATTERN_IN_SYSTEM_MEMORY
                : MFX_IOPATTERN_IN_VIDEO_MEMORY);

            m_mfxVppVideoParams.IOPattern |= (mfxU16)MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

            MSDK_MEMCPY_VAR(m_mfxVppVideoParams.vpp.In, &m_mfxVideoParams.mfx.FrameInfo, sizeof(mfxFrameInfo));
            MSDK_MEMCPY_VAR(m_mfxVppVideoParams.vpp.Out, &m_mfxVppVideoParams.vpp.In, sizeof(mfxFrameInfo));

            if (m_fourcc)
            {
                m_mfxVppVideoParams.vpp.Out.FourCC = m_fourcc;
            }

            if (m_vppOutWidth && m_vppOutHeight)
            {

                m_mfxVppVideoParams.vpp.Out.CropW = m_vppOutWidth;
                m_mfxVppVideoParams.vpp.Out.Width = MSDK_ALIGN16(m_vppOutWidth);
                m_mfxVppVideoParams.vpp.Out.CropH = m_vppOutHeight;
                m_mfxVppVideoParams.vpp.Out.Height = (MFX_PICSTRUCT_PROGRESSIVE == m_mfxVppVideoParams.vpp.Out.PicStruct) ?
                    MSDK_ALIGN16(m_vppOutHeight) : MSDK_ALIGN32(m_vppOutHeight);
            }

            m_mfxVppVideoParams.AsyncDepth = m_mfxVideoParams.AsyncDepth;

            m_VppExtParams.clear();
            AllocAndInitVppFilters();
            m_VppExtParams.push_back((mfxExtBuffer*)&m_VppDoNotUse);
            if (m_diMode)
            {
                m_VppExtParams.push_back((mfxExtBuffer*)&m_VppDeinterlacing);
            }

            m_mfxVppVideoParams.ExtParam = &m_VppExtParams[0];
            m_mfxVppVideoParams.NumExtParam = (mfxU16)m_VppExtParams.size();
            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::AllocFrames()
        {
            MSDK_CHECK_POINTER(m_pmfxDEC, MFX_ERR_NULL_PTR);

            mfxStatus sts = MFX_ERR_NONE;

            mfxFrameAllocRequest Request;
            mfxFrameAllocRequest VppRequest[2];

            mfxU16 nSurfNum = 0; // number of surfaces for decoder
            mfxU16 nVppSurfNum = 0; // number of surfaces for vpp

            MSDK_ZERO_MEMORY(Request);
            MSDK_ZERO_MEMORY(VppRequest[0]);
            MSDK_ZERO_MEMORY(VppRequest[1]);

            sts = m_pmfxDEC->Query(&m_mfxVideoParams, &m_mfxVideoParams);
            MSDK_IGNORE_MFX_STS(sts, MFX_WRN_INCOMPATIBLE_VIDEO_PARAM);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // calculate number of surfaces required for decoder
            sts = m_pmfxDEC->QueryIOSurf(&m_mfxVideoParams, &Request);
            if (MFX_WRN_PARTIAL_ACCELERATION == sts)
            {
                m_Logger->warn("CDecodingPipeline::AllocFrames: partial acceleration");
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
                m_bDecOutSysmem = true;
            }
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            if (m_bVppIsUsed)
            {
                // respecify memory type between Decoder and VPP
                m_mfxVideoParams.IOPattern = (mfxU16)(m_bDecOutSysmem ?
                MFX_IOPATTERN_OUT_SYSTEM_MEMORY :
                                                MFX_IOPATTERN_OUT_VIDEO_MEMORY);

                // recalculate number of surfaces required for decoder
                sts = m_pmfxDEC->QueryIOSurf(&m_mfxVideoParams, &Request);
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);


                sts = InitVppParams();
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                sts = m_pmfxVPP->Query(&m_mfxVppVideoParams, &m_mfxVppVideoParams);
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_INCOMPATIBLE_VIDEO_PARAM);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                // VppRequest[0] for input frames request, VppRequest[1] for output frames request
                sts = m_pmfxVPP->QueryIOSurf(&m_mfxVppVideoParams, VppRequest);
                if (MFX_WRN_PARTIAL_ACCELERATION == sts) {
                    m_Logger->warn("CDecodingPipeline::AllocFrames: partial acceleration");
                    MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
                }
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                if ((VppRequest[0].NumFrameSuggested < m_mfxVppVideoParams.AsyncDepth) ||
                    (VppRequest[1].NumFrameSuggested < m_mfxVppVideoParams.AsyncDepth))
                    return MFX_ERR_MEMORY_ALLOC;


                // If surfaces are shared by 2 components, c1 and c2. NumSurf = c1_out + c2_in - AsyncDepth + 1
                // The number of surfaces shared by vpp input and decode output
                nSurfNum = Request.NumFrameSuggested + VppRequest[0].NumFrameSuggested - m_mfxVideoParams.AsyncDepth + 1;

                // The number of surfaces for vpp output
                nVppSurfNum = VppRequest[1].NumFrameSuggested;

                // prepare allocation request
                Request.NumFrameSuggested = Request.NumFrameMin = nSurfNum;

                // surfaces are shared between vpp input and decode output
                Request.Type = MFX_MEMTYPE_EXTERNAL_FRAME | MFX_MEMTYPE_FROM_DECODE | MFX_MEMTYPE_FROM_VPPIN;
            }

            if ((Request.NumFrameSuggested < m_mfxVideoParams.AsyncDepth) &&
                (m_impl & MFX_IMPL_HARDWARE_ANY))
                return MFX_ERR_MEMORY_ALLOC;

            Request.Type |= (m_bDecOutSysmem) ?
                MFX_MEMTYPE_SYSTEM_MEMORY
                : MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET;

#ifdef LIBVA_SUPPORT
            if (!m_bVppIsUsed &&
                (m_export_mode != vaapiAllocatorParams::DONOT_EXPORT))
            {
                Request.Type |= MFX_MEMTYPE_EXPORT_FRAME;
            }
#endif

            // alloc frames for decoder
            sts = m_pGeneralAllocator->Alloc(m_pGeneralAllocator->pthis, &Request, &m_mfxResponse);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            if (m_bVppIsUsed)
            {
                // alloc frames for VPP
#ifdef LIBVA_SUPPORT
                if (m_export_mode != vaapiAllocatorParams::DONOT_EXPORT)
                {
                    VppRequest[1].Type |= MFX_MEMTYPE_EXPORT_FRAME;
                }
#endif
                VppRequest[1].NumFrameSuggested = VppRequest[1].NumFrameMin = nVppSurfNum;
                MSDK_MEMCPY_VAR(VppRequest[1].Info, &(m_mfxVppVideoParams.vpp.Out), sizeof(mfxFrameInfo));

                sts = m_pGeneralAllocator->Alloc(m_pGeneralAllocator->pthis, &VppRequest[1], &m_mfxVppResponse);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                // prepare mfxFrameSurface1 array for decoder
                nVppSurfNum = m_mfxVppResponse.NumFrameActual;

                // AllocVppBuffers should call before AllocBuffers to set the value of m_OutputSurfacesNumber
                sts = AllocVppBuffers(nVppSurfNum);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
            }

            // prepare mfxFrameSurface1 array for decoder
            nSurfNum = m_mfxResponse.NumFrameActual;

            sts = AllocBuffers(nSurfNum);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            for (int i = 0; i < nSurfNum; i++)
            {
                // initating each frame:
                MSDK_MEMCPY_VAR(m_pSurfaces[i].frame.Info, &(Request.Info), sizeof(mfxFrameInfo));
                if (m_bExternalAlloc)
                {
                    m_pSurfaces[i].frame.Data.MemId = m_mfxResponse.mids[i];
                }
                else
                {
                    sts = m_pGeneralAllocator->Lock(m_pGeneralAllocator->pthis, m_mfxResponse.mids[i], &(m_pSurfaces[i].frame.Data));
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                }
            }

            // prepare mfxFrameSurface1 array for VPP
            for (int i = 0; i < nVppSurfNum; i++) {
                MSDK_MEMCPY_VAR(m_pVppSurfaces[i].frame.Info, &(VppRequest[1].Info), sizeof(mfxFrameInfo));
                if (m_bExternalAlloc) {
                    m_pVppSurfaces[i].frame.Data.MemId = m_mfxVppResponse.mids[i];
                }
                else {
                    sts = m_pGeneralAllocator->Lock(m_pGeneralAllocator->pthis, m_mfxVppResponse.mids[i], &(m_pVppSurfaces[i].frame.Data));
                    if (MFX_ERR_NONE != sts) {
                        return sts;
                    }
                }
            }
            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::CreateAllocator()
        {
            mfxStatus sts = MFX_ERR_NONE;

            m_pGeneralAllocator = new GeneralAllocator();

            // initialize memory allocator
            sts = m_pGeneralAllocator->Init(m_pmfxAllocatorParams);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            return MFX_ERR_NONE;
        }

        void CDecodingPipeline::DeleteFrames()
        {
            FreeBuffers();

            m_pCurrentFreeSurface = NULL;
            MSDK_SAFE_FREE(m_pCurrentFreeOutputSurface);

            m_pCurrentFreeVppSurface = NULL;

            // delete frames
            if (m_pGeneralAllocator)
            {
                m_pGeneralAllocator->Free(m_pGeneralAllocator->pthis, &m_mfxResponse);
            }

            return;
        }

        void CDecodingPipeline::DeleteAllocator()
        {
            // delete allocator
            MSDK_SAFE_DELETE(m_pGeneralAllocator);
            MSDK_SAFE_DELETE(m_pmfxAllocatorParams);
        }

        // function for allocating a specific external buffer
        template <typename Buffer>
        mfxStatus CDecodingPipeline::AllocateExtBuffer()
        {
            std::auto_ptr<Buffer> pExtBuffer(new Buffer());
            if (!pExtBuffer.get())
                return MFX_ERR_MEMORY_ALLOC;

            init_ext_buffer(*pExtBuffer);

            m_ExtBuffers.push_back(reinterpret_cast<mfxExtBuffer*>(pExtBuffer.release()));

            return MFX_ERR_NONE;
        }

        void CDecodingPipeline::AttachExtParam()
        {
            m_mfxVideoParams.ExtParam = reinterpret_cast<mfxExtBuffer**>(&m_ExtBuffers[0]);
            m_mfxVideoParams.NumExtParam = static_cast<mfxU16>(m_ExtBuffers.size());
        }

        void CDecodingPipeline::DeleteExtBuffers()
        {
            for (std::vector<mfxExtBuffer *>::iterator it = m_ExtBuffers.begin(); it != m_ExtBuffers.end(); ++it)
                delete *it;
            m_ExtBuffers.clear();
        }

        mfxStatus CDecodingPipeline::AllocateExtMVCBuffers()
        {
            mfxU32 i;

            mfxExtMVCSeqDesc* pExtMVCBuffer = (mfxExtMVCSeqDesc*)m_mfxVideoParams.ExtParam[0];
            MSDK_CHECK_POINTER(pExtMVCBuffer, MFX_ERR_MEMORY_ALLOC);

            pExtMVCBuffer->View = new mfxMVCViewDependency[pExtMVCBuffer->NumView];
            MSDK_CHECK_POINTER(pExtMVCBuffer->View, MFX_ERR_MEMORY_ALLOC);
            for (i = 0; i < pExtMVCBuffer->NumView; ++i)
            {
                MSDK_ZERO_MEMORY(pExtMVCBuffer->View[i]);
            }
            pExtMVCBuffer->NumViewAlloc = pExtMVCBuffer->NumView;

            pExtMVCBuffer->ViewId = new mfxU16[pExtMVCBuffer->NumViewId];
            MSDK_CHECK_POINTER(pExtMVCBuffer->ViewId, MFX_ERR_MEMORY_ALLOC);
            for (i = 0; i < pExtMVCBuffer->NumViewId; ++i)
            {
                MSDK_ZERO_MEMORY(pExtMVCBuffer->ViewId[i]);
            }
            pExtMVCBuffer->NumViewIdAlloc = pExtMVCBuffer->NumViewId;

            pExtMVCBuffer->OP = new mfxMVCOperationPoint[pExtMVCBuffer->NumOP];
            MSDK_CHECK_POINTER(pExtMVCBuffer->OP, MFX_ERR_MEMORY_ALLOC);
            for (i = 0; i < pExtMVCBuffer->NumOP; ++i)
            {
                MSDK_ZERO_MEMORY(pExtMVCBuffer->OP[i]);
            }
            pExtMVCBuffer->NumOPAlloc = pExtMVCBuffer->NumOP;

            return MFX_ERR_NONE;
        }

        void CDecodingPipeline::DeallocateExtMVCBuffers()
        {
            mfxExtMVCSeqDesc* pExtMVCBuffer = (mfxExtMVCSeqDesc*)m_mfxVideoParams.ExtParam[0];
            if (pExtMVCBuffer != NULL)
            {
                MSDK_SAFE_DELETE_ARRAY(pExtMVCBuffer->View);
                MSDK_SAFE_DELETE_ARRAY(pExtMVCBuffer->ViewId);
                MSDK_SAFE_DELETE_ARRAY(pExtMVCBuffer->OP);
            }

            MSDK_SAFE_DELETE(m_mfxVideoParams.ExtParam[0]);

            m_bIsExtBuffers = false;
        }

        mfxStatus CDecodingPipeline::ResetDecoder()
        {
            tstInputParams *pParams = m_pInputParams;
            mfxStatus sts = MFX_ERR_NONE;

            // close decoder
            sts = m_pmfxDEC->Close();
            MSDK_IGNORE_MFX_STS(sts, MFX_ERR_NOT_INITIALIZED);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // close VPP
            if (m_pmfxVPP)
            {
                sts = m_pmfxVPP->Close();
                MSDK_IGNORE_MFX_STS(sts, MFX_ERR_NOT_INITIALIZED);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
            }

            // free allocated frames
            DeleteFrames();

            // initialize parameters with values from parsed header
            sts = InitMfxParams(pParams);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // in case of HW accelerated decode frames must be allocated prior to decoder initialization
            sts = AllocFrames();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // init decoder
            sts = m_pmfxDEC->Init(&m_mfxVideoParams);
            if (MFX_WRN_PARTIAL_ACCELERATION == sts)
            {
                m_Logger->warn("CDecodingPipeline::ResetDecoder: partial acceleration");
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
            }
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            if (m_pmfxVPP)
            {
                sts = m_pmfxVPP->Init(&m_mfxVppVideoParams);
                if (MFX_WRN_PARTIAL_ACCELERATION == sts)
                {
                    m_Logger->warn("CDecodingPipeline::ResetDecoder: partial acceleration");
                    MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
                }
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
            }

            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::WriteNextFrameToBuffer(mfxFrameSurface1* frame)
        {
            mfxU32 h, w;

            switch (frame->Info.FourCC)
            {
            case MFX_FOURCC_YV12:
                m_pOutput->u32Size += frame->Info.CropW * frame->Info.CropH;

                for (mfxU32 i = 0; i < frame->Info.CropH; i++)
                {
                    memcpy(m_pOutput->pBuffer, frame->Data.Y + (frame->Info.CropY * frame->Data.Pitch + frame->Info.CropX) + i * frame->Data.Pitch, frame->Info.CropW);
                }

                w = frame->Info.CropW / 2;
                h = frame->Info.CropH / 2;

                m_pOutput->u32Size += (w * h) + (w * h);

                for (mfxU32 i = 0; i < h; i++)
                {
                    memcpy(m_pOutput->pBuffer, (frame->Data.U + (frame->Info.CropY * frame->Data.Pitch / 2 + frame->Info.CropX / 2) + i * frame->Data.Pitch / 2), w);
                }
                for (mfxU32 i = 0; i < h; i++)
                {
                    memcpy(m_pOutput->pBuffer, (frame->Data.V + (frame->Info.CropY * frame->Data.Pitch / 2 + frame->Info.CropX / 2) + i * frame->Data.Pitch / 2), w);
                }
                break;

            case MFX_FOURCC_NV12:

                m_pOutput->u32Size += frame->Info.CropW * frame->Info.CropH;

                for (mfxU32 i = 0; i < frame->Info.CropH; i++)
                {
                    memcpy(m_pOutput->pBuffer, frame->Data.Y + (frame->Info.CropY * frame->Data.Pitch + frame->Info.CropX) + i * frame->Data.Pitch, frame->Info.CropW);
                }

                h = frame->Info.CropH / 2;
                w = frame->Info.CropW;

                m_pOutput->u32Size += (w * h) + (w * h);

                for (mfxU32 i = 0; i < h; i++)
                {
                    for (mfxU32 j = 0; j < w; j += 2)
                    {
                        memcpy(m_pOutput->pBuffer, (frame->Data.UV + (frame->Info.CropY * frame->Data.Pitch / 2 + frame->Info.CropX) + i * frame->Data.Pitch + j), 1);
                    }
                }
                for (mfxU32 i = 0; i < h; i++)
                {
                    for (mfxU32 j = 0; j < w; j += 2)
                    {
                        memcpy(m_pOutput->pBuffer, (frame->Data.UV + (frame->Info.CropY * frame->Data.Pitch / 2 + frame->Info.CropX) + i * frame->Data.Pitch + j), 1);
                    }
                }

                break;

            case MFX_FOURCC_P010:

                m_pOutput->u32Size += 2 * frame->Info.CropW * frame->Info.CropH;

                for (mfxU32 i = 0; i < frame->Info.CropH; i++)
                {
                    memcpy(m_pOutput->pBuffer, (frame->Data.Y + (frame->Info.CropY * frame->Data.Pitch + frame->Info.CropX) + i * frame->Data.Pitch), 2 * frame->Info.CropW);
                }

                h = frame->Info.CropH / 2;
                w = frame->Info.CropW;

                m_pOutput->u32Size += 2 * w * h;

                for (mfxU32 i = 0; i < h; i++)
                {
                    memcpy(m_pOutput->pBuffer, (frame->Data.UV + (frame->Info.CropY * frame->Data.Pitch / 2 + frame->Info.CropX) + i * frame->Data.Pitch), 2 * w);
                }
                break;

            case MFX_FOURCC_RGB4:
            case 100: //DXGI_FORMAT_AYUV
            case MFX_FOURCC_A2RGB10:
                mfxU8* ptr;

                if (frame->Info.CropH > 0 && frame->Info.CropW > 0)
                {
                    w = frame->Info.CropW;
                    h = frame->Info.CropH;
                }
                else
                {
                    w = frame->Info.Width;
                    h = frame->Info.Height;
                }

                ptr = MSDK_MIN(MSDK_MIN(frame->Data.R, frame->Data.G), frame->Data.B);
                ptr = ptr + frame->Info.CropX + frame->Info.CropY * frame->Data.Pitch;

                m_pOutput->u32Size += 4 * w * h;

                for (mfxU32 i = 0; i < h; i++)
                {
                    memcpy(m_pOutput->pBuffer, (ptr + i * frame->Data.Pitch), 4 * w);
                }

                break;

            default:
                return MFX_ERR_UNSUPPORTED;
            }
            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::DeliverOutput(mfxFrameSurface1* frame)
        {
            CAutoTimer timer_fwrite(m_tick_fwrite);

            mfxStatus res = MFX_ERR_NONE, sts = MFX_ERR_NONE;

            if (!frame)
            {
                return MFX_ERR_NULL_PTR;
            }

            if (m_bExternalAlloc) {
                res = m_pGeneralAllocator->Lock(m_pGeneralAllocator->pthis, frame->Data.MemId, &(frame->Data));
                if (MFX_ERR_NONE == res) {
                    res = WriteNextFrameToBuffer(frame);
                    sts = m_pGeneralAllocator->Unlock(m_pGeneralAllocator->pthis, frame->Data.MemId, &(frame->Data));
                }
                if ((MFX_ERR_NONE == res) && (MFX_ERR_NONE != sts)) {
                    res = sts;
                }
            }
            else {
                res = WriteNextFrameToBuffer(frame);
            }

            return res;
        }

        void CDecodingPipeline::PrintPerFrameStat(bool force)
        {
#ifdef TRACE_PERFORMANCE


#define MY_COUNT 1 // TODO: this will be cmd option
#define MY_THRESHOLD 10000.0

            if (!(m_output_count % MY_COUNT) || force) {
                double fps, fps_fread, fps_fwrite;

                m_timer_overall.Sync();

                fps = (m_tick_overall) ? m_output_count / CTimer::ConvertToSeconds(m_tick_overall) : 0.0;
                fps_fread = (m_tick_fread) ? m_output_count / CTimer::ConvertToSeconds(m_tick_fread) : 0.0;
                fps_fwrite = (m_tick_fwrite) ? m_output_count / CTimer::ConvertToSeconds(m_tick_fwrite) : 0.0;
                // decoding progress
                m_Logger->trace("CDecodingPipeline::PrintPerFrameStat: ") \
                    << "\n Frame number: " << m_output_count \
                    << "\n fps:          " << fps \
                    << "\n fread_fps:    " << ((fps_fread < MY_THRESHOLD) ? fps_fread : 0.0) \
                    << "\n fwrite_fps:   " << ((fps_fwrite < MY_THRESHOLD) ? fps_fwrite : 0.0);
                fflush(NULL);
            }
#endif
        }

        mfxStatus CDecodingPipeline::SyncOutputSurface(mfxU32 wait)
        {
            if (!m_pCurrentOutputSurface) {
                m_pCurrentOutputSurface = m_OutputSurfacesPool.GetSurface();
            }
            if (!m_pCurrentOutputSurface) {
                return MFX_ERR_MORE_DATA;
            }

            mfxStatus sts = m_mfxSession.SyncOperation(m_pCurrentOutputSurface->syncp, wait);

            if (MFX_WRN_IN_EXECUTION == sts) {
                return sts;
            }
            if (MFX_ERR_NONE == sts) {
                // we got completely decoded frame - pushing it to the delivering thread...
                ++m_synced_count;
                if (m_bPrintLatency) {
                    m_vLatency.push_back(m_timer_overall.Sync() - m_pCurrentOutputSurface->surface->submit);
                }
                else {
                    PrintPerFrameStat();
                }
                m_output_count = m_synced_count;
                sts = DeliverOutput(&(m_pCurrentOutputSurface->surface->frame));
                if (MFX_ERR_NONE != sts) {
                    sts = MFX_ERR_UNKNOWN;
                }
                ReturnSurfaceToBuffers(m_pCurrentOutputSurface);
                m_pCurrentOutputSurface = NULL;
            }

            if (MFX_ERR_NONE != sts) {
                sts = MFX_ERR_UNKNOWN;
            }

            return sts;
        }

        mfxStatus CDecodingPipeline::RunDecoding(tstBitStream *pPayload)
        {
            mfxStatus           sts = MFX_ERR_NONE;
            if (!m_bFirstFrameInitialized)
            {
                sts = InitForFirstFrame();
                if (sts != MFX_ERR_NONE)
                {
                    m_Logger->error("RunDecoding: InitForFirstFrame failed!!!");
                    return sts;
                }
            }
            mfxFrameSurface1*   pOutSurface = NULL;
            mfxBitstream*       pBitstream = &m_mfxBS;
            bool                bErrIncompatibleVideoParams = false;
            CTimeInterval<>     decodeTimer(m_bIsCompleteFrame);
            time_t start_time = time(0);

            if (MFX_CODEC_CAPTURE == this->m_mfxVideoParams.mfx.CodecId)
            {
                pBitstream = 0;
            }

            mfxPayload dec_payload;
            mfxU64 ts;
            dec_payload.Data = (mfxU8*)pPayload->pBuffer;
            dec_payload.BufSize = pPayload->u32Size;
            dec_payload.Type = 5;

            int count = 0;

            while (((sts == MFX_ERR_NONE) || (MFX_ERR_MORE_DATA == sts) || (MFX_ERR_MORE_SURFACE == sts)) && (m_nFrames > m_output_count)){
                if (pBitstream && (MFX_ERR_MORE_DATA == sts)) {
                    // we almost reached end of stream, need to pull buffered data now
                    pBitstream = NULL;
                    m_mfxBS.DataFlag |= MFX_BITSTREAM_EOS;
                    sts = MFX_ERR_NONE;
                }
                if ((MFX_ERR_NONE == sts) || (MFX_ERR_MORE_DATA == sts) || (MFX_ERR_MORE_SURFACE == sts)) {
                    SyncFrameSurfaces();
                    SyncVppFrameSurfaces();
                    if (!m_pCurrentFreeSurface) {
                        m_pCurrentFreeSurface = m_FreeSurfacesPool.GetSurface();
                    }
                    if (!m_pCurrentFreeVppSurface) {
                        m_pCurrentFreeVppSurface = m_FreeVppSurfacesPool.GetSurface();
                    }
#ifndef __SYNC_WA
                    if (!m_pCurrentFreeSurface || !m_pCurrentFreeVppSurface) {
#else
                    if (!m_pCurrentFreeSurface || (!m_pCurrentFreeVppSurface && m_bVppIsUsed) || (m_OutputSurfacesPool.GetSurfaceCount() == m_mfxVideoParams.AsyncDepth)) {
#endif
                        // we stuck with no free surface available, now we will sync...
                        sts = SyncOutputSurface(MSDK_DEC_WAIT_INTERVAL);
                        if (MFX_ERR_MORE_DATA == sts) {
                            sts = MFX_ERR_NOT_FOUND;
                            m_Logger->critical("CDecodingPipeline::RunDecoding: failed to find output surface, that's a bug!");
                            break;
                        }
                        // note: MFX_WRN_IN_EXECUTION will also be treated as an error at this point
                        continue;
                    }

                    if (!m_pCurrentFreeOutputSurface) {
                        m_pCurrentFreeOutputSurface = GetFreeOutputSurface();
                    }
                    if (!m_pCurrentFreeOutputSurface) {
                        sts = MFX_ERR_NOT_FOUND;
                        break;
                    }
                }

                if ((MFX_ERR_NONE == sts) || (MFX_ERR_MORE_DATA == sts) || (MFX_ERR_MORE_SURFACE == sts)) {
                    if (m_bIsCompleteFrame) {
                        m_pCurrentFreeSurface->submit = m_timer_overall.Sync();
                    }
                    pOutSurface = NULL;
                    do {
                        sts = m_pmfxDEC->DecodeFrameAsync(pBitstream, &(m_pCurrentFreeSurface->frame), &pOutSurface, &(m_pCurrentFreeOutputSurface->syncp));
                        count++;
                        if (pBitstream && MFX_ERR_MORE_DATA == sts && pBitstream->MaxLength == pBitstream->DataLength)
                        {
                            mfxStatus status = ExtendMfxBitstream(pBitstream, pBitstream->MaxLength * 2);
                            MSDK_CHECK_RESULT(status, MFX_ERR_NONE, status);
                        }

                        if (MFX_WRN_DEVICE_BUSY == sts) {
                            if (m_bIsCompleteFrame) {
                                //in low latency mode device busy leads to increasing of latency
                                // m_Logger->warn("CDecodingPipeline::RunDecoding:  latency increased due to MFX_WRN_DEVICE_BUSY");
                            }
                            mfxStatus _sts = SyncOutputSurface(MSDK_DEC_WAIT_INTERVAL);
                            // note: everything except MFX_ERR_NONE are errors at this point
                            if (MFX_ERR_NONE == _sts) {
                                sts = MFX_WRN_DEVICE_BUSY;
                            }
                            else {
                                sts = _sts;
                                if (MFX_ERR_MORE_DATA == sts) {
                                    // we can't receive MFX_ERR_MORE_DATA and have no output - that's a bug
                                    sts = MFX_WRN_DEVICE_BUSY;//MFX_ERR_NOT_FOUND;
                                }
                            }
                        }
                    } while (MFX_WRN_DEVICE_BUSY == sts);

                    if (sts > MFX_ERR_NONE) {
                        // ignoring warnings...
                        if (m_pCurrentFreeOutputSurface->syncp) {
                            MSDK_SELF_CHECK(pOutSurface);
                            // output is available
                            sts = MFX_ERR_NONE;
                        }
                        else {
                            // output is not available
                            sts = MFX_ERR_MORE_SURFACE;
                        }
                    }
                    else if ((MFX_ERR_MORE_DATA == sts) && pBitstream) {
                        if (m_bIsCompleteFrame && pBitstream->DataLength)
                        {
                            // In low_latency mode decoder have to process bitstream completely
                            m_Logger->error("CDecodingPipeline::RunDecoding:  Incorrect decoder behavior in low latency mode (bitstream length is not equal to 0 after decoding)");
                            sts = MFX_ERR_UNDEFINED_BEHAVIOR;
                            continue;
                        }
                    }
                    else if ((MFX_ERR_MORE_DATA == sts) && !pBitstream) {
                        // that's it - we reached end of stream; now we need to render bufferred data...
                        do {
                            sts = SyncOutputSurface(MSDK_DEC_WAIT_INTERVAL);
                        } while (MFX_ERR_NONE == sts);

                        if (MFX_ERR_MORE_DATA == sts) {
                            sts = MFX_ERR_NONE;
                        }
                        break;
                    }
                    else if (MFX_ERR_INCOMPATIBLE_VIDEO_PARAM == sts) {
                        bErrIncompatibleVideoParams = true;
                        // need to go to the buffering loop prior to reset procedure
                        pBitstream = NULL;
                        sts = MFX_ERR_NONE;
                        continue;
                    }
                }

                if ((MFX_ERR_NONE == sts) || (MFX_ERR_MORE_DATA == sts) || (MFX_ERR_MORE_SURFACE == sts)) {
                    // if current free surface is locked we are moving it to the used surfaces array
                    /*if (m_pCurrentFreeSurface->frame.Data.Locked)*/ {
                        m_UsedSurfacesPool.AddSurface(m_pCurrentFreeSurface);
                        m_pCurrentFreeSurface = NULL;
                    }
                }
                if (MFX_ERR_NONE == sts) {
                    if (m_bVppIsUsed)
                    {
                        do {
                            if ((m_pCurrentFreeVppSurface->frame.Info.CropW == 0) ||
                                (m_pCurrentFreeVppSurface->frame.Info.CropH == 0)) {
                                m_pCurrentFreeVppSurface->frame.Info.CropW = pOutSurface->Info.CropW;
                                m_pCurrentFreeVppSurface->frame.Info.CropH = pOutSurface->Info.CropH;
                                m_pCurrentFreeVppSurface->frame.Info.CropX = pOutSurface->Info.CropX;
                                m_pCurrentFreeVppSurface->frame.Info.CropY = pOutSurface->Info.CropY;
                            }
                            if (pOutSurface->Info.PicStruct != m_pCurrentFreeVppSurface->frame.Info.PicStruct) {
                                m_pCurrentFreeVppSurface->frame.Info.PicStruct = pOutSurface->Info.PicStruct;
                            }
                            if ((pOutSurface->Info.PicStruct == 0) && (m_pCurrentFreeVppSurface->frame.Info.PicStruct == 0)) {
                                m_pCurrentFreeVppSurface->frame.Info.PicStruct = pOutSurface->Info.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
                            }

                            if (m_diMode && m_pCurrentFreeVppSurface)
                                m_pCurrentFreeVppSurface->frame.Info.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;

                            sts = m_pmfxVPP->RunFrameVPPAsync(pOutSurface, &(m_pCurrentFreeVppSurface->frame), NULL, &(m_pCurrentFreeOutputSurface->syncp));

                            if (MFX_WRN_DEVICE_BUSY == sts) {
                                MSDK_SLEEP(1); // just wait and then repeat the same call to RunFrameVPPAsync
                            }
                        } while (MFX_WRN_DEVICE_BUSY == sts);

                        // process errors
                        if (MFX_ERR_MORE_DATA == sts) { // will never happen actually
                            continue;
                        }
                        else if (MFX_ERR_NONE != sts) {
                            break;
                        }

                        m_UsedVppSurfacesPool.AddSurface(m_pCurrentFreeVppSurface);
                        msdk_atomic_inc16(&(m_pCurrentFreeVppSurface->render_lock));

                        m_pCurrentFreeOutputSurface->surface = m_pCurrentFreeVppSurface;
                        m_OutputSurfacesPool.AddSurface(m_pCurrentFreeOutputSurface);

                        m_pCurrentFreeOutputSurface = NULL;
                        m_pCurrentFreeVppSurface = NULL;
                    }
                    else
                    {
                        msdkFrameSurface* surface = FindUsedSurface(pOutSurface);

                        msdk_atomic_inc16(&(surface->render_lock));

                        m_pCurrentFreeOutputSurface->surface = surface;
                        m_OutputSurfacesPool.AddSurface(m_pCurrentFreeOutputSurface);
                        m_pCurrentFreeOutputSurface = NULL;
                    }
                    if (pPayload)
                    {
                        mfxPayload dec_payload;
                        mfxU64 ts;
                        dec_payload.Data = (mfxU8*)pPayload->pBuffer;
                        dec_payload.BufSize = pPayload->u32Size;
                        dec_payload.Type = 5;
                        dec_payload.NumBit = dec_payload.BufSize * 8;

                        while (dec_payload.NumBit != 0)
                        {
                            mfxStatus stats = m_pmfxDEC->GetPayload(&ts, &dec_payload);
                            if ((stats == MFX_ERR_NONE) && (dec_payload.Type == 5))
                            {
                                memcpy(pPayload->pBuffer, dec_payload.Data, dec_payload.BufSize);
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                }
            } //while processing

            PrintPerFrameStat(true);

            if (m_bPrintLatency && m_vLatency.size() > 0) {
                unsigned int frame_idx = 0;
                msdk_tick sum = 0;
                for (std::vector<msdk_tick>::iterator it = m_vLatency.begin(); it != m_vLatency.end(); ++it)
                {
                    sum += *it;
                    m_Logger->debug("CDecodingPipeline::RunDecoding: ")
                        << "Frame = " << (++frame_idx)
                        << "latency = " << (CTimer::ConvertToSeconds(*it) * 1000);
                }
                m_Logger->debug("CDecodingPipeline::RunDecoding: Latency summary (ms): ")
                    << " AVG: " << (CTimer::ConvertToSeconds((msdk_tick)((mfxF64)sum / m_vLatency.size())) * 1000)
                    << " MAX: " << (CTimer::ConvertToSeconds(*std::max_element(m_vLatency.begin(), m_vLatency.end())) * 1000)
                    << " MIN: " << (CTimer::ConvertToSeconds(*std::min_element(m_vLatency.begin(), m_vLatency.end())) * 1000);
            }

            // exit in case of other errors
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // if we exited main decoding loop with ERR_INCOMPATIBLE_PARAM we need to send this status to caller
            if (bErrIncompatibleVideoParams) {
                sts = MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
            }

            return sts; // ERR_NONE or ERR_INCOMPATIBLE_VIDEO_PARAM
        }

        void CDecodingPipeline::PrintInfo()
        {
            mfxFrameInfo Info = m_mfxVideoParams.mfx.FrameInfo;
            mfxF64 dFrameRate = CalculateFrameRate(Info.FrameRateExtN, Info.FrameRateExtD);
            const char* sMemType = "system";

            mfxIMPL impl;
            m_mfxSession.QueryIMPL(&impl);

            mfxVersion ver;
            m_mfxSession.QueryVersion(&ver);

            const char* sImpl =  "sw";

            m_Logger->debug("CDecodingPipeline::PrintInfo ") \
                << "\n Input video CodecID  = " << m_mfxVideoParams.mfx.CodecId \
                << "\n Output format        = " << m_mfxVideoParams.mfx.FrameInfo.FourCC \
                << "\n Resolution           = " << Info.Width << " x " << Info.Height \
                << "\n Crop X,Y,W,H         = " << Info.CropX << ", " << Info.CropY << ", " << Info.CropW << ", " << Info.CropH \
                << "\n Frame rate           = " << dFrameRate \
                << "\n Memory type          = " << sMemType \
                << "\n MediaSDK impl        = " << sImpl \
                << "\n MediaSDK version     = " << ver.Major << "." << ver.Minor;

            return;
        }
    }
}