/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2005-2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#include "mfx_samples_config.h"
#include "sample_defs.h"

#if defined(_WIN32) || defined(_WIN64)
#include <tchar.h>
#include <windows.h>
#endif

#include "hw_device.h"
#include "sample_params.h"

#include "sysmem_allocator.h"
#include "mfxplugin++.h"
#include "plugin_loader.h"
#include "DEC_inputs.h"
#include "pipeline_decode.h"

#if defined(_WIN32) || defined(_WIN64)
#include "d3d_allocator.h"
#include "d3d11_allocator.h"
#include "d3d_device.h"
#include "d3d11_device.h"
#endif

#pragma warning(disable : 4100)

#define __SYNC_WA // avoid sync issue on Media SDK side

namespace RW{
    namespace DEC{

        CDecodingPipeline::CDecodingPipeline(std::shared_ptr<spdlog::logger> Logger) :
            m_Logger(Logger)
        {

            MSDK_ZERO_MEMORY(m_mfxBS);

            m_pmfxDEC = NULL;
            m_pmfxVPP = NULL;

            MSDK_ZERO_MEMORY(m_mfxVideoParams);
            MSDK_ZERO_MEMORY(m_mfxVppVideoParams);

            m_pMFXAllocator = NULL;
            m_pmfxAllocatorParams = NULL;
            m_memType = SYSTEM_MEMORY;
            m_bExternalAlloc = false;
            m_bSysmemBetween = false;
            MSDK_ZERO_MEMORY(m_mfxResponse);
            MSDK_ZERO_MEMORY(m_mfxVppResponse);

            m_pCurrentFreeSurface = NULL;
            m_pCurrentFreeVppSurface = NULL;
            m_pCurrentFreeOutputSurface = NULL;
            m_pCurrentOutputSurface = NULL;

            m_pDeliverOutputSemaphore = NULL;
            m_pDeliveredEvent = NULL;
            m_error = MFX_ERR_NONE;
            m_bStopDeliverLoop = false;

            m_bIsCompleteFrame = false;
            m_fourcc = MFX_FOURCC_RGB4;
            m_bPrintLatency = false;

            m_bFirstFrameInitialized = false;

            m_nTimeout = 0;
            m_nMaxFps = 0;

            m_vLatency.reserve(1000); // reserve some space to reduce dynamic reallocation impact on pipeline execution

            MSDK_ZERO_MEMORY(m_VppDoNotUse);
            m_VppDoNotUse.Header.BufferId = MFX_EXTBUFF_VPP_DONOTUSE;
            m_VppDoNotUse.Header.BufferSz = sizeof(m_VppDoNotUse);

            m_hwdev = NULL;
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
            if (m_pInputParams->bLowLat || m_pInputParams->bCalLat)
            {
                switch (m_pInputParams->videoType)
                {
                case MFX_CODEC_HEVC:
                case MFX_CODEC_AVC:
                    m_FileReader.reset(new CH264FrameReader());

                    m_bIsCompleteFrame = true;
                    m_bPrintLatency = m_pInputParams->bCalLat;
                    break;
                default:
                    return MFX_ERR_UNSUPPORTED; // latency mode is supported only for H.264 and JPEG codecs
                }
            }

            m_fourcc = m_pInputParams->fourcc;
            m_memType = m_pInputParams->memType;
            m_bSysmemBetween = (m_pInputParams->bUseHWLib) ? false : true;
            m_nMaxFps = m_pInputParams->nMaxFPS;
            m_nFrames = m_pInputParams->nFrames ? m_pInputParams->nFrames : MFX_INFINITE;

            if (MFX_CODEC_CAPTURE != m_pInputParams->videoType)
            {
                m_FileReader->Close();
                //msdk_char dummy[1024];
                //sts = m_FileReader->Init(dummy);
                //MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
            }

            mfxInitParam initPar;
            mfxExtThreadsParam threadsPar;
            mfxExtBuffer* extBufs[1];
            mfxVersion version;     // real API version with which library is initialized

            MSDK_ZERO_MEMORY(initPar);
            MSDK_ZERO_MEMORY(threadsPar);

            // we set version to 1.0 and later we will query actual version of the library which will got leaded
            initPar.Version.Major = 1;
            initPar.Version.Minor = 0;

            init_ext_buffer(threadsPar);

            bool needInitExtPar = false;

            if (m_pInputParams->nThreadsNum) {
                threadsPar.NumThread = m_pInputParams->nThreadsNum;
                needInitExtPar = true;
            }
            if (m_pInputParams->SchedulingType) {
                threadsPar.SchedulingType = m_pInputParams->SchedulingType;
                needInitExtPar = true;
            }
            if (m_pInputParams->Priority) {
                threadsPar.Priority = m_pInputParams->Priority;
                needInitExtPar = true;
            }
            if (needInitExtPar) {
                extBufs[0] = (mfxExtBuffer*)&threadsPar;
                initPar.ExtParam = extBufs;
                initPar.NumExtParam = 1;
            }

            // Init session
            if (m_pInputParams->bUseHWLib) {
                // try searching on all display adapters
                initPar.Implementation = MFX_IMPL_HARDWARE_ANY;

                // if d3d11 surfaces are used ask the library to run acceleration through D3D11
                // feature may be unsupported due to OS or MSDK API version
                if (D3D11_MEMORY == m_pInputParams->memType)
                    initPar.Implementation |= MFX_IMPL_VIA_D3D11;

                sts = m_mfxSession.InitEx(initPar);

                // MSDK API version may not support multiple adapters - then try initialize on the default
                if (MFX_ERR_NONE != sts) {
                    initPar.Implementation = (initPar.Implementation & !MFX_IMPL_HARDWARE_ANY) | MFX_IMPL_HARDWARE;
                    sts = m_mfxSession.InitEx(initPar);
                }
            }
            else {
                initPar.Implementation = MFX_IMPL_SOFTWARE;
                sts = m_mfxSession.InitEx(initPar);
            }

            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = MFXQueryVersion(m_mfxSession, &version); // get real API version of the loaded library
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            if (m_pInputParams->bLowLat && !CheckVersion(&version, MSDK_FEATURE_LOW_LATENCY)) {
                m_Logger->error("CDecodingPipeline::Init: Low Latency mode is not supported API version ") << version.Major << "." << version.Minor;
                return MFX_ERR_UNSUPPORTED;
            }

            // create decoder
            m_pmfxDEC = new MFXVideoDECODE(m_mfxSession);
            MSDK_CHECK_POINTER(m_pmfxDEC, MFX_ERR_MEMORY_ALLOC);

            //create VPP
            m_pmfxVPP = new MFXVideoVPP(m_mfxSession);
            if (!m_pmfxVPP) return MFX_ERR_MEMORY_ALLOC;
            // set video type in parameters
            m_mfxVideoParams.mfx.CodecId = m_pInputParams->videoType;

            // prepare bit stream
            if (MFX_CODEC_CAPTURE != m_pInputParams->videoType)
            {
                sts = InitMfxBitstream(&m_mfxBS, pParams->uBitstreamBufferSize);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
            }

            if (CheckVersion(&version, MSDK_FEATURE_PLUGIN_API)) {
                /* Here we actually define the following codec initialization scheme:
                *  1. If plugin path or guid is specified: we load user-defined plugin (example: VP8 sample decoder plugin)
                *  2. If plugin path not specified:
                *    2.a) we check if codec is distributed as a mediasdk plugin and load it if yes
                *    2.b) if codec is not in the list of mediasdk plugins, we assume, that it is supported inside mediasdk library
                */
                // Load user plug-in, should go after CreateAllocator function (when all callbacks were initialized)
                if (m_pInputParams->pluginParams.type == MFX_PLUGINLOAD_TYPE_FILE && strlen(m_pInputParams->pluginParams.strPluginPath))
                {
                    m_pUserModule.reset(new MFXVideoUSER(m_mfxSession));
                    if (m_pInputParams->videoType == MFX_CODEC_HEVC)
                    {
                        m_pPlugin.reset(LoadPlugin(MFX_PLUGINTYPE_VIDEO_DECODE, m_mfxSession, m_pInputParams->pluginParams.pluginGuid, 1, m_pInputParams->pluginParams.strPluginPath, (mfxU32)strlen(m_pInputParams->pluginParams.strPluginPath)));
                    }
                    if (m_pPlugin.get() == NULL) sts = MFX_ERR_UNSUPPORTED;
                }
                else
                {
                    if (AreGuidsEqual(m_pInputParams->pluginParams.pluginGuid, MSDK_PLUGINGUID_NULL))
                    {
                        mfxIMPL impl = m_pInputParams->bUseHWLib ? MFX_IMPL_HARDWARE : MFX_IMPL_SOFTWARE;
                        m_pInputParams->pluginParams.pluginGuid = msdkGetPluginUID(impl, MSDK_VDECODE, m_pInputParams->videoType);
                    }
                    if (!AreGuidsEqual(m_pInputParams->pluginParams.pluginGuid, MSDK_PLUGINGUID_NULL))
                    {
                        m_pPlugin.reset(LoadPlugin(MFX_PLUGINTYPE_VIDEO_DECODE, m_mfxSession, m_pInputParams->pluginParams.pluginGuid, 1));
                        if (m_pPlugin.get() == NULL) sts = MFX_ERR_UNSUPPORTED;
                    }
                }
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
            }

            return sts;
        }

        mfxStatus CDecodingPipeline::InitForFirstFrame()
        {
            // Populate parameters. Involves DecodeHeader call
            mfxStatus sts = InitMfxParams();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = CreateAllocator();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // in case of HW accelerated decode frames must be allocated prior to decoder initialization
            sts = AllocFrames();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = m_pmfxDEC->Init(&m_mfxVideoParams);
            if (MFX_WRN_PARTIAL_ACCELERATION == sts)
            {
                m_Logger->alert("CDecodingPipeline::InitForFirstFrame: WARNING! Partial acceleration.");
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
            }
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = m_pmfxDEC->GetVideoParam(&m_mfxVideoParams);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = m_pmfxVPP->Init(&m_mfxVppVideoParams);
            if (MFX_WRN_PARTIAL_ACCELERATION == sts) {
                m_Logger->alert("CDecodingPipeline::InitForFirstFrame: WARNING! partial acceleration");
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
            }
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = m_pmfxVPP->GetVideoParam(&m_mfxVppVideoParams);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            if (sts == MFX_ERR_NONE)
            {
                m_bFirstFrameInitialized = true;
            }

            return sts;
        }



        void CDecodingPipeline::Close()
        {
            WipeMfxBitstream(&m_mfxBS);
            MSDK_SAFE_DELETE(m_pmfxDEC);
            MSDK_SAFE_DELETE(m_pmfxVPP);

            DeleteFrames();

            m_pPlugin.reset();
            m_mfxSession.Close();
            if (m_FileReader.get())
                m_FileReader->Close();

            // allocator if used as external for MediaSDK must be deleted after decoder
            DeleteAllocator();

            return;
        }

        mfxStatus CDecodingPipeline::InitMfxParams()
        {
            MSDK_CHECK_POINTER(m_pmfxDEC, MFX_ERR_NULL_PTR);
            mfxStatus sts = MFX_ERR_NONE;

            // try to find a sequence header in the stream
            // if header is not found this function exits with error (e.g. if device was lost and there's no header in the remaining stream)
            if (MFX_CODEC_CAPTURE == m_pInputParams->videoType)
            {
                m_mfxVideoParams.mfx.CodecId = MFX_CODEC_CAPTURE;
                m_mfxVideoParams.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
                m_mfxVideoParams.mfx.FrameInfo.Width = MSDK_ALIGN32(m_pInputParams->Width);
                m_mfxVideoParams.mfx.FrameInfo.Height = MSDK_ALIGN32(m_pInputParams->Height);
                m_mfxVideoParams.mfx.FrameInfo.CropW = m_pInputParams->Width;
                m_mfxVideoParams.mfx.FrameInfo.CropH = m_pInputParams->Height;
                m_mfxVideoParams.mfx.FrameInfo.FourCC = m_pInputParams->fourcc;
                if (!m_mfxVideoParams.mfx.FrameInfo.FourCC)
                    m_mfxVideoParams.mfx.FrameInfo.FourCC = MFX_FOURCC_RGB4;
                if (!m_mfxVideoParams.mfx.FrameInfo.ChromaFormat)
                {
                    if (MFX_FOURCC_NV12 == m_mfxVideoParams.mfx.FrameInfo.FourCC)
                        m_mfxVideoParams.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
                    else if (MFX_FOURCC_RGB4 == m_mfxVideoParams.mfx.FrameInfo.FourCC)
                        m_mfxVideoParams.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV444;
                }
            }

            for (; MFX_CODEC_CAPTURE != m_pInputParams->videoType;)
            {
                // parse bit stream and fill mfx params
                sts = m_pmfxDEC->DecodeHeader(&m_mfxBS, &m_mfxVideoParams);

                if (MFX_ERR_MORE_DATA == sts)
                {
                    if (m_mfxBS.MaxLength == m_mfxBS.DataLength)
                    {
                        sts = ExtendMfxBitstream(&m_mfxBS, m_mfxBS.MaxLength * 2);
                        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                    }
                    // read a portion of data
                    sts = m_FileReader->ReadNextFrame(&m_mfxBS);
                    if (MFX_ERR_MORE_DATA == sts &&
                        !(m_mfxBS.DataFlag & MFX_BITSTREAM_EOS))
                    {
                        m_mfxBS.DataFlag |= MFX_BITSTREAM_EOS;
                        sts = MFX_ERR_NONE;
                    }
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                    continue;
                }
                else
                {
                    // if input is interlaced JPEG stream
                    if (m_mfxBS.PicStruct == MFX_PICSTRUCT_FIELD_TFF || m_mfxBS.PicStruct == MFX_PICSTRUCT_FIELD_BFF)
                    {
                        m_mfxVideoParams.mfx.FrameInfo.CropH *= 2;
                        m_mfxVideoParams.mfx.FrameInfo.Height = MSDK_ALIGN16(m_mfxVideoParams.mfx.FrameInfo.CropH);
                        m_mfxVideoParams.mfx.FrameInfo.PicStruct = m_mfxBS.PicStruct;
                    }

                    break;
                }
            }

            // check DecodeHeader status
            if (MFX_WRN_PARTIAL_ACCELERATION == sts)
            {
                m_Logger->alert("CDecodingPipeline::InitMfxParams: Partial acceleration!");
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
            }
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // specify memory type between Decoder and VPP
            m_mfxVideoParams.IOPattern = (mfxU16)(m_bSysmemBetween ? MFX_IOPATTERN_OUT_SYSTEM_MEMORY : MFX_IOPATTERN_OUT_VIDEO_MEMORY);
            //// specify memory type
            //m_mfxVideoParams.IOPattern = (mfxU16)(m_memType != SYSTEM_MEMORY ? MFX_IOPATTERN_OUT_VIDEO_MEMORY : MFX_IOPATTERN_OUT_SYSTEM_MEMORY);

            m_mfxVideoParams.AsyncDepth = m_pInputParams->nAsyncDepth;

            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::CreateHWDevice()
        {
#if D3D_SURFACES_SUPPORT
            mfxStatus sts = MFX_ERR_NONE;

            HWND window = NULL;

#if MFX_D3D11_SUPPORT
            if (D3D11_MEMORY == m_memType)
                m_hwdev = new CD3D11Device();
            if (m_hwdev) {
                reinterpret_cast<CD3D11Device *>(m_hwdev)->DefineFormat((m_fourcc == MFX_FOURCC_A2RGB10) ? true : false);
            }
            else
#endif // #if MFX_D3D11_SUPPORT
                m_hwdev = new CD3D9Device();
            if (m_hwdev) {
                reinterpret_cast<CD3D9Device *>(m_hwdev)->DefineFormat((m_fourcc == MFX_FOURCC_A2RGB10) ? true : false);
            }

            if (NULL == m_hwdev)
                return MFX_ERR_MEMORY_ALLOC;

            sts = m_hwdev->Init(
                window, 0,
                MSDKAdapter::GetNumber(m_mfxSession));
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
#endif
            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::ResetDevice()
        {
            return m_hwdev->Reset();
        }

        mfxStatus CDecodingPipeline::AllocAndInitVppDoNotUse()
        {
            m_VppDoNotUse.NumAlg = 4;

            m_VppDoNotUse.AlgList = new mfxU32[m_VppDoNotUse.NumAlg];
            if (!m_VppDoNotUse.AlgList) return MFX_ERR_NULL_PTR;

            m_VppDoNotUse.AlgList[0] = MFX_EXTBUFF_VPP_DENOISE; // turn off denoising (on by default)
            m_VppDoNotUse.AlgList[1] = MFX_EXTBUFF_VPP_SCENE_ANALYSIS; // turn off scene analysis (on by default)
            m_VppDoNotUse.AlgList[2] = MFX_EXTBUFF_VPP_DETAIL; // turn off detail enhancement (on by default)
            m_VppDoNotUse.AlgList[3] = MFX_EXTBUFF_VPP_PROCAMP; // turn off processing amplified (on by default)

            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::InitVppParams()
        {
            m_mfxVppVideoParams.IOPattern = (mfxU16)(m_bSysmemBetween ? MFX_IOPATTERN_IN_SYSTEM_MEMORY : MFX_IOPATTERN_IN_VIDEO_MEMORY);

            m_mfxVppVideoParams.IOPattern |= (m_memType != SYSTEM_MEMORY) ? MFX_IOPATTERN_OUT_VIDEO_MEMORY : MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

            MSDK_MEMCPY_VAR(m_mfxVppVideoParams.vpp.In, &m_mfxVideoParams.mfx.FrameInfo, sizeof(mfxFrameInfo));
            MSDK_MEMCPY_VAR(m_mfxVppVideoParams.vpp.Out, &m_mfxVppVideoParams.vpp.In, sizeof(mfxFrameInfo));

            m_mfxVppVideoParams.vpp.Out.FourCC = m_fourcc;

            m_mfxVppVideoParams.AsyncDepth = m_mfxVideoParams.AsyncDepth;

            AllocAndInitVppDoNotUse();
            m_VppExtParams[0] = (mfxExtBuffer*)&m_VppDoNotUse;

            m_mfxVppVideoParams.ExtParam = &m_VppExtParams[0];
            m_mfxVppVideoParams.NumExtParam = 1;
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
                m_Logger->alert("CDecodingPipeline::AllocFrames: Partial acceleration!");
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
            }
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // respecify memory type between Decoder and VPP
            m_mfxVideoParams.IOPattern = (mfxU16)(m_bSysmemBetween ? MFX_IOPATTERN_OUT_SYSTEM_MEMORY : MFX_IOPATTERN_OUT_VIDEO_MEMORY);

            // recalculate number of surfaces required for decoder
            sts = m_pmfxDEC->QueryIOSurf(&m_mfxVideoParams, &Request);
            MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            if (Request.NumFrameSuggested < m_mfxVideoParams.AsyncDepth)
                return MFX_ERR_MEMORY_ALLOC;

            sts = InitVppParams();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = m_pmfxVPP->Query(&m_mfxVppVideoParams, &m_mfxVppVideoParams);
            MSDK_IGNORE_MFX_STS(sts, MFX_WRN_INCOMPATIBLE_VIDEO_PARAM);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // VppRequest[0] for input frames request, VppRequest[1] for output frames request
            sts = m_pmfxVPP->QueryIOSurf(&m_mfxVppVideoParams, VppRequest);
            if (MFX_WRN_PARTIAL_ACCELERATION == sts) {
                m_Logger->alert("CDecodingPipeline::AllocFrames: WARNING: partial acceleration\n");
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

            Request.Type |= (m_bSysmemBetween) ? MFX_MEMTYPE_SYSTEM_MEMORY : MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET;

            // alloc frames for decoder
            sts = m_pMFXAllocator->Alloc(m_pMFXAllocator->pthis, &Request, &m_mfxResponse);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // alloc frames for VPP
            VppRequest[1].NumFrameSuggested = VppRequest[1].NumFrameMin = nVppSurfNum;
            MSDK_MEMCPY_VAR(VppRequest[1].Info, &(m_mfxVppVideoParams.vpp.Out), sizeof(mfxFrameInfo));

            sts = m_pMFXAllocator->Alloc(m_pMFXAllocator->pthis, &(VppRequest[1]), &m_mfxVppResponse);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // prepare mfxFrameSurface1 array for decoder
            nVppSurfNum = m_mfxVppResponse.NumFrameActual;

            // AllocVppBuffers should call before AllocBuffers to set the value of m_OutputSurfacesNumber
            sts = AllocVppBuffers(nVppSurfNum);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

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
                    sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, m_mfxResponse.mids[i], &(m_pSurfaces[i].frame.Data));
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
                    sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, m_mfxVppResponse.mids[i], &(m_pVppSurfaces[i].frame.Data));
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

            if (m_memType != SYSTEM_MEMORY)
            {
#if D3D_SURFACES_SUPPORT
                sts = CreateHWDevice();
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                // provide device manager to MediaSDK
                mfxHDL hdl = NULL;
                mfxHandleType hdl_t =
#if MFX_D3D11_SUPPORT
                    D3D11_MEMORY == m_memType ? MFX_HANDLE_D3D11_DEVICE :
#endif // #if MFX_D3D11_SUPPORT
                    MFX_HANDLE_D3D9_DEVICE_MANAGER;

                sts = m_hwdev->GetHandle(hdl_t, &hdl);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                sts = m_mfxSession.SetHandle(hdl_t, hdl);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                // create D3D allocator
#if MFX_D3D11_SUPPORT
                if (D3D11_MEMORY == m_memType)
                {
                    m_pMFXAllocator = new D3D11FrameAllocator;
                    MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

                    D3D11AllocatorParams *pd3d11AllocParams = new D3D11AllocatorParams;
                    MSDK_CHECK_POINTER(pd3d11AllocParams, MFX_ERR_MEMORY_ALLOC);
                    pd3d11AllocParams->pDevice = reinterpret_cast<ID3D11Device *>(hdl);

                    m_pmfxAllocatorParams = pd3d11AllocParams;
                }
                else
#endif // #if MFX_D3D11_SUPPORT
                {
                    m_pMFXAllocator = new D3DFrameAllocator;
                    MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

                    D3DAllocatorParams *pd3dAllocParams = new D3DAllocatorParams;
                    MSDK_CHECK_POINTER(pd3dAllocParams, MFX_ERR_MEMORY_ALLOC);
                    pd3dAllocParams->pManager = reinterpret_cast<IDirect3DDeviceManager9 *>(hdl);

                    m_pmfxAllocatorParams = pd3dAllocParams;
                }

                /* In case of video memory we must provide MediaSDK with external allocator
                thus we demonstrate "external allocator" usage model.
                Call SetAllocator to pass allocator to mediasdk */
                sts = m_mfxSession.SetFrameAllocator(m_pMFXAllocator);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                m_bExternalAlloc = true;
#endif
            }
            else
            {
                // create system memory allocator
                m_pMFXAllocator = new SysMemFrameAllocator;
                MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

                /* In case of system memory we demonstrate "no external allocator" usage model.
                We don't call SetAllocator, MediaSDK uses internal allocator.
                We use system memory allocator simply as a memory manager for application*/
            }

            // initialize memory allocator
            sts = m_pMFXAllocator->Init(m_pmfxAllocatorParams);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            return MFX_ERR_NONE;
        }

        void CDecodingPipeline::DeleteFrames()
        {
            FreeBuffers();

            m_pCurrentFreeSurface = NULL;
            m_pCurrentFreeVppSurface = NULL;
            MSDK_SAFE_FREE(m_pCurrentFreeOutputSurface);

            // delete frames
            if (m_pMFXAllocator)
            {
                m_pMFXAllocator->Free(m_pMFXAllocator->pthis, &m_mfxResponse);
                m_pMFXAllocator->FreeFrames(&m_mfxVppResponse);
            }

            return;
        }

        void CDecodingPipeline::DeleteAllocator()
        {
            // delete allocator
            MSDK_SAFE_DELETE(m_pMFXAllocator);
            MSDK_SAFE_DELETE(m_pmfxAllocatorParams);
            MSDK_SAFE_DELETE(m_hwdev);
        }

        mfxStatus CDecodingPipeline::ResetDecoder()
        {
            mfxStatus sts = MFX_ERR_NONE;

            // close decoder
            sts = m_pmfxDEC->Close();
            MSDK_IGNORE_MFX_STS(sts, MFX_ERR_NOT_INITIALIZED);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // free allocated frames
            DeleteFrames();

            // initialize parameters with values from parsed header
            sts = InitMfxParams();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // in case of HW accelerated decode frames must be allocated prior to decoder initialization
            sts = AllocFrames();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // init decoder
            sts = m_pmfxDEC->Init(&m_mfxVideoParams);
            if (MFX_WRN_PARTIAL_ACCELERATION == sts)
            {
                m_Logger->alert("CDecodingPipeline::ResetDecoder: WARNING! Partial acceleration");
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
            }
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // init VPP
            sts = m_pmfxVPP->Init(&m_mfxVppVideoParams);
            if (MFX_WRN_PARTIAL_ACCELERATION == sts) {
                msdk_printf(MSDK_STRING("WARNING: partial acceleration\n"));
                MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
            }
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::DeliverOutput(mfxFrameSurface1* frame)
        {
            CAutoTimer timer_fwrite(m_tick_fwrite);

            mfxStatus sts = MFX_ERR_NONE;

            if (!frame)
            {
                return MFX_ERR_NULL_PTR;
            }

            if (m_bExternalAlloc)
            {
                sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, frame->Data.MemId, &(frame->Data));
            }

            if (MFX_ERR_NONE == sts)
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
                if (m_bExternalAlloc)
                {

                    sts = m_pMFXAllocator->Unlock(m_pMFXAllocator->pthis, frame->Data.MemId, &(frame->Data));
                }
            }
            return sts;
        }

        mfxStatus CDecodingPipeline::DeliverLoop(void)
        {
            mfxStatus res = MFX_ERR_NONE;

            while (!m_bStopDeliverLoop) {
                m_pDeliverOutputSemaphore->Wait();
                if (m_bStopDeliverLoop) {
                    continue;
                }
                if (MFX_ERR_NONE != m_error) {
                    continue;
                }
                msdkOutputSurface* pCurrentDeliveredSurface = m_DeliveredSurfacesPool.GetSurface();
                if (!pCurrentDeliveredSurface) {
                    m_error = MFX_ERR_NULL_PTR;
                    continue;
                }
                mfxFrameSurface1* frame = &(pCurrentDeliveredSurface->surface->frame);

                m_error = DeliverOutput(frame);
                ReturnSurfaceToBuffers(pCurrentDeliveredSurface);

                pCurrentDeliveredSurface = NULL;
                ++m_output_count;
                m_pDeliveredEvent->Signal();
            }
            return res;
        }

        unsigned int MFX_STDCALL CDecodingPipeline::DeliverThreadFunc(void* ctx)
        {
            CDecodingPipeline* pipeline = (CDecodingPipeline*)ctx;

            mfxStatus sts;
            sts = pipeline->DeliverLoop();

            return 0;
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
            if (!m_bFirstFrameInitialized)
            {
                InitForFirstFrame();
            }
            mfxFrameSurface1*   pOutSurface = NULL;
            mfxBitstream*       pBitstream = &m_mfxBS;
            mfxStatus           sts = MFX_ERR_NONE;
            bool                bErrIncompatibleVideoParams = false;
            CTimeInterval<>     decodeTimer(m_bIsCompleteFrame);
            MSDKThread          *pDeliverThread = NULL;

            mfxPayload dec_payload;
            mfxU64 ts;
            dec_payload.Data = (mfxU8*)pPayload->pBuffer;
            dec_payload.BufSize = pPayload->u32Size;
            dec_payload.Type = 5;
            while (((sts == MFX_ERR_NONE) || (MFX_ERR_MORE_DATA == sts) || (MFX_ERR_MORE_SURFACE == sts)) && (m_nFrames > m_output_count))
            {
                if (MFX_ERR_NONE != m_error) {
                    m_Logger->error("CDecodingPipeline::RunDecoding: DeliverOutput return error = ") << std::to_string(m_error);
                    break;
                }
                if (pBitstream && ((MFX_ERR_MORE_DATA == sts) || m_bIsCompleteFrame)) {
                    CAutoTimer timer_fread(m_tick_fread);
                    sts = m_FileReader->ReadNextFrame(pBitstream); // read more data to input bit stream

                    if (MFX_ERR_MORE_DATA == sts) {
                        // we almost reached end of stream, need to pull buffered data now
                        pBitstream = NULL;
                        sts = MFX_ERR_NONE;
                    }
                    else if (MFX_ERR_NONE != sts) {
                        break;
                    }
                }
                if ((MFX_ERR_NONE == sts) || (MFX_ERR_MORE_DATA == sts) || (MFX_ERR_MORE_SURFACE == sts)) {
                    // here we check whether output is ready, though we do not wait...
#ifndef __SYNC_WA
                    mfxStatus _sts = SyncOutputSurface(0);
                    if (MFX_ERR_UNKNOWN == _sts) {
                        sts = _sts;
                        break;
                    } else if (MFX_ERR_NONE == _sts) {
                        continue;
                    }
#endif
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
                    if (!m_pCurrentFreeSurface || !m_pCurrentFreeVppSurface || (m_OutputSurfacesPool.GetSurfaceCount() == m_mfxVppVideoParams.AsyncDepth)) {
#endif
                        // we stuck with no free surface available, now we will sync...
                        sts = SyncOutputSurface(MSDK_DEC_WAIT_INTERVAL);
                        if (MFX_ERR_MORE_DATA == sts) {
                            m_Logger->error("CDecodingPipeline::RunDecoding: FATAL! Failed to find output surface, that's a bug!");
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

                        dec_payload.NumBit = 100;

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
                        if (MFX_WRN_DEVICE_BUSY == sts) {
                            if (m_bIsCompleteFrame) {
                                //in low latency mode device busy leads to increasing of latency
                                //m_Logger->alert("CDecodingPipeline::RunDecoding: latency increased due to MFX_WRN_DEVICE_BUSY!");
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
                    else if ((MFX_ERR_MORE_DATA == sts) && !pBitstream) {
                        // that's it - we reached end of stream; now we need to render bufferred data...
                        do {
                            sts = SyncOutputSurface(MSDK_DEC_WAIT_INTERVAL);
                        } while (MFX_ERR_NONE == sts);

                        while (m_synced_count != m_output_count) {
                            m_pDeliveredEvent->Wait();
                        }

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
                    //msdkFrameSurface* surface = FindUsedSurface(pOutSurface);

                    //msdk_atomic_inc16(&(surface->render_lock));

                    //m_pCurrentFreeOutputSurface->surface = surface;
                    m_OutputSurfacesPool.AddSurface(m_pCurrentFreeOutputSurface);
                    m_pCurrentFreeOutputSurface = NULL;
                    m_pCurrentFreeVppSurface = NULL;
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

            MSDK_SAFE_DELETE(pDeliverThread);
            MSDK_SAFE_DELETE(m_pDeliverOutputSemaphore);
            MSDK_SAFE_DELETE(m_pDeliveredEvent);
            MSDK_SAFE_DELETE(pDeliverThread);
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
            const char* sMemType = m_memType == D3D9_MEMORY ? "d3d"
                : (m_memType == D3D11_MEMORY ? "d3d11" : "system");

            mfxIMPL impl;
            m_mfxSession.QueryIMPL(&impl);

            mfxVersion ver;
            m_mfxSession.QueryVersion(&ver);

            const char* sImpl = (MFX_IMPL_VIA_D3D11 == MFX_IMPL_VIA_MASK(impl)) ? "hw_d3d11"
                : (MFX_IMPL_HARDWARE & impl) ? "hw" : "sw";

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
