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
            MSDK_ZERO_MEMORY(m_mfxVideoParams);

            m_pMFXAllocator = NULL;
            m_pmfxAllocatorParams = NULL;
            m_memType = SYSTEM_MEMORY;
            m_bExternalAlloc = false;
            MSDK_ZERO_MEMORY(m_mfxResponse);

            m_pCurrentFreeSurface = NULL;
            m_pCurrentFreeOutputSurface = NULL;
            m_pCurrentOutputSurface = NULL;

            m_pDeliverOutputSemaphore = NULL;
            m_pDeliveredEvent = NULL;
            m_error = MFX_ERR_NONE;
            m_bStopDeliverLoop = false;

            m_bIsExtBuffers = false;
            m_bIsVideoWall = false;
            m_bIsCompleteFrame = false;
            m_bPrintLatency = false;

            m_bFirstFrameInitialized = false;

            m_nTimeout = 0;
            m_nMaxFps = 0;

            m_vLatency.reserve(1000); // reserve some space to reduce dynamic reallocation impact on pipeline execution

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
                    //m_FileReader.reset(new CH264FrameReader());
                    m_bIsCompleteFrame = true;
                    m_bPrintLatency = m_pInputParams->bCalLat;
                    break;
                case MFX_CODEC_JPEG:
                    //m_FileReader.reset(new CJPEGFrameReader());
                    m_bIsCompleteFrame = true;
                    m_bPrintLatency = m_pInputParams->bCalLat;
                    break;
                case CODEC_VP8:
                    //m_FileReader.reset(new CIVFFrameReader());
                    m_bIsCompleteFrame = true;
                    m_bPrintLatency = m_pInputParams->bCalLat;
                    break;
                default:
                    return MFX_ERR_UNSUPPORTED; // latency mode is supported only for H.264 and JPEG codecs
                }
            }

            m_memType = m_pInputParams->memType;
            m_nMaxFps = m_pInputParams->nMaxFPS;
            m_nFrames = m_pInputParams->nFrames ? m_pInputParams->nFrames : MFX_INFINITE;

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

            if ((m_pInputParams->videoType == MFX_CODEC_JPEG) && !CheckVersion(&version, MSDK_FEATURE_JPEG_DECODE)) {
                m_Logger->error("CDecodingPipeline::Init: Jpeg is not supported in the API version ") << version.Major << "." << version.Minor;
                return MFX_ERR_UNSUPPORTED;
            }
            if (m_pInputParams->bLowLat && !CheckVersion(&version, MSDK_FEATURE_LOW_LATENCY)) {
                m_Logger->error("CDecodingPipeline::Init: Low Latency mode is not supported API version ") << version.Major << "." << version.Minor;
                return MFX_ERR_UNSUPPORTED;
            }

            // create decoder
            m_pmfxDEC = new MFXVideoDECODE(m_mfxSession);
            MSDK_CHECK_POINTER(m_pmfxDEC, MFX_ERR_MEMORY_ALLOC);

            // set video type in parameters
            m_mfxVideoParams.mfx.CodecId = m_pInputParams->videoType;

            // prepare bit stream
            if (MFX_CODEC_CAPTURE != m_pInputParams->videoType)
            {
                sts = InitMfxBitstream(&m_mfxBS, 1024 * 1024);
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
                    if (m_pInputParams->videoType == CODEC_VP8 || m_pInputParams->videoType == MFX_CODEC_HEVC)
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

            DeleteFrames();

            if (m_bIsExtBuffers)
            {
                DeallocateExtMVCBuffers();
                DeleteExtBuffers();
            }

            m_pPlugin.reset();
            m_mfxSession.Close();
            //m_FileWriter.Close();
            //if (m_FileReader.get())
            //    m_FileReader->Close();

            // allocator if used as external for MediaSDK must be deleted after decoder
            DeleteAllocator();

            return;
        }

        mfxStatus CDecodingPipeline::InitMfxParams()
        {
            MSDK_CHECK_POINTER(m_pmfxDEC, MFX_ERR_NULL_PTR);
            mfxStatus sts = MFX_ERR_NONE;
            mfxU32 &numViews = m_pInputParams->numViews;

            for (; MFX_CODEC_CAPTURE != m_pInputParams->videoType;)
            {
                // trying to find PicStruct information in AVI headers
                if (m_mfxVideoParams.mfx.CodecId == MFX_CODEC_JPEG)
                    MJPEG_AVI_ParsePicStruct(&m_mfxBS);

                // parse bit stream and fill mfx params
                sts = m_pmfxDEC->DecodeHeader(&m_mfxBS, &m_mfxVideoParams);
                if (m_pPlugin.get() && m_pInputParams->videoType == CODEC_VP8 && !sts) {
                    // force set format to nv12 as the vp8 plugin uses yv12
                    m_mfxVideoParams.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
                }
                if (MFX_ERR_MORE_DATA == sts)
                {
                    if (m_mfxBS.MaxLength == m_mfxBS.DataLength)
                    {
                        sts = ExtendMfxBitstream(&m_mfxBS, m_mfxBS.MaxLength * 2);
                        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                    }
                    // read a portion of data
                    //sts = m_FileReader->ReadNextFrame(&m_mfxBS);
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

            numViews = 1;

            // specify memory type
            m_mfxVideoParams.IOPattern = (mfxU16)(m_memType != SYSTEM_MEMORY ? MFX_IOPATTERN_OUT_VIDEO_MEMORY : MFX_IOPATTERN_OUT_SYSTEM_MEMORY);

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
            else
#endif // #if MFX_D3D11_SUPPORT
                m_hwdev = new CD3D9Device();

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

        mfxStatus CDecodingPipeline::AllocFrames()
        {
            MSDK_CHECK_POINTER(m_pmfxDEC, MFX_ERR_NULL_PTR);

            mfxStatus sts = MFX_ERR_NONE;

            mfxFrameAllocRequest Request;

            mfxU16 nSurfNum = 0; // number of surfaces for decoder

            MSDK_ZERO_MEMORY(Request);

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

            if (Request.NumFrameSuggested < m_mfxVideoParams.AsyncDepth)
                return MFX_ERR_MEMORY_ALLOC;

            nSurfNum = MSDK_MAX(Request.NumFrameSuggested, 1);

            // prepare allocation request
            Request.NumFrameSuggested = Request.NumFrameMin = nSurfNum;

            // alloc frames for decoder
            sts = m_pMFXAllocator->Alloc(m_pMFXAllocator->pthis, &Request, &m_mfxResponse);
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
            MSDK_SAFE_FREE(m_pCurrentFreeOutputSurface);

            // delete frames
            if (m_pMFXAllocator)
            {
                m_pMFXAllocator->Free(m_pMFXAllocator->pthis, &m_mfxResponse);
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

            return MFX_ERR_NONE;
        }

        mfxStatus CDecodingPipeline::DeliverOutput(mfxFrameSurface1* frame)
        {
            CAutoTimer timer_fwrite(m_tick_fwrite);

            mfxStatus sts = MFX_ERR_NONE;

            if (!frame) {
                return MFX_ERR_NULL_PTR;
            }

            if (m_bExternalAlloc) {
                    sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, frame->Data.MemId, &(frame->Data));
                    if (MFX_ERR_NONE == sts) {

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
                m_Logger->trace("CDecodingPipeline::PrintPerFrameStat: ") << "Frame number: " << m_output_count << " fps: " << fps << " fread_fps: " << ((fps_fread < MY_THRESHOLD) ? fps_fread : 0.0 )<< " fwrite_fps: " <<
                    ((fps_fwrite < MY_THRESHOLD) ? fps_fwrite : 0.0);
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
            time_t start_time = time(0);
            MSDKThread * pDeliverThread = NULL;

            mfxPayload dec_payload;
            mfxU64 ts;
            dec_payload.Data = (mfxU8*)pPayload->pBuffer;
            dec_payload.BufSize = pPayload->u32Size;
            int count = 0;
            while (((sts == MFX_ERR_NONE) || (MFX_ERR_MORE_DATA == sts) || (MFX_ERR_MORE_SURFACE == sts)) && (m_nFrames > m_output_count)){
                if (MFX_ERR_NONE != m_error) {
                    m_Logger->error("CDecodingPipeline::RunDecoding: DeliverOutput return error = ") << std::to_string( m_error );
                    break;
                }
                if (pBitstream && ((MFX_ERR_MORE_DATA == sts) || m_bIsCompleteFrame)) {
                    CAutoTimer timer_fread(m_tick_fread);
                    //sts = m_FileReader->ReadNextFrame(pBitstream); // read more data to input bit stream

                    if (MFX_ERR_MORE_DATA == sts) {
                        if (!m_bIsVideoWall) {
                            // we almost reached end of stream, need to pull buffered data now
                            pBitstream = NULL;
                            sts = MFX_ERR_NONE;
                        }
                        else {
                            // videowall mode: decoding in a loop
                            //m_FileReader->Reset();
                            sts = MFX_ERR_NONE;
                            continue;
                        }
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
                    if (!m_pCurrentFreeSurface) {
                        m_pCurrentFreeSurface = m_FreeSurfacesPool.GetSurface();
                    }
#ifndef __SYNC_WA
                    if (!m_pCurrentFreeSurface) {
#else
                    if (!m_pCurrentFreeSurface || (m_OutputSurfacesPool.GetSurfaceCount() == m_mfxVideoParams.AsyncDepth)) {
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

                // exit by timeout
                if ((MFX_ERR_NONE == sts) && m_bIsVideoWall && (time(0) - start_time) >= m_nTimeout) {
                    sts = MFX_ERR_NONE;
                    break;
                }

                if ((MFX_ERR_NONE == sts) || (MFX_ERR_MORE_DATA == sts) || (MFX_ERR_MORE_SURFACE == sts)) {
                    if (m_bIsCompleteFrame) {
                        m_pCurrentFreeSurface->submit = m_timer_overall.Sync();
                    }
                    pOutSurface = NULL;
                    do {
                        sts = m_pmfxDEC->DecodeFrameAsync(pBitstream, &(m_pCurrentFreeSurface->frame), &pOutSurface, &(m_pCurrentFreeOutputSurface->syncp));

                        dec_payload.NumBit = 100;

                        while ((dec_payload.NumBit != 0))
                        {
                            sts = m_pmfxDEC->GetPayload(&ts, &dec_payload);
                            if ((sts == MFX_ERR_NONE) && (dec_payload.Type == 5))
                            {
                                memcpy(pPayload->pBuffer, dec_payload.Data, dec_payload.BufSize);

                                count++;
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
                    msdkFrameSurface* surface = FindUsedSurface(pOutSurface);

                    msdk_atomic_inc16(&(surface->render_lock));

                    m_pCurrentFreeOutputSurface->surface = surface;
                    m_OutputSurfacesPool.AddSurface(m_pCurrentFreeOutputSurface);
                    m_pCurrentFreeOutputSurface = NULL;
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
            //m_Logger->debug("CDecodingPipeline::PrintInfo: Decoding Sample Version = ") << MSDK_SAMPLE_VERSION;
            m_Logger->debug("CDecodingPipeline::PrintInfo: Input video CodecID = ") << m_mfxVideoParams.mfx.CodecId;//CodecIdToStr(m_mfxVideoParams.mfx.CodecId).c_str();
            m_Logger->debug("CDecodingPipeline::PrintInfo: Output format = ") << m_mfxVideoParams.mfx.FrameInfo.FourCC;//CodecIdToStr(m_mfxVideoParams.mfx.FrameInfo.FourCC).c_str(); 

            mfxFrameInfo Info = m_mfxVideoParams.mfx.FrameInfo;
            m_Logger->debug("CDecodingPipeline::PrintInfo:   Resolution = ") << Info.Width << " x " << Info.Height;
            m_Logger->debug("CDecodingPipeline::PrintInfo:   Crop X,Y,W,H = ") << Info.CropX << ", " << Info.CropY << ", " << Info.CropW << ", " << Info.CropH;

            mfxF64 dFrameRate = CalculateFrameRate(Info.FrameRateExtN, Info.FrameRateExtD);
            m_Logger->debug("CDecodingPipeline::PrintInfo:   Frame rate = ") << dFrameRate;

            const char* sMemType = m_memType == D3D9_MEMORY ? "d3d"
                : (m_memType == D3D11_MEMORY ? "d3d11" : "system");
            m_Logger->debug("CDecodingPipeline::PrintInfo: Memory type = ") << sMemType;

            mfxIMPL impl;
            m_mfxSession.QueryIMPL(&impl);

            const char* sImpl = (MFX_IMPL_VIA_D3D11 == MFX_IMPL_VIA_MASK(impl)) ? "hw_d3d11"
                : (MFX_IMPL_HARDWARE & impl) ? "hw" : "sw";
            m_Logger->debug("CDecodingPipeline::PrintInfo: MediaSDK impl = ") << sImpl;

            mfxVersion ver;
            m_mfxSession.QueryVersion(&ver);
            m_Logger->debug("CDecodingPipeline::PrintInfo: MediaSDK version = ") << ver.Major << "." << ver.Minor;


            return;
        }
    }
}
