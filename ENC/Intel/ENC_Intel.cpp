
#include "ENC_Intel.hpp"

#include "pipeline_encode.h"
#include "../../SSR/live555/SSR_live555.hpp"


#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

namespace RW
{
    namespace ENC
    {
        namespace INTEL
        {
            void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
            {
                switch (SubModuleType)
                {
                case CORE::tenSubModule::nenStream_Simple:
                {
                    RW::SSR::LIVE555::tstMyControlStruct *data = static_cast<RW::SSR::LIVE555::tstMyControlStruct*>(*Data);
                    data->pstBitStream = (this->pstBitStream);
                    break;
                }
                default:
                    break;
                }
                SAFE_DELETE_ARRAY(this->pPayload->pBuffer);
                SAFE_DELETE_ARRAY(this->pInput->pBuffer);
                SAFE_DELETE(this->pPayload);
                SAFE_DELETE(this->pInput);
            }

            ENC_Intel::ENC_Intel(std::shared_ptr<spdlog::logger> Logger)
                : RW::CORE::AbstractModule(Logger)
            {

            }

            ENC_Intel::~ENC_Intel()
            {
            }

            CORE::tstModuleVersion ENC_Intel::ModulVersion() {
                CORE::tstModuleVersion version = { 0, 1 };
                return version;
            }

            CORE::tenSubModule ENC_Intel::SubModulType()
            {
                return CORE::tenSubModule::nenEncode_INTEL;
            }

            tenStatus ENC_Intel::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;

                m_Logger->debug("Initialise nenEncode_INTEL");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
                stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);
                if (data == nullptr)
                {
                    m_Logger->error("Initialise: Data of stMyInitialiseControlStruct is empty!");
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }

                m_pParams = data->pParams;

                if (0 == m_pParams->nWidth || 0 == m_pParams->nHeight)
                {
                    m_Logger->error("Initialise: Height and Width need to be specified!");
                    return tenStatus::nenError;
                }
                if (MFX_CODEC_MPEG2 != m_pParams->CodecId &&
                    MFX_CODEC_AVC != m_pParams->CodecId &&
                    MFX_CODEC_JPEG != m_pParams->CodecId &&
                    MFX_CODEC_VP8 != m_pParams->CodecId &&
                    MFX_CODEC_HEVC != m_pParams->CodecId)
                {
                    m_Logger->error("Initialise: Unknown codec!");
                    return tenStatus::nenError;
                }

                if (MFX_TARGETUSAGE_BEST_QUALITY != m_pParams->nTargetUsage && MFX_TARGETUSAGE_BEST_SPEED != m_pParams->nTargetUsage)
                {
                    m_pParams->nTargetUsage = MFX_TARGETUSAGE_BALANCED;
                }

                if (m_pParams->dFrameRate <= 0)
                {
                    m_pParams->dFrameRate = 30;
                }

                // if no destination picture width or height wasn't specified set it to the source picture size
                if (m_pParams->nDstWidth == 0)
                {
                    m_pParams->nDstWidth = m_pParams->nWidth;
                }

                if (m_pParams->nDstHeight == 0)
                {
                    m_pParams->nDstHeight = m_pParams->nHeight;
                }

                // calculate default bitrate based on the resolution (a parameter for encoder, so Dst resolution is used)
                if (m_pParams->nBitRate == 0)
                {
                    m_pParams->nBitRate = CalculateDefaultBitrate(m_pParams->CodecId, m_pParams->nTargetUsage, m_pParams->nDstWidth,
                        m_pParams->nDstHeight, m_pParams->dFrameRate);
                }

                // if nv12 option wasn't specified we expect input YUV file in YUV420 color format
                if (!m_pParams->ColorFormat)
                {
                    m_pParams->ColorFormat = MFX_FOURCC_YV12;
                }

                if (!m_pParams->nPicStruct)
                {
                    m_pParams->nPicStruct = MFX_PICSTRUCT_PROGRESSIVE;
                }

                if ((m_pParams->nRateControlMethod == MFX_RATECONTROL_LA) && (!m_pParams->bUseHWLib))
                {
                    m_Logger->error("Initialise: Look ahead BRC is supported only with -hw option!");
                    return tenStatus::nenError;
                }

                if ((m_pParams->nMaxSliceSize) && (!m_pParams->bUseHWLib))
                {
                    m_Logger->error("Initialise: MaxSliceSize option is supported only with -hw option!");
                    return tenStatus::nenError;
                }

                if ((m_pParams->nRateControlMethod == MFX_RATECONTROL_LA) && (m_pParams->CodecId != MFX_CODEC_AVC))
                {
                    m_Logger->error("Initialise: Look ahead BRC is supported only with H.264 encoder!");
                    return tenStatus::nenError;
                }

                if ((m_pParams->nMaxSliceSize) && (m_pParams->CodecId != MFX_CODEC_AVC))
                {
                    m_Logger->error("Initialise: axSliceSize option is supported only with H.264 encoderc!");
                    return tenStatus::nenError;
                }

                if (m_pParams->nLADepth && (m_pParams->nLADepth < 10 || m_pParams->nLADepth > 100))
                {
                    if ((m_pParams->nLADepth != 1) || (!m_pParams->nMaxSliceSize))
                    {
                        m_Logger->error("Initialise: Unsupported value of -lad parameter, must be in range [10, 100] or 1 in case of -mss option!");
                        return tenStatus::nenError;
                    }
                }

                if (m_pParams->nAsyncDepth == 0)
                {
                    m_pParams->nAsyncDepth = 4; //set by default;
                }

                // Ignoring user-defined Async Depth for LA
                if (m_pParams->nMaxSliceSize)
                {
                    m_pParams->nAsyncDepth = 1;
                }

                if (m_pParams->nRateControlMethod == 0)
                {
                    m_pParams->nRateControlMethod = MFX_RATECONTROL_CBR;
                }




                mfxStatus sts = MFX_ERR_NONE; // return value check

                // Choosing which pipeline to use
                m_pPipeline.reset(new CEncodingPipeline);

                if (!m_pPipeline.get())
                {
                    m_Logger->error("Initialise: pPipeline is invalid: MFX_ERR_MEMORY_ALLOC");
                    return tenStatus::nenError;
                }

                sts = m_pPipeline->Init(m_pParams);
                if (sts != MFX_ERR_NONE)
                {
                    m_Logger->error("Initialise: Init failed!");
                    return tenStatus::nenError;
                }

                m_pPipeline->PrintInfo();



#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Initialise for nenEncode_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return enStatus;
            }

            tenStatus ENC_Intel::DoRender(CORE::tstControlStruct * ControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;
                m_Logger->debug("DoRender nenEncode_INTEL");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

                stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
                if (data == nullptr)
                {
                    m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
                    return tenStatus::nenError;;
                }

                mfxStatus sts = MFX_ERR_NONE; // return value check

                m_pPipeline->SetInputData((uint8_t *)data->pInput->pBuffer);

                for (;;)
                {
                    sts = m_pPipeline->Run();

                    if (MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts)
                    {
                        msdk_printf(MSDK_STRING("\nERROR: Hardware device was lost or returned an unexpected error. Recovering...\n"));
                        sts = m_pPipeline->ResetDevice();
                        if (sts != MFX_ERR_NONE)
                        {
                            m_Logger->error("DoRender: ResetDevice failed!");
                            return tenStatus::nenError;
                        }

                        sts = m_pPipeline->ResetMFXComponents(m_pParams);
                        if (sts != MFX_ERR_NONE)
                        {
                            m_Logger->error("DoRender: ResetMFXComponents failed!");
                            return tenStatus::nenError;
                        }
                        continue;
                    }
                    else
                    {
                        if (sts != MFX_ERR_NONE)
                        {
                            m_Logger->error("DoRender: m_pPipeline->Run failed!");
                            return tenStatus::nenError;
                        }
                        data->pstBitStream = m_pPipeline->GetBitstream();

                        break;
                    }
                }

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to DoRender for nenEncode_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return enStatus;
            }

            tenStatus ENC_Intel::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenEncode_INTEL");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

                m_pPipeline->Close();


#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Deinitialise for nenEncode_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return tenStatus::nenSuccess;
            }
        }
    }
}