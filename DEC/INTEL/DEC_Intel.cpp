

#include "DEC_Intel.hpp"
#include "pipeline_decode.h"
#include "..\..\VPL\QT_simple\VPL_FrameProcessor.hpp"
#include "..\..\IMP\ConvColor_NV12toRGB\IMP_ConvColorFramesNV12ToRGB.hpp"

namespace RW{
    namespace DEC{
        namespace INTEL{
            void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
            {
                switch (SubModuleType)
                {
                case RW::CORE::tenSubModule::nenPlayback_Simple:
                {
                    RW::VPL::QT_SIMPLE::tstMyControlStruct *data = static_cast<RW::VPL::QT_SIMPLE::tstMyControlStruct*>(*Data);
                    data->pstBitStream = this->pOutput;
                    data->pPayload = this->pPayload;
                    break;
                }
                case RW::CORE::tenSubModule::nenGraphic_ColorNV12ToRGB:
                {
                    RW::IMP::COLOR_NV12TORGB::tstMyControlStruct *data = static_cast<RW::IMP::COLOR_NV12TORGB::tstMyControlStruct*>(*Data);

                    data->pInput = new IMP::tstInputOutput;
                    data->pInput->cuDevice = 0;
                    data->pInput->pBitstream = this->pOutput;
                    data->pPayload = this->pPayload;
                    break;
                }
                default:
                    break;
                }
                SAFE_DELETE_ARRAY(this->pstEncodedStream->pBuffer);
                SAFE_DELETE(this->pstEncodedStream);
                //this->pOutput = nullptr;
                //this->pPayload = nullptr;
            }

            DEC_Intel::DEC_Intel(std::shared_ptr<spdlog::logger> Logger) :
                RW::CORE::AbstractModule(Logger),
				m_pPipeline(new CDecodingPipeline(Logger))
			{
            }

            DEC_Intel::~DEC_Intel()
            {
            }

            CORE::tstModuleVersion DEC_Intel::ModulVersion() {
                CORE::tstModuleVersion version = { 0, 1 };
                return version;
            }

            CORE::tenSubModule DEC_Intel::SubModulType()
            {
                return CORE::tenSubModule::nenDecoder_INTEL;
            }

            tenStatus DEC_Intel::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
            {

                tenStatus enStatus = tenStatus::nenSuccess;
                m_Logger->debug("Initialise nenDecoder_INTEL");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

                stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);

                if (!data)
                {
                    m_Logger->error("DEC_Intel::Initialise: Data of tstMyInitialiseControlStruct is empty!");
                    return tenStatus::nenError;
                }

                mfxStatus sts = MFX_ERR_NONE; // return value check

                if (!IsDecodeCodecSupported(data->inputParams->videoType))
                {
                    m_Logger->error("DEC_Intel::Initialise: Unsupported codec");
                    sts = MFX_ERR_UNSUPPORTED;
                }

                sts = m_pPipeline->Init(data->inputParams);

                if (sts != MFX_ERR_NONE)
                {
                    enStatus = tenStatus::nenError;
                }

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "DEC_Intel::Initialise: Time to Initialise for nenDecoder_INTEL module: " << (RW::CORE::HighResClock::diffMilli(t1, t2).count()) << "ms.";
#endif
                return enStatus;
            }

            tenStatus DEC_Intel::DoRender(CORE::tstControlStruct * ControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;

                m_Logger->debug("DoRender nenDecoder_INTEL");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
                stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
                if (!data)
                {
                    m_Logger->error("DEC_Intel::DoRender: Data of stMyControlStruct is empty!");
                    return tenStatus::nenError;
                }
                if (!data->pstEncodedStream)
                {
                    m_Logger->error("DEC_Intel::DoRender: data->pstEncodedStream is empty!");
                    return tenStatus::nenError;
                }
                else{
                    m_pPipeline->SetEncodedData(data->pstEncodedStream);
                }

                mfxStatus sts = MFX_ERR_NONE; // return value check

                for (;;)
                {
                    sts = m_pPipeline->RunDecoding();
                    if (sts == MFX_ERR_NONE)
                    {
                        RW::tstPayloadMsg *pMsg = m_pPipeline->GetPayloadMsg();
                        if (pMsg){
                            data->pPayload = new tstBitStream;
                            data->pPayload->pBuffer = pMsg;
                            data->pPayload->u32Size = (uint32_t) sizeof(RW::tstPayloadMsg);
                        }
                        else
                            m_Logger->error("DEC_Intel::DoRender: GetPayloadMsg failed!");

                        data->pOutput = m_pPipeline->GetOutput();
                        if (!data->pOutput)
                            m_Logger->error("DEC_Intel::DoRender: GetOutput failed!");

						static int counter;
						WriteBufferToFile(data->pOutput->pBuffer, data->pOutput->u32Size, "DEC", counter);

						break;
                    }
                    else if (MFX_ERR_INCOMPATIBLE_VIDEO_PARAM == sts || MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts || data->pOutput == nullptr)
                    {
                        if (MFX_ERR_INCOMPATIBLE_VIDEO_PARAM == sts)
                        {
                            m_Logger->error("DEC_Intel::DoRender: Incompatible video parameters detected.");
                        }
                        else if (data->pOutput == nullptr)
                        {
                            m_Logger->error("DEC_Intel::DoRender: Output data is NULL!");
                        }
                        else
                        {
                            m_Logger->error("DEC_Intel::DoRender: Hardware device was lost or returned unexpected error. Recovering ...");
                            sts = m_pPipeline->ResetDevice();
                        }
                        sts = m_pPipeline->ResetDecoder();
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }

                if (sts != MFX_ERR_NONE)
                {
                    enStatus = tenStatus::nenError;
                }
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "DEC_Intel::DoRender: Time to DoRender for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return enStatus;
            }

            tenStatus DEC_Intel::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenDecoder_INTEL");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "DEC_Intel::Deinitialise: Time to Deinitialise for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return tenStatus::nenSuccess;
            }

        }
    }
}


