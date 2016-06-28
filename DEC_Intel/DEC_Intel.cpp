

#include "DEC_Intel.hpp"
#include "pipeline_decode.h"
#include "..\VideoPlayer\VideoPlayer.hpp"

namespace RW{
    namespace DEC{
        void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
        {
			switch (SubModuleType)
			{
			case RW::CORE::tenSubModule::nenPlayback_Simple:
			{
				RW::VPL::tstMyControlStruct *data = static_cast<RW::VPL::tstMyControlStruct*>(*Data);
				data->pstBitStream = this->pOutput;
				break;
			}
			default:
				break;
			}
        }



        DEC_Intel::DEC_Intel(std::shared_ptr<spdlog::logger> Logger) :
            RW::CORE::AbstractModule(Logger)
        {
                m_pPipeline = new CDecodingPipeline(m_Logger);
            }

        DEC_Intel::~DEC_Intel()
        {
            if (m_pPipeline)
            {
                delete m_pPipeline;
                m_pPipeline = nullptr;
            }
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

            if (data == NULL)
            {
                m_Logger->error("Initialise: Data of tstMyInitialiseControlStruct is empty!");
                return tenStatus::nenError;
            }

            mfxStatus sts = MFX_ERR_NONE; // return value check

            if (!IsDecodeCodecSupported(data->inputParams->videoType))
            {
                m_Logger->error("Unsupported codec");
                sts = MFX_ERR_UNSUPPORTED;
            }

			sts = m_pPipeline->Init(data->inputParams);

            // print stream info
            m_pPipeline->PrintInfo();

            if (sts != MFX_ERR_NONE)
            {
                enStatus = tenStatus::nenError;
            }

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Time to Initialise for nenDecoder_INTEL module: " << (RW::CORE::HighResClock::diffMilli(t1, t2).count()) << "ms.";
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
                m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
                return tenStatus::nenError;
            }
            if (!data->pOutput)
            {
                m_Logger->error("DoRender: pOutput of stMyControlStruct is empty!");
                return tenStatus::nenError;
            }

            mfxStatus sts = MFX_ERR_NONE; // return value check

            m_pPipeline->SetEncodedData(data->pstEncodedStream);
            m_pPipeline->SetOutput(data->pOutput);

            sts = m_pPipeline->RunDecoding(data->pPayload);
            data->pOutput = m_pPipeline->GetOutput();
            if (sts != MFX_ERR_NONE)
            {
                enStatus = tenStatus::nenError;
            }

            if (MFX_ERR_INCOMPATIBLE_VIDEO_PARAM == sts || MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts || data->pOutput == nullptr)
            {
                if (MFX_ERR_INCOMPATIBLE_VIDEO_PARAM == sts)
                {
                    m_Logger->error("Incompatible video parameters detected. Recovering...");
                }
                else
                {
                    m_Logger->error("Hardware device was lost or returned unexpected error. Recovering...");
                    sts = m_pPipeline->ResetDevice();
                }

                sts = m_pPipeline->ResetDecoder();
            }
            if (sts != MFX_ERR_NONE)
            {
                enStatus = tenStatus::nenError;
            }
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Time to DoRender for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
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
            m_Logger->trace() << "Time to Deinitialise for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return tenStatus::nenSuccess;
        }

    }

}


