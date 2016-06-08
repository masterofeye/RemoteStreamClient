#include "IMP_CropFrames.hpp"
#include "opencv2/cudev/common.hpp"

namespace RW{
	namespace IMP{

        IMP_CropFrames::IMP_CropFrames(std::shared_ptr<spdlog::logger> Logger) :
            RW::CORE::AbstractModule(Logger)
        {
                m_pstRect = nullptr;
            }


		IMP_CropFrames::~IMP_CropFrames()
		{
		}

		CORE::tstModuleVersion IMP_CropFrames::ModulVersion() {
			CORE::tstModuleVersion version = { 0, 1 };
			return version;
		}

		CORE::tenSubModule IMP_CropFrames::SubModulType()
		{
			return CORE::tenSubModule::nenGraphic_Crop;
		}

		tenStatus IMP_CropFrames::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
		{
            m_Logger->debug("Initialise nenGraphic_Crop");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);

			if (data == nullptr)
			{
				m_Logger->error("Initialise: Data of tstMyInitialiseControlStruct is empty!");
                return tenStatus::nenError;
			}
            if (data->pstFrameRect == nullptr)
            {
                m_Logger->error("Initialise: Rect struct is empty!");
                return tenStatus::nenError;
            }
            m_pstRect = data->pstFrameRect;

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
			m_Logger->trace() << "Time to Initialise for nenGraphic_Crop module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return enStatus;
		}

		tenStatus IMP_CropFrames::DoRender(CORE::tstControlStruct * ControlStruct)
		{
            m_Logger->debug("DoRender nenGraphic_Crop");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);

			if (data == nullptr)
			{
				m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}
            if (m_pstRect == nullptr)
            {
                m_Logger->error("DoRender: Rect struct is empty!");
                return tenStatus::nenError;
            }

            IMP_Base impBase = IMP_Base(m_Logger);
            enStatus = impBase.tensProcessInput(data->pcInput, data->pcOutput);
            cv::cuda::GpuMat *pgMat = impBase.cuGetGpuMat();
            if (enStatus != tenStatus::nenSuccess || pgMat == nullptr)
            {
                m_Logger->error("DoRender: impBase.tensProcessInput did not succeed!");
                return enStatus;
            }

            if (m_pstRect->iWidth > pgMat->cols || m_pstRect->iHeight > pgMat->rows
                || m_pstRect->iWidth == 0 || m_pstRect->iHeight == 0)
			{
				m_Logger->error("DoRender: Invalid frame size parameters!");
                return tenStatus::nenError;
			}
            else if (m_pstRect->iWidth < pgMat->cols || m_pstRect->iHeight < pgMat->rows)
			{
                cv::Rect rect(m_pstRect->iPosX, m_pstRect->iPosY, m_pstRect->iWidth, m_pstRect->iHeight);
                *pgMat = (*pgMat)(rect);
			}

            impBase.vSetGpuMat(pgMat);
            enStatus = impBase.tensProcessOutput(data->pcOutput);

            if (enStatus != tenStatus::nenSuccess || data->pcOutput == nullptr)
            {
                m_Logger->error("DoRender: impBase.tensProcessOutput did not succeed!");
            }

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
			m_Logger->trace() << "Time to DoRender for nenGraphic_Crop module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return enStatus;
		}

		tenStatus IMP_CropFrames::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
		{

            m_Logger->debug("Deinitialise nenGraphic_Crop");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
			m_Logger->trace() << "Time to Deinitialise for nenGraphic_Crop module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return tenStatus::nenSuccess;
		}
	}
}