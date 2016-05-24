#include "IMP_CropFrames.hpp"

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

namespace RW{
	namespace IMP{

        IMP_CropFrames::IMP_CropFrames(std::shared_ptr<spdlog::logger> Logger) :
            RW::CORE::AbstractModule(Logger)
        {
                m_pstRect = NULL;
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

			if (data == NULL)
			{
				m_Logger->error("Initialise: Data of tstMyInitialiseControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}

            *m_pstRect = data->stFrameRect;

            if (m_pstRect == NULL)
            {
                m_Logger->error("Initialise: Rect struct is empty!");
                enStatus = tenStatus::nenError;
                return enStatus;
            }

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Time to Initialise nenGraphic_Crop module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
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

			if (data == NULL)
			{
				m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}
            cInputBase input = data->cInput;

            IMP_Base impBase = IMP_Base();
            enStatus = impBase.tensProcessInput(&input);
            if (enStatus != tenStatus::nenSuccess)
            {
                m_Logger->error("DoRender: impBase.tensProcessInput did not succeed!");
                return enStatus;
            }

            cv::cuda::GpuMat gMat = impBase.cuGetGpuMat();
            if (gMat.data == NULL)
            {
                enStatus = tenStatus::nenError;
                m_Logger->error("DoRender: GpuMat is NULL!");
                return enStatus;
            }

            if (m_pstRect->iWidth > gMat.cols || m_pstRect->iHeight > gMat.rows
                || m_pstRect->iWidth == 0 || m_pstRect->iHeight == 0)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("DoRender: Invalid frame size parameters!");
				return enStatus;
			}
            else if (m_pstRect->iWidth < gMat.cols || m_pstRect->iHeight < gMat.rows)
			{
                cv::Rect rect(m_pstRect->iPosX, m_pstRect->iPosY, m_pstRect->iWidth, m_pstRect->iHeight);
                gMat = gMat(rect);
			}

            cOutputBase output = data->cOutput;

            impBase.vSetGpuMat(gMat);
            enStatus = impBase.tensProcessOutput(&output);
            if (enStatus != tenStatus::nenSuccess)
            {
                m_Logger->error("DoRender: impBase.tensProcessOutput did not succeed!");
            }

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Time to load DoRender nenGraphic_Crop module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
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
            m_Logger->trace() << "Time to load Deinitialise nenGraphic_Crop module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
			return enStatus;
		}
	}
}