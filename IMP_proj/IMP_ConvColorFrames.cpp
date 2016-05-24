#include "IMP_ConvColorFrames.hpp"
#include "opencv2/opencv.hpp"
//#include "opencv2/core/cuda.hpp"
#include "opencv2/cudev/common.hpp"
#include "opencv2/cudaimgproc.hpp"

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif
namespace RW{
	namespace IMP{

		IMP_ConvColorFrames::IMP_ConvColorFrames(std::shared_ptr<spdlog::logger> Logger) :
			RW::CORE::AbstractModule(Logger)
		{
			}


		IMP_ConvColorFrames::~IMP_ConvColorFrames()
		{
		}

		CORE::tstModuleVersion IMP_ConvColorFrames::ModulVersion() {
			CORE::tstModuleVersion version = { 0, 1 };
			return version;
		}

		CORE::tenSubModule IMP_ConvColorFrames::SubModulType()
		{
			return CORE::tenSubModule::nenGraphic_Color;
		}

		tenStatus IMP_ConvColorFrames::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
		{
            m_Logger->debug("Initialise nenGraphic_Color");

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            file_logger->trace() << "Time to Initialise for nenGraphic_Color module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return tenStatus::nenSuccess;
		}

		tenStatus IMP_ConvColorFrames::DoRender(CORE::tstControlStruct * ControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

            m_Logger->debug("DoRender nenGraphic_Color");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

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
            }

            cv::cuda::GpuMat gMat = impBase.cuGetGpuMat();
            if (gMat.data == NULL)
            {
                enStatus = tenStatus::nenError;
                m_Logger->error("DoRender: GpuMat is NULL!");
                return enStatus;
            }

            cv::cuda::cvtColor(gMat, gMat, cv::COLOR_RGB2YUV);
            if (gMat.data == NULL)
			{
				m_Logger->error("DoRender: Data of cuMat is empty! cvtColor did not succeed!");
				enStatus = tenStatus::nenError;
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
            m_Logger->trace() << "Time to DoRender for nenGraphic_Color module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif

			return enStatus;
		}

		tenStatus IMP_ConvColorFrames::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
		{
            m_Logger->debug("Deinitialise nenGraphic_Color");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Time to deinitialise nenGraphic_Color module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
			return enStatus;
		}
	}
}