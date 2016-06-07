#include "IMP_MergeFrames.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/cudev/common.hpp"

namespace RW{
	namespace IMP{

		IMP_MergeFrames::IMP_MergeFrames(std::shared_ptr<spdlog::logger> Logger) :
			RW::CORE::AbstractModule(Logger)
		{
		}


		IMP_MergeFrames::~IMP_MergeFrames()
		{
		}

		CORE::tstModuleVersion IMP_MergeFrames::ModulVersion() {
			CORE::tstModuleVersion version = { 0, 1 };
			return version;
		}

		CORE::tenSubModule IMP_MergeFrames::SubModulType()
		{
			return CORE::tenSubModule::nenGraphic_Merge;
		}

		tenStatus IMP_MergeFrames::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

            m_Logger->debug("Initialise nenGraphic_Merge");

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
			m_Logger->trace() << "Time to Initialise for nenGraphic_Merge module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return enStatus;
		}

		tenStatus IMP_MergeFrames::DoRender(CORE::tstControlStruct * ControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

            m_Logger->debug("DoRender nenGraphic_Merge");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

			stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);

			if (data == nullptr)
			{
				m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
                return tenStatus::nenError;
			}
            if (data->pcInput == nullptr)
            {
                m_Logger->error("DoRender: Input is empty!");
                return tenStatus::nenError;
            }

            cv::cuda::GpuMat *pgMat1; 
            cv::cuda::GpuMat *pgMat2;

            IMP_Base impBase1 = IMP_Base(m_Logger);
            {
                enStatus = impBase1.tensProcessInput(data->pcInput->_pInput1, data->pcOutput);
                pgMat1 = impBase1.cuGetGpuMat();
                if (enStatus != tenStatus::nenSuccess || pgMat1 == nullptr)
                {
                    m_Logger->error("Initialise: impBase.Initialise for gMat1 did not succeed!");
                    return enStatus;
                }
            }
            {
                IMP_Base impBase2 = IMP_Base(m_Logger);
                enStatus = impBase2.tensProcessInput(data->pcInput->_pInput2, data->pcOutput);
                cv::cuda::GpuMat *pgMat2 = impBase2.cuGetGpuMat();
                if (enStatus != tenStatus::nenSuccess || pgMat2 == nullptr)
                {
                    m_Logger->error("Initialise: impBase.Initialise for gMat2 did not succeed!");
                    return enStatus;
                }

                enStatus = ApplyMerge(pgMat1, pgMat2);

				//delete Mat2 outside where it has been created!
                //if (pgMat2)
                //{
                //    delete pgMat2;
                //    pgMat2 = nullptr;
                //}
                if (enStatus != tenStatus::nenSuccess || pgMat1 == nullptr)
                {
                    m_Logger->error("DoRender: ApplyMerge did not succeed!");
                    return enStatus;
                }
            }

            impBase1.vSetGpuMat(pgMat1);
            enStatus = impBase1.tensProcessOutput(data->pcOutput);

            if (enStatus != tenStatus::nenSuccess || data->pcOutput == nullptr)
            {
                m_Logger->error("DoRender: impBase.tensProcessOutput did not succeed!");
                return enStatus;
            }

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
			m_Logger->trace() << "Time to DoRender for nenGraphic_Merge module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
             return enStatus;
		}

		tenStatus IMP_MergeFrames::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
		{
            m_Logger->debug("Deinitialise nenGraphic_Merge");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
			m_Logger->trace() << "Time to Deinitialise for nenGraphic_Merge module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return tenStatus::nenSuccess;
		}

		tenStatus IMP_MergeFrames::ApplyMerge(cv::cuda::GpuMat *pgMat1, cv::cuda::GpuMat *pgMat2)
		{
            if (pgMat1 == nullptr || pgMat2 == nullptr)
            {
                m_Logger->error("ApplyMerge: gMat1 or gMat2 is empty!");
                return tenStatus::nenError;
            }

			tenStatus enStatus = tenStatus::nenSuccess;
			if (pgMat1->type() != pgMat2->type()
				|| pgMat2->cols <= 0 || pgMat2->rows <= 0
				|| pgMat1->cols <= 0 || pgMat1->rows <= 0)
			{
                m_Logger->error("ApplyMerge: Invalid frame parameters (size or type)!");
                return tenStatus::nenError;
			}

            // equalizing rows
            if (pgMat1->rows > pgMat2->rows)
            {
                cv::Size sSize = cv::Size(pgMat2->cols, pgMat1->rows);
                cv::resize(*pgMat2, *pgMat2, sSize);
            }
            else
            {
                cv::Size sSize = cv::Size(pgMat1->cols, pgMat2->rows);
                cv::resize(*pgMat1, *pgMat1, sSize);
            }

            // horizontal concatenation
            cv::hconcat(*pgMat1, *pgMat2, *pgMat1);

            if (pgMat1 == nullptr)
            {
                m_Logger->error("ApplyMerge: gMat1 is empty! Apply Merge failed.");
                return tenStatus::nenError;
            }

			return enStatus;
		}
	}
}