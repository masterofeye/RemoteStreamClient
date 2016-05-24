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
			stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);

			if (data == NULL)
			{
				m_Logger->error("Initialise: Data of tstMyInitialiseControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point m_tStart = RW::CORE::HighResClock::now();
#endif

			m_Logger->debug("Initialise");
			return enStatus;
		}

		tenStatus IMP_MergeFrames::DoRender(CORE::tstControlStruct * ControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);

			if (data == NULL)
			{
				m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}
            cInputBase input1 = *(data->cInput._pInput1);
            cInputBase input2 = *(data->cInput._pInput2);

            IMP_Base impBase1 = IMP_Base();
            enStatus = impBase1.tensProcessInput(&input1);
            if (enStatus != tenStatus::nenSuccess)
            {
                m_Logger->error("Initialise: impBase.Initialise did not succeed!");
            }

            cv::cuda::GpuMat gMat1 = impBase1.cuGetGpuMat();

            if (enStatus != tenStatus::nenSuccess)
            {
                return enStatus;
            }
            IMP_Base impBase2 = IMP_Base();
            enStatus = impBase2.tensProcessInput(&input2);
            if (enStatus != tenStatus::nenSuccess)
            {
                m_Logger->error("Initialise: impBase.Initialise did not succeed!");
            }

            cv::cuda::GpuMat gMat2 = impBase2.cuGetGpuMat();

            enStatus = ApplyMerge(gMat1, gMat2, &gMat1);
            if (enStatus != tenStatus::nenSuccess || gMat1.data == NULL)
			{
				m_Logger->error("DoRender: ApplyMerge did not succeed!");
			}

            cOutputBase output = data->cOutput;

            impBase1.vSetGpuMat(gMat1);
            enStatus = impBase1.tensProcessOutput(&output);
            if (enStatus != tenStatus::nenSuccess)
            {
                m_Logger->error("DoRender: impBase.tensProcessOutput did not succeed!");
            }

			m_Logger->debug("DoRender");
			return enStatus;
		}

		tenStatus IMP_MergeFrames::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
		{

#ifdef TRACE_PERFORMANCE
            if (m_u32NumFramesEncoded > 0)
            {
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Execution Time of Module ENC: " << RW::CORE::HighResClock::diffMilli(m_tStart, t1).count() << "ms.";
                m_Logger->trace() << "Number of encoded files: " << m_u32NumFramesEncoded;
            }
#endif
            m_Logger->debug("Deinitialise");
			return tenStatus::nenSuccess;
		}

		tenStatus IMP_MergeFrames::ApplyMerge(cv::cuda::GpuMat gMat1, cv::cuda::GpuMat gMat2, cv::cuda::GpuMat *pgMat)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			if (gMat1.type() != gMat2.type()
				|| gMat2.cols <= 0 || gMat2.rows <= 0
				|| gMat1.cols <= 0 || gMat1.rows <= 0)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("ApplyMerge: Invalid frame parameters (size or type)!");
			}

			int iRows = (gMat1.rows > gMat2.rows ? gMat1.rows : gMat2.rows);
			cv::cuda::GpuMat gMat = cv::cuda::GpuMat();
			gMat.create(iRows, (gMat1.cols + gMat2.cols), gMat1.type());

			cv::Rect rect1(0, 0, gMat1.cols, gMat1.rows);
			cv::Rect rect2(gMat1.cols, 0, gMat2.cols, gMat2.rows);
			cv::Rect rect(0, 0, (gMat1.cols + gMat2.cols), iRows);

			gMat(rect1) = gMat1;
			gMat(rect2) = gMat2;
			*pgMat = gMat(rect);
			if (pgMat == NULL)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("ApplyMerge: pgMat is empty!");
			}

			return enStatus;
		}
	}
}