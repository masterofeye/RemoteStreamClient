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

			cInputBase input1 = data->cInput1;
			cInputBase input2 = data->cInput2;

			if (data->bNeedConversion)
			{
				IMP_Base impBase1 = IMP_Base();
				enStatus = impBase1.Initialise(&input1);
				m_cuMat1 = impBase1.cuGetGpuMat();

				if (enStatus == tenStatus::nenSuccess)
				{
					IMP_Base impBase2 = IMP_Base();
					enStatus = impBase2.Initialise(&input1);
					m_cuMat2 = impBase2.cuGetGpuMat();
				}
			}
			else
			{
				m_cuMat1 = input1._gMat;
				m_cuMat2 = input2._gMat;
			}

			m_Logger->debug("Initialise");
			return enStatus;
		}

		tenStatus IMP_MergeFrames::DoRender(CORE::tstControlStruct * ControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);

			ApplyMerge(m_cuMat1, m_cuMat2, &m_cuMat1);

			m_Logger->debug("DoRender");
			return enStatus;
		}

		tenStatus IMP_MergeFrames::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyDeinitialiseControlStruct* data = static_cast<stMyDeinitialiseControlStruct*>(DeinitialiseControlStruct);
			cOutputBase output = data->cOutput;

			if (data->bNeedConversion)
			{
				IMP_Base impBase = IMP_Base();
				impBase.vSetGpuMat(m_cuMat1);
				enStatus = impBase.Deinitialise(&output);
			}
			else
			{
				*output._pgMat = m_cuMat1;
			}

			m_Logger->debug("Deinitialise");
			return enStatus;
		}

		tenStatus IMP_MergeFrames::ApplyMerge(cv::cuda::GpuMat gMat1, cv::cuda::GpuMat gMat2, cv::cuda::GpuMat *pgMat)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			if (gMat1.type() != gMat2.type()
				|| gMat2.cols <= 0 || gMat2.rows <= 0
				|| gMat1.cols <= 0 || gMat1.rows <= 0)
			{
				enStatus = tenStatus::nenError;
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

			return enStatus;
		}
	}
}