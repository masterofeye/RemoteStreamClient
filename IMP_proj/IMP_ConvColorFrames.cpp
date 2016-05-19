#include "IMP_ConvColorFrames.hpp"
#include "opencv2/opencv.hpp"
//#include "opencv2/core/cuda.hpp"
#include "opencv2/cudev/common.hpp"
#include "opencv2/cudaimgproc.hpp"

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
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);

			if (data == NULL)
			{
				m_Logger->error("Initialise: Data of tstMyInitialiseControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			cInputBase input = data->cInput;

			if (data->bNeedConversion)
			{
				IMP_Base impBase = IMP_Base();
				enStatus = impBase.Initialise(&input);
				if (enStatus != tenStatus::nenSuccess)
				{
					m_Logger->error("Initialise: impBase.Initialise did not succeed!");
				}

				m_cuMat = impBase.cuGetGpuMat();
			}
			else
			{
				m_cuMat = input._gMat;
			}

			if (m_cuMat.data == NULL)
			{
				m_Logger->error("Initialise: Data of cuMat is empty! Initialise failed!");
			}

			m_Logger->debug("Initialise");
			return enStatus;
		}

		tenStatus IMP_ConvColorFrames::DoRender(CORE::tstControlStruct * ControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

			if (m_cuMat.data == NULL)
			{
				m_Logger->error("DoRender: Data of cuMat is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			cv::cuda::cvtColor(m_cuMat, m_cuMat, cv::COLOR_RGB2YUV);
			if (m_cuMat.data == NULL)
			{
				m_Logger->error("DoRender: Data of cuMat is empty! cvtColor did not succeed!");
				enStatus = tenStatus::nenError;
			}

			m_Logger->debug("DoRender");
			return enStatus;
		}
		tenStatus IMP_ConvColorFrames::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyDeinitialiseControlStruct* data = static_cast<stMyDeinitialiseControlStruct*>(DeinitialiseControlStruct);
			if (data == NULL)
			{
				m_Logger->error("Deinitialise: Data of stMyDeinitialiseControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}
			if (m_cuMat.data == NULL)
			{
				m_Logger->error("Deinitialise: Data of cuMat is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			cOutputBase output = data->cOutput;

			if (data->bNeedConversion)
			{
				IMP_Base impBase = IMP_Base();
				impBase.vSetGpuMat(m_cuMat);
				enStatus = impBase.Deinitialise(&output);
				if (enStatus != tenStatus::nenSuccess || output._pcuArray == NULL)
				{
					m_Logger->error("Deinitialise: impBase.Deinitialise did not succeed!");
				}
			}
			else
			{
				*output._pgMat = m_cuMat;
			}

			m_Logger->debug("Deinitialise");
			return enStatus;
		}
	}
}