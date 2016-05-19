#include "IMP_CropFrames.hpp"


namespace RW{
	namespace IMP{

        IMP_CropFrames::IMP_CropFrames(std::shared_ptr<spdlog::logger> Logger) :
            RW::CORE::AbstractModule(Logger)
        {
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

		tenStatus IMP_CropFrames::DoRender(CORE::tstControlStruct * ControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);

			if (data == NULL)
			{
				m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}
			if (m_cuMat.data == NULL)
			{
				m_Logger->error("DoRender: Data of cuMat is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			if (data->stFrameRect.iWidth > m_cuMat.cols || data->stFrameRect.iHeight > m_cuMat.rows
				|| data->stFrameRect.iWidth == 0 || data->stFrameRect.iHeight == 0)
			{
				enStatus = tenStatus::nenError;
				m_Logger->error("DoRender: Invalid frame size parameters!");
				return enStatus;
			}
			else if (data->stFrameRect.iWidth < m_cuMat.cols || data->stFrameRect.iHeight < m_cuMat.rows)
			{
				cv::Rect rect(data->stFrameRect.iPosX, data->stFrameRect.iPosY, data->stFrameRect.iWidth, data->stFrameRect.iHeight);
				m_cuMat = m_cuMat(rect);
			}

			m_Logger->debug("DoRender");
			return enStatus;
		}

		tenStatus IMP_CropFrames::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
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