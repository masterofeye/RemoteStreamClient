#include "IMP_CropFrames.hpp"
#include "opencv2/cudev/common.hpp"
#include "IMP_MergeFrames.hpp"
#include "IMP_ConvColorFrames.hpp"

namespace RW{
	namespace IMP{
		namespace CROP{
			IMP_CropFrames::IMP_CropFrames(std::shared_ptr<spdlog::logger> Logger) :
				RW::CORE::AbstractModule(Logger)
			{
			}

			void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
			{
				switch (SubModuleType)
				{
				case CORE::tenSubModule::nenGraphic_Merge:
				{
					IMP::MERGE::tstMyControlStruct *data = static_cast<IMP::MERGE::tstMyControlStruct*>(*Data);
					for (int iIndex = 0; iIndex < data->pvInput->size(); iIndex++)
					{
						data->pvInput->at(iIndex)->_pgMat = this->pvOutput->at(iIndex);
					}
					break;
				}
				case CORE::tenSubModule::nenGraphic_Color:
				{
					IMP::COLOR::tstMyControlStruct *data = static_cast<IMP::COLOR::tstMyControlStruct*>(*Data);
					data->pInput->_pgMat = this->pvOutput->at(0);
				}
				default:
					break;
				}
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

				if (!data)
				{
					m_Logger->error("Initialise: Data of tstMyInitialiseControlStruct is empty!");
					return tenStatus::nenError;
				}
				if (data->vFrameRect.empty())
				{
					m_Logger->error("Initialise: Rect struct is empty!");
					return tenStatus::nenError;
				}
				m_vRect = data->vFrameRect;

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
					return tenStatus::nenError;
				}
				if (data->pvOutput->size() != m_vRect.size())
				{
					m_Logger->error("DoRender: Not enough ouput data for the related rectangles!");
					return tenStatus::nenError;
				}

				IMP_Base impBase = IMP_Base(m_Logger);
				enStatus = impBase.tensProcessInput(data->pInput, data->pvOutput->at(0));
				cv::cuda::GpuMat *pgMat = impBase.cuGetGpuMat();
				if (enStatus != tenStatus::nenSuccess || pgMat == nullptr)
				{
					m_Logger->error("DoRender: impBase.tensProcessInput did not succeed!");
					return enStatus;
				}

				cv::cuda::GpuMat originalImg = *pgMat;
				for (int iIndex = 0; iIndex < m_vRect.size(); iIndex++)
				{

					if (m_vRect[iIndex].width > pgMat->cols || m_vRect[iIndex].height > pgMat->rows
						|| m_vRect[iIndex].width == 0 || m_vRect[iIndex].height == 0)
					{
						m_Logger->error("DoRender: Invalid frame size parameters!");
						return tenStatus::nenError;
					}
					else if (m_vRect[iIndex].width < pgMat->cols || m_vRect[iIndex].height < pgMat->rows)
					{
						if (data->pvOutput->at(iIndex))
						{
							cv::cuda::GpuMat outMat = (*pgMat)(m_vRect[iIndex]);
							*(data->pvOutput->at(iIndex)) = outMat;
						}
						else
						{
							m_Logger->error("DoRender: Invalid output parameter!");
							return tenStatus::nenError;				
						}
					}
					pgMat = &originalImg;

					if (enStatus != tenStatus::nenSuccess || data->pvOutput->at(iIndex) == nullptr)
					{
						m_Logger->error("DoRender: impBase.tensProcessOutput did not succeed!");
					}
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
}