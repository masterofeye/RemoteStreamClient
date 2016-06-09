#include "IMP_MergeFrames.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/cudev/common.hpp"
#include "IMP_ConvColorFrames.hpp"
#include "IMP_CropFrames.hpp"

namespace RW{
	namespace IMP{
		namespace MERGE
		{
			IMP_MergeFrames::IMP_MergeFrames(std::shared_ptr<spdlog::logger> Logger) :
				RW::CORE::AbstractModule(Logger)
			{
			}

			IMP_MergeFrames::~IMP_MergeFrames()
			{
			}

			void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
			{
				switch (SubModuleType)
				{
				case CORE::tenSubModule::nenGraphic_Crop:
				{
					IMP::CROP::tstMyControlStruct *data = static_cast<IMP::CROP::tstMyControlStruct*>(*Data);
					data->pInput->_pgMat = this->pOutput;
				}
				case CORE::tenSubModule::nenGraphic_Color:
				{
					IMP::COLOR::tstMyControlStruct *data = static_cast<IMP::COLOR::tstMyControlStruct*>(*Data);
					data->pInput->_pgMat = this->pOutput;
					break;
				}
				default:
					break;
				}

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

				if (!data)
				{
					m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
					return tenStatus::nenError;
				}
				if (data->pvInput->empty())
				{
					m_Logger->error("DoRender: pvInput is empty!");
					return tenStatus::nenError;
				}
				if (!data->pOutput)
				{
					m_Logger->error("DoRender: pOutput is empty!");
					return tenStatus::nenError;
				}

				IMP_Base impBase = IMP_Base(m_Logger);

				for (int iIndex = 0; iIndex < data->pvInput->size(); iIndex++)
				{
					if (data->pOutput == data->pvInput->at(iIndex)->_pgMat)
					{
						iIndex++;
					}
					cInputBase *pInput = data->pvInput->at(iIndex);

					enStatus = impBase.tensProcessInput(pInput, data->pOutput);
					cv::cuda::GpuMat *pgMat = impBase.cuGetGpuMat();
					if (enStatus != tenStatus::nenSuccess || pgMat == nullptr)
					{
						m_Logger->error("Initialise: impBase.Initialise for gMat1 did not succeed!");
						return enStatus;
					}

					if (!data->pOutput)
					{
						data->pOutput = pgMat;
					}
					else
					{
						enStatus = ApplyMerge(data->pOutput, pgMat);
					}
				}

				if (enStatus != tenStatus::nenSuccess || data->pOutput == nullptr)
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
					|| pgMat2->cols <= 0 || pgMat2->rows <= 0)
				{
					m_Logger->error("ApplyMerge: Invalid frame parameters (size or type)!");
					return tenStatus::nenError;
				}

				int iRows = (pgMat1->rows > pgMat2->rows) ? pgMat1->rows : pgMat2->rows;
				cv::cuda::GpuMat gMat(iRows, (pgMat1->cols + pgMat2->cols), pgMat1->type());

				cv::Rect rect1(0, 0, pgMat1->cols, pgMat1->rows);
				cv::Rect rect2(pgMat1->cols, 0, pgMat2->cols, pgMat2->rows);
				pgMat1->copyTo(gMat(rect1));
				pgMat2->copyTo(gMat(rect2));
				//gMat(rect1) = *pgMat1;
				//gMat(rect2) = *pgMat2;

				*pgMat1 = gMat;
	
				if (pgMat1 == nullptr)
				{
					m_Logger->error("ApplyMerge: gMat1 is empty! Apply Merge failed.");
					return tenStatus::nenError;
				}

				return enStatus;
			}
		}
	}
}