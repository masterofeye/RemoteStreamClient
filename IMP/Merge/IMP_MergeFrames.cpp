#include "IMP_MergeFrames.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/cudev/common.hpp"
#include "..\ConvColor_BGRtoNV12\IMP_ConvColorFramesBGRToNV12.hpp"
#include "..\Crop\IMP_CropFrames.hpp"
#if defined (SERVER)
#include "..\..\ENC\NVENC\ENC_CudaInterop.hpp"
#include "..\..\ENC\Intel\ENC_Intel.hpp"
#endif

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
					data->pInput = this->pOutput;
                    data->pPayload = this->pPayload;
                    data->pvOutput = nullptr;
                    break;
				}
				case CORE::tenSubModule::nenGraphic_ColorBGRToNV12:
				{
					auto* data = static_cast<IMP::COLOR_BGRTONV12::tstMyControlStruct*>(*Data);
					data->pData = this->pOutput;
                    data->pPayload = this->pPayload;
                    break;
				}
#if defined (SERVER)
                case CORE::tenSubModule::nenEncode_NVIDIA:
                {
                    CUdeviceptr arrYUV;
                    size_t pitch;

                    cudaError err = cudaMallocPitch((void**)&arrYUV, &pitch, this->pOutput->cols, this->pOutput->rows * 3 / 2);
                    if (err != CUDA_SUCCESS)
                        printf("RW::IMP::MERGE::stMyControlStruct::UpdateData case CORE::tenSubModule::nenEncode_NVIDIA: cudaMallocPitch failed!");

                    IMP::IMP_Base imp;
                    tenStatus enStatus = imp.GpuMatToGpuNV12(this->pOutput, &arrYUV);
                    if (enStatus != tenStatus::nenSuccess)
                        printf("RW::IMP::MERGE::stMyControlStruct::UpdateData case CORE::tenSubModule::nenEncode_NVIDIA: GpuMatToGpuYUV failed!");

                    RW::ENC::NVENC::tstMyControlStruct *data = static_cast<RW::ENC::NVENC::tstMyControlStruct*>(*Data);

                    data->pcuYUVArray = arrYUV;
                    data->pPayload = this->pPayload;

                    SAFE_DELETE(this->pOutput);
                    break;
                }
                case CORE::tenSubModule::nenEncode_INTEL:
                {
                    uint32_t u32Size = this->pOutput->cols* this->pOutput->rows * 3 / 2;
                    uint8_t *pu8Output = new uint8_t[u32Size];

                    IMP::IMP_Base imp;
                    tenStatus enStatus = imp.GpuMatToCpuNV12(this->pOutput, pu8Output);
                    if (enStatus != tenStatus::nenSuccess)
                        printf("RW::IMP::MERGE::stMyControlStruct::UpdateData case CORE::tenSubModule::nenEncode_INTEL: GpuMatToCpuYUV failed!");

                    RW::ENC::INTEL::tstMyControlStruct *data = static_cast<RW::ENC::INTEL::tstMyControlStruct*>(*Data);

                    data->pInput = new RW::tstBitStream;
                    data->pInput->pBuffer = pu8Output;
                    data->pInput->u32Size = u32Size;

                    data->pPayload = this->pPayload;

                    SAFE_DELETE(this->pOutput);
                    break;
                }
#endif
                default:
					break;
				}
                uint8_t count = 1;
                while (this->pvInput->size() > count){
                    SAFE_DELETE(this->pvInput->at(count));
                    count++;
                }
                SAFE_DELETE(this->pvInput);
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

				for (int iIndex = 0; iIndex < data->pvInput->size(); iIndex++)
				{
                    cv::cuda::GpuMat *pgMat = data->pvInput->at(iIndex);
					
					if (!data->pOutput)
					{
						data->pOutput = pgMat;
					}
					else
					{
						enStatus = ApplyMerge(data->pOutput, pgMat);
					}
                    if (enStatus != tenStatus::nenSuccess || data->pOutput == nullptr)
                    {
                        m_Logger->error("DoRender: Failed!");
                        return enStatus;
                    }
                    //cv::Mat mat;
                    //data->pOutput->download(mat);
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