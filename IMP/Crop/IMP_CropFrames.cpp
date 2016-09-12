#include "IMP_CropFrames.hpp"
#include "opencv2/cudev/common.hpp"
#include "..\Merge\IMP_MergeFrames.hpp"
#include "..\ConvColor_BGRtoYUV420\IMP_ConvColorFramesBGRToYUV420.hpp"
#if defined (SERVER)
#include "..\..\ENC\NVENC\ENC_CudaInterop.hpp"
#include "..\..\ENC\Intel\ENC_Intel.hpp"
#endif

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

				    data->pvInput = this->pvOutput;
                    data->pPayload = this->pPayload;
                    data->pOutput = nullptr;
					break;
				}
				case CORE::tenSubModule::nenGraphic_ColorBGRToYUV:
				{
					IMP::COLOR_BGRTOYUV::tstMyControlStruct *data = static_cast<IMP::COLOR_BGRTOYUV::tstMyControlStruct*>(*Data);
                    data->pData = this->pvOutput->at(0);

                    uint8_t count = 0;
                    while (this->pvOutput->size() > count){
                        SAFE_DELETE( this->pvOutput->at(count));
                        count++;
                    }
                    data->pPayload = this->pPayload;
                    SAFE_DELETE(this->pvOutput);
                    break;
				}
#if defined (SERVER)
                case CORE::tenSubModule::nenEncode_NVIDIA:
                {
                    CUdeviceptr arrYUV;
                    size_t pitch;

                    cudaError err = cudaMallocPitch((void**)&arrYUV, &pitch, this->pvOutput->at(0)->cols, this->pvOutput->at(0)->rows * 3 / 2);
                    if (err != CUDA_SUCCESS)
                        printf("RW::IMP::CROP::stMyControlStruct::UpdateData case CORE::tenSubModule::nenEncode_NVIDIA: cudaMallocPitch failed!");

                    IMP::IMP_Base imp;
                    tenStatus enStatus = imp.GpuMatToGpuYUV(this->pvOutput->at(0), &arrYUV);
                    if (enStatus != tenStatus::nenSuccess)
                        printf("RW::IMP::CROP::stMyControlStruct::UpdateData case CORE::tenSubModule::nenEncode_NVIDIA: GpuMatToGpuYUV failed!");

                    RW::ENC::NVENC::tstMyControlStruct *data = static_cast<RW::ENC::NVENC::tstMyControlStruct*>(*Data);
                    data->pcuYUVArray = arrYUV;

                    data->pstBitStream = nullptr;
                    data->pPayload = this->pPayload;

                    uint8_t count = 0;
                    while (this->pvOutput->size() > count){
                        SAFE_DELETE(this->pvOutput->at(count));
                        count++;
                    }
                    SAFE_DELETE(this->pvOutput);
                    break;
                }
                case CORE::tenSubModule::nenEncode_INTEL:
                {
                    uint32_t u32Size = this->pvOutput->at(0)->cols* this->pvOutput->at(0)->rows * 3 / 2;
                    uint8_t *pu8Output = new uint8_t[u32Size];

                    IMP::IMP_Base imp;
                    tenStatus enStatus = imp.GpuMatToCpuNV12(this->pvOutput->at(0), pu8Output);
                    if (enStatus != tenStatus::nenSuccess)
                    {
                        printf("RW::IMP::CROP::stMyControlStruct::UpdateData case CORE::tenSubModule::nenEncode_INTEL: GpuMatToCpuYUV failed!");
                    }

                    RW::ENC::INTEL::tstMyControlStruct *data = static_cast<RW::ENC::INTEL::tstMyControlStruct*>(*Data);

                    data->pInput = new RW::tstBitStream;
                    data->pInput->pBuffer = pu8Output;
                    data->pInput->u32Size = u32Size;

                    data->pstBitStream = nullptr;
                    data->pPayload = this->pPayload;

                    uint8_t count = 1;
                    while (this->pvOutput->size() > count){
                        SAFE_DELETE(this->pvOutput->at(count));
                        count++;
                    }
                    SAFE_DELETE(this->pvOutput);
                    break;
                }
#endif
                default:
					break;
				}
                SAFE_DELETE(this->pInput);
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
                if (!data->pInput)
                {
                    m_Logger->error("DoRender: data->pInput is empty!");
                    return tenStatus::nenError;
                }
                if (!data->pvOutput)
                    data->pvOutput = new std::vector < cv::cuda::GpuMat*>();

				cv::cuda::GpuMat *pgMat = data->pInput;
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
                        cv::cuda::GpuMat* pOutput = new cv::cuda::GpuMat();
                        *pOutput = (*pgMat)(m_vRect[iIndex]);
                        data->pvOutput->push_back(pOutput);
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