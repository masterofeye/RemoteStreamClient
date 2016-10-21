#include "..\ConvColor_BGRtoNV12\IMP_ConvColorFramesBGRToNV12.hpp"
#include "..\Crop\IMP_CropFrames.hpp"
#include "opencv2\opencv.hpp"
#include "opencv2\cudev\common.hpp"
#include "opencv2\cudaimgproc.hpp"
#if defined (SERVER)
#include "..\..\ENC\NVENC\ENC_CudaInterop.hpp"
#include "..\..\ENC\Intel\ENC_Intel.hpp"
#endif

namespace RW{
	namespace IMP{
		namespace COLOR_BGRTONV12
		{
			IMP_ConvColorFramesBGRToNV12::IMP_ConvColorFramesBGRToNV12(std::shared_ptr<spdlog::logger> Logger) :
				RW::CORE::AbstractModule(Logger)
			{
			}


			IMP_ConvColorFramesBGRToNV12::~IMP_ConvColorFramesBGRToNV12()
			{
			}

			void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
			{
#ifdef TEST
				cv::Mat dummy(pData->rows, pData->cols, pData->type());
				pData->download(dummy);
				static int count;
				WriteBufferToFile(dummy.data, dummy.total() * dummy.channels(), "Server_IMP_YUV444", count);
#endif

				switch (SubModuleType)
				{
				case CORE::tenSubModule::nenGraphic_Crop:
				{
					IMP::CROP::tstMyControlStruct *data = static_cast<IMP::CROP::tstMyControlStruct*>(*Data);
					data->pInput = this->pData;
                    data->pPayload = this->pPayload;
                    break;
				}
#if defined (SERVER)
                case CORE::tenSubModule::nenEncode_NVIDIA:
				{
					RW::ENC::NVENC::tstMyControlStruct *data = static_cast<RW::ENC::NVENC::tstMyControlStruct*>(*Data);
					CUdeviceptr arrYUV;
                    IMP::IMP_Base imp;
                    tenStatus enStatus = imp.GpuMatToGpuNV12(this->pData, &arrYUV);
                    if (enStatus != tenStatus::nenSuccess)
                        printf("RW::IMP::COLOR_BGRTOYUV::stMyControlStruct::UpdateData case nenEncode_NVIDIA: GpuMatToGpuYUV failed!\n");

                    data->pcuYUVArray = arrYUV;
                    data->pPayload = this->pPayload;

                    SAFE_DELETE(this->pData);
                    break;
				}
                case CORE::tenSubModule::nenEncode_INTEL:
                {
                    uint32_t u32Size = this->pData->cols* this->pData->rows * 3 / 2;
                    uint8_t *pu8Output = new uint8_t[u32Size];

                    IMP::IMP_Base imp;
                    tenStatus enStatus = imp.GpuMatToCpuNV12(this->pData, pu8Output);
                    if (enStatus != tenStatus::nenSuccess)
                    {
                        printf("RW::IMP::COLOR_BGRTOYUV::stMyControlStruct::UpdateData case nenEncode_INTEL: GpuMatToCpuYUV failed!\n");
                    }
					
                    RW::ENC::INTEL::tstMyControlStruct *data = static_cast<RW::ENC::INTEL::tstMyControlStruct*>(*Data);

                    data->pInput = new RW::tstBitStream; 
                    data->pInput->pBuffer = pu8Output;
                    data->pInput->u32Size = u32Size;

                    data->pPayload = this->pPayload;

                    SAFE_DELETE(this->pData);
                    break;
                }
#endif
				default:
					break;
				}
            }

			CORE::tstModuleVersion IMP_ConvColorFramesBGRToNV12::ModulVersion() {
				CORE::tstModuleVersion version = { 0, 1 };
				return version;
			}

			CORE::tenSubModule IMP_ConvColorFramesBGRToNV12::SubModulType()
			{
                return CORE::tenSubModule::nenGraphic_ColorBGRToNV12;
			}

			tenStatus IMP_ConvColorFramesBGRToNV12::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
			{
				m_Logger->debug("Initialise nenGraphic_Color");
#ifdef TRACE_PERFORMANCE
				RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
				RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
				m_Logger->trace() << "Time to Initialise for nenGraphic_Color module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
				return tenStatus::nenSuccess;
			}

			tenStatus IMP_ConvColorFramesBGRToNV12::DoRender(CORE::tstControlStruct * ControlStruct)
			{
				tenStatus enStatus = tenStatus::nenSuccess;

				m_Logger->debug("DoRender nenGraphic_Color");
#ifdef TRACE_PERFORMANCE
				RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
				stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);

				if (!data)
				{
					m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
					return tenStatus::nenError;
				}
				if (!data->pData)
				{
					m_Logger->error("DoRender: pData is empty!");
					return tenStatus::nenError;
				}

                cv::cuda::cvtColor(*data->pData, *data->pData, cv::COLOR_BGR2YUV);// , 0, cv::cuda::Stream::Stream());

                if (enStatus != tenStatus::nenSuccess){
					m_Logger->error("DoRender: impBase.tensProcessOutput did not succeed!");
				}

#ifdef TRACE_PERFORMANCE
				RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
				m_Logger->trace() << "Time to DoRender for nenGraphic_Color module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
				return enStatus;
			}

			tenStatus IMP_ConvColorFramesBGRToNV12::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
			{
				m_Logger->debug("Deinitialise nenGraphic_ColorBGRToNV12");
#ifdef TRACE_PERFORMANCE
				RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
				RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
				m_Logger->trace() << "Time to Deinitialise for nenGraphic_Color module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
				return tenStatus::nenSuccess;
			}
		}
	}
}