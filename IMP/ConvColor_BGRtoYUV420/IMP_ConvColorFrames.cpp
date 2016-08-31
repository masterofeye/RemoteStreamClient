#include "IMP_ConvColorFrames.hpp"
#include "..\Crop\IMP_CropFrames.hpp"
#include "opencv2\opencv.hpp"
#include "opencv2\cudev\common.hpp"
#include "opencv2\cudaimgproc.hpp"
#if defined (SERVER)
#include "..\ENC\NVENC\ENC_CudaInterop.hpp"
#include "..\ENC\Intel\ENC_Intel.hpp"
#endif

namespace RW{
	namespace IMP{
		namespace COLOR
		{
			IMP_ConvColorFrames::IMP_ConvColorFrames(std::shared_ptr<spdlog::logger> Logger) :
				RW::CORE::AbstractModule(Logger)
			{
			}


			IMP_ConvColorFrames::~IMP_ConvColorFrames()
			{
			}

			void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
			{
				switch (SubModuleType)
				{
				case CORE::tenSubModule::nenGraphic_Crop:
				{
					IMP::CROP::tstMyControlStruct *data = static_cast<IMP::CROP::tstMyControlStruct*>(*Data);
					data->pInput->_pgMat = this->pOutput->_pgMat;
                    break;
				}
#if defined (SERVER)
                case CORE::tenSubModule::nenEncode_NVIDIA:
				{
                    CUdeviceptr arrYUV;
                    size_t pitch;
                    
                    cudaError err = cudaMallocPitch((void**)&arrYUV, &pitch, this->pOutput->_pgMat->cols, this->pOutput->_pgMat->rows * 3 / 2);

                    IMP::IMP_Base imp;
                    tenStatus enStatus = imp.GpuMatToGpuYUV(this->pOutput->_pgMat, &arrYUV);

					RW::ENC::NVENC::tstMyControlStruct *data = static_cast<RW::ENC::NVENC::tstMyControlStruct*>(*Data);
                    this->pOutput->_pcuYUV420 = arrYUV;
                    data->pcuYUVArray = arrYUV;
                    SAFE_DELETE(this->pOutput->_pgMat);

					break;
				}
                case CORE::tenSubModule::nenEncode_INTEL:
                {
                    uint8_t *pu8Output = new uint8_t[this->pOutput->_pgMat->cols* this->pOutput->_pgMat->rows * 3 / 2];

                    IMP::IMP_Base imp;
                    tenStatus enStatus = imp.GpuMatToCpuYUV(this->pOutput->_pgMat, pu8Output);

                    RW::ENC::INTEL::tstMyControlStruct *data = static_cast<RW::ENC::INTEL::tstMyControlStruct*>(*Data);
                    this->pOutput->_pu8Array = pu8Output;
                    data->pInputArray = pu8Output;
                    SAFE_DELETE(this->pOutput->_pgMat);
                    break;
                }
#endif
				default:
					break;
				}
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

			tenStatus IMP_ConvColorFrames::DoRender(CORE::tstControlStruct * ControlStruct)
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
				if (!data->pInput)
				{
					m_Logger->error("DoRender: pInput is empty!");
					return tenStatus::nenError;
				}

				cv::cuda::GpuMat *pgMat = data->pInput->_pgMat;

                cv::cuda::cvtColor(*pgMat, *pgMat, cv::COLOR_BGR2YUV);// , 0, cv::cuda::Stream::Stream());

                data->pOutput->_pgMat = pgMat;

				if (enStatus != tenStatus::nenSuccess){
					m_Logger->error("DoRender: impBase.tensProcessOutput did not succeed!");
				}

#ifdef TRACE_PERFORMANCE
				RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
				m_Logger->trace() << "Time to DoRender for nenGraphic_Color module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
				return enStatus;
			}

			tenStatus IMP_ConvColorFrames::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
			{
				m_Logger->debug("Deinitialise nenGraphic_Color");
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