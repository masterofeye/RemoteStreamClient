#include "IMP_ConvColorFrames.hpp"
#include "opencv2/opencv.hpp"
//#include "opencv2/core/cuda.hpp"
#include "opencv2/cudev/common.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "..\ENC_proj\ENC_CudaInterop.hpp"

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
				case CORE::tenSubModule::nenEncode_NVIDIA:
				{
					RW::ENC::tstMyControlStruct *data = static_cast<RW::ENC::tstMyControlStruct*>(*Data);

					data->pcuYUVArray = this->cuArray;
					break;
				}
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
				IMP_Base impBase = IMP_Base(m_Logger);

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

				bool bInternalGpuMat = false;
				cv::cuda::GpuMat *pgMat = data->pInput->_pgMat;
				if (!pgMat)
				{
					pgMat = new cv::cuda::GpuMat();
					bInternalGpuMat = true;
				}
				IMP::IMP_Base cBase = IMP_Base(m_Logger);
				enStatus = impBase.tensProcessInput(data->pInput, pgMat);
				pgMat = impBase.cuGetGpuMat();

				cv::cuda::cvtColor(*pgMat, *pgMat, cv::COLOR_BGR2YUV);// , 0, cv::cuda::Stream::Stream());

				//data->pcuArray = (CUarray*)pgMat;

				//size_t sArraySize = m_pgMat->cols *  m_pgMat->rows * sizeof(uint8_t);
				//cuMemcpyHtoA(*data->pcuArray, 0, m_pgMat->data, sArraySize);

				//CUDA_MEMCPY2D* pCopy = new CUDA_MEMCPY2D();
				//pCopy->srcHost = m_pgMat->data;
				//pCopy->srcMemoryType = CU_MEMORYTYPE_HOST;
				//pCopy->dstArray = *pOutput->_pcuArray;
				//pCopy->dstMemoryType = CU_MEMORYTYPE_ARRAY;
				//pCopy->Height = m_pgMat->rows;
				//pCopy->WidthInBytes = m_pgMat->cols * sizeof(uint8_t);
				//cuMemcpy2D(pCopy);

				size_t pitch;
				cudaArray *u_dev = (cudaArray*)data->cuArray;
				cudaError err = cudaMallocPitch((void**)&u_dev, &pitch, pgMat->cols * sizeof(uint8_t) * 3 /*channels*/, pgMat->rows);
				if (err != cudaSuccess)
					return tenStatus::nenError;

				err = cudaMemcpy2D(u_dev, pitch, pgMat->data, pgMat->step, pgMat->cols * sizeof(uint8_t) * 3 /*channels*/, pgMat->rows, cudaMemcpyDeviceToDevice);
				if (err != cudaSuccess)
					return tenStatus::nenError;

				data->cuArray = (CUarray)u_dev;

				if (bInternalGpuMat)
				{
					delete pgMat;
					pgMat = nullptr;
				}

				if (enStatus != tenStatus::nenSuccess || !data->cuArray)
				{
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