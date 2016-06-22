#include "IMP_Base.h"
#include <cuda_runtime.h>
#include "opencv2/cudev/common.hpp"

// CUDA D3D9 kernel
extern "C" void IMP_CopyTo420(uint8_t *pArrayFull, uint8_t *pArrayYUV420, int iWidth, int iHeight, size_t pitchY);

namespace RW{
	namespace IMP{

		tenStatus IMP_Base::tensProcessOutput(cv::cuda::GpuMat *pgMat, cOutputBase *pOutput)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			if (!pOutput)
			{
				m_Logger->error("IMP_Base::tensProcessOutput: pOutput is empty!");
				return tenStatus::nenError;
			}

			if (pOutput->_bExportImg)
			{
				if (!pOutput->_pcuYUV420)
				{
					m_Logger->error("IMP_Base::tensProcessOutput: Empty cuda array!");
					enStatus = tenStatus::nenError;
				}

				int iWidth = pgMat->cols;
				int iHeight = pgMat->rows;

				/* Check for previous errors */
				cudaError err;
				err = cudaPeekAtLastError();
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaDeviceSynchronize();
				if (err != cudaSuccess) return tenStatus::nenError;
				/* Check done */

				size_t pitchY;
				uint8_t* arrayY;
				uint8_t* arrayYUV420 = (uint8_t *)pOutput->_pcuYUV420;

				err = cudaMallocPitch((void**)&arrayY, &pitchY, iWidth, 1);
				if (err != cudaSuccess) return tenStatus::nenError;

				IMP_CopyTo420(pgMat->data, arrayYUV420, iWidth, iHeight, pitchY);

				cudaFree(arrayY);

				pOutput->_pcuYUV420 = (CUdeviceptr)arrayYUV420;
			}
			else
			{
				if (!pgMat)
				{
					m_Logger->error("IMP_Base::tensProcessInput: pgMat is empty!");
					enStatus = tenStatus::nenError;
				}
				pOutput->_pgMat = pgMat;
			}

			return enStatus;
		}

		tenStatus IMP_Base::tensProcessInput(cInputBase *pInput, cv::cuda::GpuMat *pgMat)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			if (!pInput)
            {
				m_Logger->error("IMP_Base::tensProcessInput: pInput is empty!");
				return tenStatus::nenError;
            }
			if (!pgMat)
            {
                m_Logger->error("IMP_Base::tensProcessInput: pOutput is empty!");
                return tenStatus::nenError;
            }

			if (pInput->_bImportImg)
			{
				if (!pInput->_stImg.pvImg)
				{
					m_Logger->error("IMP_Base::tensProcessInput: Empty Input data!");
					enStatus = tenStatus::nenError;
				}
				if (pInput->_stImg.iWidth == 0 || pInput->_stImg.iHeight == 0)
				{
					m_Logger->error("IMP_Base::tensProcessInput: Invalid parameters (iWidth or iHeight)");
					enStatus = tenStatus::nenError;
				}

				if (pInput->_stImg.pvImg)
				{
					cv::Mat mat = cv::Mat(pInput->_stImg.iHeight, pInput->_stImg.iWidth, CV_8UC3, pInput->_stImg.pvImg);
					pgMat->upload(mat);
				}
			}
			else
			{
				pgMat = (cv::cuda::GpuMat*)pInput->_pgMat;
			}

			if (!pgMat)
            {
                m_Logger->error("IMP_Base::tensProcessInput: pgMat is empty!");
				enStatus = tenStatus::nenError;
            }
			return enStatus;
		}
	}
}