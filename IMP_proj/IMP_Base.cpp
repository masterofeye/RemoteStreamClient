#include "IMP_Base.h"
#include <cuda_runtime.h>
#include "opencv2/cudev/common.hpp"

// CUDA D3D9 kernel
extern "C" void IMP_CopyTo420(uint8_t *pArrayFull, uint8_t *pArrayY, uint8_t *pArrayU, uint8_t *pArrayV, int width, int height);

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
				if (!pOutput->_cuArrayY || !pOutput->_cuArrayUV)
				{
					m_Logger->error("IMP_Base::tensProcessOutput: Empty cuda array!");
					enStatus = tenStatus::nenError;
				}

				int width = pgMat->cols;
				int height = pgMat->rows;

				/* Check if pgMat is okay */
				cv::Mat mat;
				pgMat->download(mat);
				cv::Mat yuv[3];
				cv::split(mat, yuv);
				cv::imwrite("C:\\dummy\\yuv444_0.png", yuv[0]);
				cv::imwrite("C:\\dummy\\yuv444_1.png", yuv[1]);
				cv::imwrite("C:\\dummy\\yuv444_2.png", yuv[2]);
				/* Check done */


				/* Check for previous errors */
				cudaError err;
				err = cudaPeekAtLastError();
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaDeviceSynchronize();
				if (err != cudaSuccess) return tenStatus::nenError;
				/* Check done */

				/* A better solution would be to do cudaMemcpy2dToArray */
				cudaArray *pCudaArrayY;
				cudaArray *pCudaArrayUV[2];
				pCudaArrayY = (cudaArray *)pOutput->_cuArrayY;
				pCudaArrayUV[0] = (cudaArray *)pOutput->_cuArrayUV[0];
				pCudaArrayUV[1] = (cudaArray *)pOutput->_cuArrayUV[1];

				size_t pitchY, pitchU, pitchV;
				uint8_t* arrayY;
				uint8_t* arrayU;
				uint8_t* arrayV;
				err = cudaMallocPitch((void**)&arrayY, &pitchY, width * sizeof(uint8_t), height);
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaMallocPitch((void**)&arrayU, &pitchU, width * sizeof(uint8_t) / 2, height / 2);
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaMallocPitch((void**)&arrayV, &pitchV, width * sizeof(uint8_t) / 2, height / 2);
				if (err != cudaSuccess) return tenStatus::nenError;

				//err = cudaMemcpy(pCudaArrayY, pgMat->data, width * height, cudaMemcpyDeviceToDevice);
				//err = cudaMemcpy2D(arrayY, pitchY, pgMat->data, pitchY, width * sizeof(uint8_t), height, cudaMemcpyDeviceToDevice);
				//if (err != cudaSuccess) return tenStatus::nenError;

				IMP_CopyTo420((uint8_t *)pgMat->data, arrayY, arrayU, arrayV, width, height);

				//err = cudaMemcpy(pCudaArrayY, arrayY,width * sizeof(uint8_t) * height, cudaMemcpyDeviceToDevice);
				//if (err != cudaSuccess) return tenStatus::nenError;
				//err = cudaMemcpy(pCudaArrayUV[0], arrayU, width * sizeof(uint8_t) / 2 * height / 2, cudaMemcpyDeviceToDevice);
				//if (err != cudaSuccess) return tenStatus::nenError;
				//err = cudaMemcpy(pCudaArrayUV[1], arrayV, width * sizeof(uint8_t) / 2 * height / 2, cudaMemcpyDeviceToDevice);
				//if (err != cudaSuccess) return tenStatus::nenError;

				/* Check if cudaArrays are okay */
				cv::Mat matDummyY(height, width, CV_8U);
				cv::Mat matDummyU(height / 2, width / 2, CV_8U);
				cv::Mat matDummyV(height / 2, width / 2, CV_8U);

				err = cudaMemcpy2D(matDummyY.data, width, arrayY, pitchY, width * sizeof(uint8_t), height, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaMemcpy2D(matDummyU.data, width / 2, arrayU, pitchU, width * sizeof(uint8_t) / 2, height / 2, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaMemcpy2D(matDummyV.data, width / 2, arrayV, pitchV, width * sizeof(uint8_t) / 2, height / 2, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) return tenStatus::nenError;

				cv::imwrite("C:\\dummy\\yuv420_0.png", matDummyY);
				cv::imwrite("C:\\dummy\\yuv420_1.png", matDummyU);
				cv::imwrite("C:\\dummy\\yuv420_2.png", matDummyV);
				/* Check done */

				cudaFree(arrayY);
				cudaFree(arrayU);
				cudaFree(arrayV);

				pOutput->_cuArrayY = (CUarray)pCudaArrayY;
				pOutput->_cuArrayUV[0] = (CUarray)pCudaArrayUV[0];
				pOutput->_cuArrayUV[1] = (CUarray)pCudaArrayUV[1];

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