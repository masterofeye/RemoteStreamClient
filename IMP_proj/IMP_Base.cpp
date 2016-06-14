#include "IMP_Base.h"
#include "cuda_runtime_api.h"
#include "opencv2/cudev/common.hpp"


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
				if (!pOutput->_cuArray)
				{
					m_Logger->error("IMP_Base::tensProcessOutput: Empty cuda array!");
					enStatus = tenStatus::nenError;
				}

				/* Check if pgMat is okay */
				cv::Mat matDummy;
				pgMat->download(matDummy);
				/* Check done */

				size_t sSize = pgMat->cols * pgMat->rows * pgMat->elemSize();

				/* Check for previous errors */
				cudaError err;
				err = cudaPeekAtLastError();
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaDeviceSynchronize();
				if (err != cudaSuccess) return tenStatus::nenError;
				/* Check done */

				/* Ugly solution: Copying into dummy array and doing memcopyArrayToArray later */
				//cudaArray *u_dev = nullptr;
				//err = cudaMalloc((void**)&u_dev, sSize / 2);
				//if (err != cudaSuccess) return tenStatus::nenError;
				//err = cudaMemset(u_dev, 0, sSize / 2);
				//if (err != cudaSuccess) return tenStatus::nenError;

				/* A better solution would be to do cudaMemcpy2dToArray */
				cudaArray *pCudaArray[3];
				pCudaArray[0] = (cudaArray *)pOutput->_cuArray[0];
				pCudaArray[1] = (cudaArray *)pOutput->_cuArray[1];
				pCudaArray[2] = (cudaArray *)pOutput->_cuArray[2];

				//size_t pitch;
				size_t pitchY;
				size_t pitchU;
				size_t pitchV;
				//err = cudaMallocPitch((void**)&u_dev, &pitch, pgMat->cols * pgMat->elemSize(), pgMat->rows); 
				err = cudaMallocPitch((void**)&pCudaArray[0], &pitchY, pgMat->cols * pgMat->elemSize(), pgMat->rows);
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaMallocPitch((void**)&pCudaArray[1], &pitchU, pgMat->cols * pgMat->elemSize() / 4, pgMat->rows);
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaMallocPitch((void**)&pCudaArray[2], &pitchU, pgMat->cols * pgMat->elemSize() / 4, pgMat->rows);
				if (err != cudaSuccess) return tenStatus::nenError;

				/* Before doing cudaMemcpy the cudaMemset and cudaMallocPitch need to be performed */
				//err = cudaMemcpy(u_dev, pgMat->data, sSize, cudaMemcpyDeviceToDevice);
				err = cudaMemcpy2DToArray(pCudaArray[0], 0, 0, pgMat->data, pitchY, pgMat->cols * pgMat->elemSize(), pgMat->rows, cudaMemcpyDeviceToDevice);
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaMemcpy2DToArray(pCudaArray[1], 0, 0, pgMat->data, pitchU + pitchY, pgMat->cols * pgMat->elemSize() / 4, pgMat->rows, cudaMemcpyDeviceToDevice);
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaMemcpy2DToArray(pCudaArray[2], 0, 0, pgMat->data, pitchV + pitchU + pitchY, pgMat->cols * pgMat->elemSize() / 4, pgMat->rows, cudaMemcpyDeviceToDevice);
				if (err != cudaSuccess) return tenStatus::nenError;

				/* Check if cudaMemcpy was successfull */
				//uint8_t *pDummy;
				//pDummy = (uint8_t *)malloc(sSize);
				//memset(pDummy, 0, sSize);
				///* If you copy from Gpu to Cpu or from Cpu to Gpu be considerate of the different pitch sizes. */
				//err = cudaMemcpy2D(pDummy, pgMat->cols * pgMat->elemSize(), u_dev, pitch, pgMat->cols * pgMat->elemSize(), pgMat->rows, cudaMemcpyDeviceToHost);
				////err = cudaMemcpy(pDummy, u_dev, sSize, cudaMemcpyDeviceToHost);
				//if (err != cudaSuccess) return tenStatus::nenError;

				//cv::Mat mat2(pgMat->rows, pgMat->cols, pgMat->type(), pDummy);
				//free(pDummy);
				/* Check done. If you use cudaMemcpy there are several rows missing at the end! Idk why. So better use cudaMemcpy2D. */
				//size_t sSize = pgMat->cols * pgMat->rows * sizeof(uint8_t);

				//err = cudaMemcpy(pDummy, Dummy, sSize * pgMat->elemSize(), cudaMemcpyDeviceToHost);
				//cv::Mat matDummy2(pgMat->rows, pgMat->cols, pgMat->type(), pDummy);


				/* Where the heck did the other data end up? YUV is [1:1/4:1/4]. 
				The matrix is subsampled. I checked, the values are not continuously written like:

				xxxxxx
				xxxxxx
				xxxxxx
				xxxxxx
				yyyyyy
				zzzzzz
				000000
				000000
				000000
				000000
				000000
				000000

				See https://en.wikipedia.org/wiki/Chroma_subsampling#4:1:1
				Maybe we need to write a cuda kernel to access the U and V channels directly. 
				Or we download for the prototype the data to cpu to extract those channels. */
				/* I did so according function loadframe in example NvEncoderCuderInterop and confirmed that data should be assembled like that by doing internet research */
				//err = cudaMemcpy2DArrayToArray(pCudaArray[0], 0, 0, u_dev, 0, 0, pgMat->cols * pgMat->elemSize(), pgMat->rows, cudaMemcpyDeviceToDevice);
				//err = cudaMemcpyArrayToArray(pCudaArray[0], 0, 0, u_dev, 0, 0, sSize, cudaMemcpyDeviceToDevice);
				//if (err != cudaSuccess) return tenStatus::nenError;
				//err = cudaMemcpyArrayToArray(pCudaArray[1], 0, 0, u_dev, 0, sSize, sSize / 4, cudaMemcpyDeviceToDevice);
				//if (err != cudaSuccess) return tenStatus::nenError;
				//err = cudaMemcpyArrayToArray(pCudaArray[2], 0, 0, u_dev, 0, sSize + sSize / 4, sSize / 4, cudaMemcpyDeviceToDevice);
				//if (err != cudaSuccess) return tenStatus::nenError;

				//err = cudaMemcpyToArray(pCudaArray[0], 0, 0, pgMat[0].data, sSize, cudaMemcpyDeviceToDevice);
				//if (err != cudaSuccess) return tenStatus::nenError;
				//err = cudaMemcpyToArray(pCudaArray[1], 0, 0, pgMat[1].data, sSize/4, cudaMemcpyDeviceToDevice);
				//if (err != cudaSuccess) return tenStatus::nenError;
				//err = cudaMemcpyToArray(pCudaArray[2], 0, 0, pgMat[2].data, sSize/4, cudaMemcpyDeviceToDevice);
				//if (err != cudaSuccess) return tenStatus::nenError;

				//err = cuMemcpyAtoA(arr[0], 0, cuArray, 0, sSize);
				//if (err != CUDA_SUCCESS) return NV_ENC_ERR_UNSUPPORTED_PARAM;
				//err = cuMemcpyAtoA(arr[1], 0, cuArray, sSize, sSize / 4);
				//if (err != CUDA_SUCCESS) return NV_ENC_ERR_UNSUPPORTED_PARAM;
				//err = cuMemcpyAtoA(arr[2], 0, cuArray, sSize + sSize / 4, sSize / 4);
				//if (err != CUDA_SUCCESS) return NV_ENC_ERR_UNSUPPORTED_PARAM;

				pOutput->_cuArray[0] = (CUarray)pCudaArray[0];
				pOutput->_cuArray[1] = (CUarray)pCudaArray[1];
				pOutput->_cuArray[2] = (CUarray)pCudaArray[2];



				/****************************************************************************************
				cudaError err;
				err = cudaPeekAtLastError();
				if (err != cudaSuccess) return tenStatus::nenError;
				err = cudaDeviceSynchronize();
				if (err != cudaSuccess) return tenStatus::nenError;

				size_t pitch;
				if (!data->cuArray) return tenStatus::nenError;
				cudaArray *u_dev = (cudaArray*)data->cuArray;
				err = cudaMallocPitch((void**)&u_dev, &pitch, pgMat->cols * sizeof(uint8_t) * 3 , pgMat->rows);
				if (err != cudaSuccess)
					return tenStatus::nenError;

				err = cudaMemcpyToArray(u_dev, 0, 0, pgMat->data, sSize, cudaMemcpyDeviceToDevice);
				//err = cudaMemcpy2D(u_dev, pitch, pgMat->data, pgMat->step, pgMat->cols * sizeof(uint8_t) * 3 , pgMat->rows, cudaMemcpyDeviceToDevice);
				//err = cudaMemcpy(u_dev, pgMat->data, sSize, cudaMemcpyDeviceToDevice);
				if (err != cudaSuccess) return tenStatus::nenError;

				uint8_t *pDummy = new uint8_t[sSize];

				//err = cudaMemcpy2D(pgMat->data, pitch, u_dev, pgMat->step, pgMat->cols * sizeof(uint8_t) * 3 , pgMat->rows, cudaMemcpyDeviceToHost);
				err = cudaMemcpyFromArray(pDummy, u_dev, 0, 0, sSize, cudaMemcpyDeviceToHost);
				//err = cudaMemcpy(pDummy, u_dev, sSize, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) return tenStatus::nenError;


				//Mat(int rows, int cols, int type, void* data, size_t step = AUTO_STEP);
				cv::Mat mat(pgMat->rows, pgMat->cols, pgMat->type(), pDummy);

				data->cuArray = (CUarray)u_dev; 
				********************************************************************************************/

				//cudaFree(u_dev);

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