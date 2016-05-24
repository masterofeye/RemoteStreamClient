#include "IMP_Base.h"
#include "cuda_runtime_api.h"


namespace RW{
	namespace IMP{

		tenStatus IMP_Base::tensProcessInput(cInputBase *pInput)
		{
            if (pInput->_pstParams != NULL)
            {
                tenStatus enStatus = tenStatus::nenSuccess;
                int iWidth = pInput->_pstParams->iWidth;
                int iHeight = pInput->_pstParams->iWidth;
                void* pvImg = pInput->_pstParams->pvImg;

                if (iWidth == 0 || iHeight == 0)
                {
                    enStatus = tenStatus::nenError;
                }

                if (pvImg != NULL)
                {
                    enStatus = tensConvertArrayToGpuMat(iWidth, iHeight, pvImg, &m_cuGpuMat);
                }
                else
                {
                    enStatus = tenStatus::nenError;
                }
                return enStatus;
            }
            else if (pInput->_pgMat != NULL)
            {
                m_cuGpuMat = *pInput->_pgMat;
                return tenStatus::nenSuccess;
            }
            else
            {
                return tenStatus::nenError;
            }
		}

		tenStatus IMP_Base::tensConvertArrayToGpuMat(int iWidth, int iHeight, void *pvImg, cv::cuda::GpuMat *pgMat)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

			if (pvImg != NULL)
			{
				pgMat = new cv::cuda::GpuMat(iHeight, iWidth, /*CV_8UC3*/ 16, pvImg, cv::Mat::AUTO_STEP);
				if (pgMat == NULL)
				{
					enStatus = tenStatus::nenError;
				}
			}
			else
			{
				enStatus = tenStatus::nenError;
			}

			//cudaArray *cuArr;
			//checkCudaErrors(cudaMemcpy2DFromArray(pgMat.data, pgMat.step * sizeof(uint8_t), cuArr, 0, 0, pgMat.cols*sizeof(uint8_t), pgMat.rows, cudaMemcpyDeviceToDevice));
			return enStatus;
		}

        tenStatus IMP_Base::tensProcessOutput(cOutputBase *pOutput)
		{
            if (pOutput->_pcuArray != NULL)
            {
                tenStatus enStatus = tenStatus::nenSuccess;
                cudaArray *pcuArr = NULL;
                cudaError cuErr = cudaMemcpy2DToArray(pcuArr, 0, 0, m_cuGpuMat.data, m_cuGpuMat.step * sizeof(size_t), m_cuGpuMat.cols*sizeof(int), m_cuGpuMat.rows*sizeof(int), cudaMemcpyDeviceToDevice);
                if (pcuArr == NULL || cuErr != cudaSuccess)
                {
                    enStatus = tenStatus::nenError;
                }
                else
                {
                    pOutput->_pcuArray = pcuArr;
                }
                return enStatus;
            }
            else if (pOutput->_pgMat != NULL)
            {
                *pOutput->_pgMat = m_cuGpuMat;
                return tenStatus::nenSuccess;
            }
            else
            {
                return tenStatus::nenError;;
            }
        }
	}
}