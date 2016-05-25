#include "IMP_Base.h"
#include "cuda_runtime_api.h"


namespace RW{
	namespace IMP{

		tenStatus IMP_Base::tensProcessInput(cInputBase *pInput)
		{
            if (pInput == nullptr)
            {
                return tenStatus::nenError;
            }

            if (pInput->_pstParams)
            {
                tenStatus enStatus = tenStatus::nenSuccess;
                int iWidth = pInput->_pstParams->iWidth;
                int iHeight = pInput->_pstParams->iWidth;
                void* pvImg = pInput->_pstParams->pvImg;

                if (iWidth == 0 || iHeight == 0)
                {
                    enStatus = tenStatus::nenError;
                }

                if (pvImg)
                {
                    enStatus = tensConvertArrayToGpuMat(iWidth, iHeight, pvImg);
                }
                else
                {
                    enStatus = tenStatus::nenError;
                }
                return enStatus;
            }
            else if (pInput->_pgMat)
            {
                m_pgMat = pInput->_pgMat;
                return tenStatus::nenSuccess;
            }
            else
            {
                return tenStatus::nenError;
            }
		}

		tenStatus IMP_Base::tensConvertArrayToGpuMat(int iWidth, int iHeight, void *pvImg)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

			if (pvImg)
			{
                m_pgMat = new cv::cuda::GpuMat(iHeight, iWidth, /*CV_8UC3*/ 16, pvImg, cv::Mat::AUTO_STEP);

                delete pvImg;
                pvImg = nullptr;

                if (m_pgMat == nullptr)
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
            if (pOutput == nullptr)
            {
                return tenStatus::nenError;
            }

            if (pOutput->_pcuArray)
            {
                tenStatus enStatus = tenStatus::nenSuccess;
                if (m_pgMat == nullptr)
                {
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }
                cudaError cuErr = cudaMemcpy2DToArray(pOutput->_pcuArray, 0, 0, m_pgMat->data, m_pgMat->step * sizeof(size_t), m_pgMat->cols*sizeof(int), m_pgMat->rows*sizeof(int), cudaMemcpyDeviceToDevice);

                delete m_pgMat;
                m_pgMat = nullptr;

                if (pOutput->_pcuArray == nullptr || cuErr != cudaSuccess)
                {
                    enStatus = tenStatus::nenError;
                }
                return enStatus;
            }
            else if (pOutput->_pgMat)
            {
                pOutput->_pgMat = m_pgMat;
                return tenStatus::nenSuccess;
            }
            else
            {
                return tenStatus::nenError;;
            }
        }
	}
}