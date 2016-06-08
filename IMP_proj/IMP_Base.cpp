#include "IMP_Base.h"
#include "cuda_runtime_api.h"
#include "opencv2/cudev/common.hpp"


namespace RW{
	namespace IMP{

		tenStatus IMP_Base::tensProcessInput(cInputBase *pInput, cOutputBase *pOutput)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			if (!pInput)
            {
				m_Logger->error("IMP_Base::tensProcessInput: pInput is empty!");
				return tenStatus::nenError;
            }
            if (!pOutput)
            {
                m_Logger->error("IMP_Base::tensProcessInput: pOutput is empty!");
                return tenStatus::nenError;
            }
		
			if (pInput->_bImportImg)
			{
				if (!pInput->_pvImg)
				{
					m_Logger->error("IMP_Base::tensProcessInput: Empty Input data!");
					enStatus = tenStatus::nenError;
				}
				void* pvImg = pInput->_pvImg;

				int iWidth = pInput->_iWidth;
				int iHeight = pInput->_iHeight;
				if (iWidth == 0 || iHeight == 0)
				{
					m_Logger->error("IMP_Base::tensProcessInput: Invalid parameters (iWidth or iHeight)");
					enStatus = tenStatus::nenError;
				}

				//Some Output data has to be set from outside!
				if (pInput->_bImportImg && pOutput->_bOutputGPU)
				{
					m_pgMat = new cv::cuda::GpuMat();
					m_bInternalGpuMat = true;
				}
				else if (pOutput->_pgMat)
				{
					m_pgMat = pOutput->_pgMat;
				}
				else
				{
					m_Logger->error("IMP_Base::tensProcessInput: No GpuMat available!");
					enStatus = tenStatus::nenError;
				}
				enStatus = tensConvertArrayToGpuMat(iWidth, iHeight, pvImg);
			}
			else
			{
				m_pgMat = (cv::cuda::GpuMat*)pInput->_pgMat;
			}

			if (!m_pgMat)
            {
                m_Logger->error("IMP_Base::tensProcessInput: m_pgMat is empty!");
				enStatus = tenStatus::nenError;
            }
			return enStatus;
		}

		tenStatus IMP_Base::tensConvertArrayToGpuMat(int iWidth, int iHeight, void *pvImg)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

			if (pvImg)
			{
                cv::cuda::GpuMat gMat = cv::cuda::GpuMat(iHeight, iWidth, CV_8UC3, pvImg, cv::Mat::AUTO_STEP);
                
                *m_pgMat = gMat;
            }
			else
			{
                m_Logger->error("IMP_Base::tensConvertArrayToGpuMat: pvImg is empty!");
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
                m_Logger->error("IMP_Base::tensProcessOutput: pOutput is empty!");
                return tenStatus::nenError;
            }

            if (pOutput->_bOutputGPU)
            {
                tenStatus enStatus = tenStatus::nenSuccess;
                if (m_pgMat == nullptr)
                {
                    m_Logger->error("IMP_Base::tensProcessOutput: m_pgMat is empty!");
                    return tenStatus::nenError;
                }
                pOutput->_pcuArray = (CUarray*)m_pgMat;

                //size_t sArraySize = m_pgMat->cols *  m_pgMat->rows * sizeof(uint8_t);
                //cuMemcpyHtoA(*pOutput->_pcuArray, 0, m_pgMat->data, sArraySize);

                //CUDA_MEMCPY2D* pCopy = new CUDA_MEMCPY2D();
                //pCopy->srcHost = m_pgMat->data;
                //pCopy->srcMemoryType = CU_MEMORYTYPE_HOST;
                //pCopy->dstArray = *pOutput->_pcuArray;
                //pCopy->dstMemoryType = CU_MEMORYTYPE_ARRAY;
                //pCopy->Height = m_pgMat->rows;
                //pCopy->WidthInBytes = m_pgMat->cols * sizeof(uint8_t);
                //cuMemcpy2D(pCopy);

                //cudaMemcpy2DToArray(pOutput->_pcuArray, 0, 0, m_pgMat->data, m_pgMat->step * sizeof(size_t), m_pgMat->cols*sizeof(int), m_pgMat->rows*sizeof(int), cudaMemcpyDeviceToDevice);

                if (m_bInternalGpuMat)
                {
                    delete m_pgMat;
                    m_pgMat = nullptr;
                    m_bInternalGpuMat = false;
                }
                if (!pOutput->_pcuArray)
                {
                    m_Logger->error("IMP_Base::tensProcessOutput: Converting to CUarray failed!");
                    enStatus = tenStatus::nenError;
                }
                return enStatus;
            }
            else
            {
                pOutput->_pgMat = m_pgMat;
                return tenStatus::nenSuccess;
            }
        }
	}
}