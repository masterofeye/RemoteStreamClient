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
                    m_Logger->error("IMP_Base::tensProcessInput: Invalid parameters (iWidth or iHeight)");
                    enStatus = tenStatus::nenError;
                }

                if (pvImg)
                {
                    enStatus = tensConvertArrayToGpuMat(iWidth, iHeight, pvImg);
                }
                else
                {
                    m_Logger->error("IMP_Base::tensProcessInput: pvImg is empty!");
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
                m_Logger->error("IMP_Base::tensProcessInput: _pgMat is empty!");
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
                    m_Logger->error("IMP_Base::tensConvertArrayToGpuMat: m_pgMat is empty!");
                    enStatus = tenStatus::nenError;
                }
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

            if (pOutput->_pcuArray)
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

                delete m_pgMat;
                m_pgMat = nullptr;

                if (pOutput->_pcuArray == nullptr)
                {
                    m_Logger->error("IMP_Base::tensProcessOutput: cudaMemcpy2DToArray(...) did not succeed!");
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
                m_Logger->error("IMP_Base::tensProcessOutput: No valid data to process!");
                return tenStatus::nenError;;
            }
        }
	}
}