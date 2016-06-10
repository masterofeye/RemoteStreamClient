#include "IMP_Base.h"
#include "cuda_runtime_api.h"
#include "opencv2/cudev/common.hpp"


namespace RW{
	namespace IMP{

		tenStatus IMP_Base::tensProcessInput(cInputBase *pInput, cv::cuda::GpuMat *pOutput)
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
			m_pgMat = pOutput;

			if (pInput->_bImportImg)
			{
				if (!pInput->_stImg.pvImg)
				{
					m_Logger->error("IMP_Base::tensProcessInput: Empty Input data!");
					enStatus = tenStatus::nenError;
				}
				void* pvImg = pInput->_stImg.pvImg;

				int iWidth = pInput->_stImg.iWidth;
				int iHeight = pInput->_stImg.iHeight;
				if (iWidth == 0 || iHeight == 0)
				{
					m_Logger->error("IMP_Base::tensProcessInput: Invalid parameters (iWidth or iHeight)");
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
                cv::Mat mat = cv::Mat(iHeight, iWidth, CV_8UC3, pvImg);
				cv::imshow("test", mat);
				m_pgMat->upload(mat);
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

	}
}