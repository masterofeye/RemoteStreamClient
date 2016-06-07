#pragma once

#include "IMP.h"

namespace RW
{
	namespace IMP
	{
		class IMP_Base
		{
		public:
			void vSetGpuMat(cv::cuda::GpuMat *pgMat){ m_pgMat = pgMat; };
            cv::cuda::GpuMat* cuGetGpuMat(){ return m_pgMat; };

            tenStatus tensProcessInput(cInputBase *pInput, cOutputBase *pOutput);
            tenStatus tensProcessOutput(cOutputBase *pOutput);

            IMP_Base(std::shared_ptr<spdlog::logger> Logger) : m_Logger(Logger)
            {
                m_bInternalGpuMat = false;
            };
			~IMP_Base()
            {
            };

		private:
            cv::cuda::GpuMat *m_pgMat;
            std::shared_ptr<spdlog::logger> m_Logger;

            bool m_bInternalGpuMat;  //we have an internal GpuMat if it was not created outside. Which means an void* pixel array is input and an CUarray is output. 

			tenStatus tensConvertArrayToGpuMat(int iWidth, int iHeight, void *pvImg);

		};
	}
}