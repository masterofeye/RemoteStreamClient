#pragma once

#include "IMP.h"

namespace RW
{
	namespace IMP
	{
		class IMP_Base
		{
		public:
            cv::cuda::GpuMat* cuGetGpuMat(){ return m_pgMat; };

            tenStatus tensProcessInput(cInputBase *pInput, cv::cuda::GpuMat *pOutput);

            IMP_Base(std::shared_ptr<spdlog::logger> Logger) : m_Logger(Logger)
            {
            };
			~IMP_Base()
            {
            };

		private:
            cv::cuda::GpuMat *m_pgMat;
            std::shared_ptr<spdlog::logger> m_Logger;

			tenStatus tensConvertArrayToGpuMat(int iWidth, int iHeight, void *pvImg);

		};
	}
}