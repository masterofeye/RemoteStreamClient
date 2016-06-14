#pragma once

#include "IMP.h"

namespace RW
{
	namespace IMP
	{
		class IMP_Base
		{
		public:
            tenStatus tensProcessInput(cInputBase *pInput, cv::cuda::GpuMat *pgMat);
			tenStatus tensProcessOutput(cv::cuda::GpuMat *pgMat, cOutputBase *pOutput);

            IMP_Base(std::shared_ptr<spdlog::logger> Logger) : m_Logger(Logger)
            {
            };
			~IMP_Base()
            {
            };

		private:
            std::shared_ptr<spdlog::logger> m_Logger;
		};
	}
}