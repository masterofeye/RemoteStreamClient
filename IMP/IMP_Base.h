#pragma once

#include "IMP.h"

namespace RW
{
	namespace IMP
	{
		class IMP_Base
		{
		public:
            tenStatus GpuMatToGpuNV12(cv::cuda::GpuMat *pgMat, CUdeviceptr *pOutput);
            tenStatus GpuMatToCpuNV12(cv::cuda::GpuMat *pgMat, uint8_t *pOutput);

            IMP_Base()
            {
            };
			~IMP_Base()
            {
            };
        private:
            size_t m_sPitch;
		};
	}
}