#pragma once

#include "IMP.h"

namespace RW
{
	namespace IMP
	{
		class IMP_Base
		{
		public:
            tenStatus GpuMatToGpuYUV(cv::cuda::GpuMat *pgMat, CUdeviceptr *pOutput);
            tenStatus GpuMatToCpuYUV(cv::cuda::GpuMat *pgMat, uint8_t *pOutput);

            IMP_Base()
            {
            };
			~IMP_Base()
            {
            };

		};
	}
}