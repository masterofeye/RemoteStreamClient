#pragma once

#include "IMP.h"

namespace RW
{
	namespace IMP
	{
		class IMP_Base
		{
		public:
			void vSetGpuMat(cv::cuda::GpuMat cuGpuMat){ m_cuGpuMat = cuGpuMat; };
			cv::cuda::GpuMat cuGetGpuMat(){ return m_cuGpuMat; };

			tenStatus Initialise(cInputBase *pInput);
			tenStatus Deinitialise(cOutputBase *pOutput);

			IMP_Base(){};
			~IMP_Base(){};

		private:
			cv::cuda::GpuMat m_cuGpuMat;

			tenStatus tensConvertArrayToGpuMat(int iWidth, int iHeight, void *pvImg, cv::cuda::GpuMat *pgMat);

		};
	}
}