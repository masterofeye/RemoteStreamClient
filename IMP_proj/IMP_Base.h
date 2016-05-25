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

            tenStatus tensProcessInput(cInputBase *pInput);
            tenStatus tensProcessOutput(cOutputBase *pOutput);

			IMP_Base()
            {
                m_pgMat = nullptr;
            };
			~IMP_Base()
            {
                if (m_pgMat)
                {
                    delete m_pgMat;
                    m_pgMat = nullptr;
                }
            };

		private:
            cv::cuda::GpuMat *m_pgMat;

			tenStatus tensConvertArrayToGpuMat(int iWidth, int iHeight, void *pvImg);

		};
	}
}