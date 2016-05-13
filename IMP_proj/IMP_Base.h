#pragma once

#include "stdint.h"
#include "opencv2/opencv.hpp"
#include "opencv2/cudev/common.hpp"
#include <exception>
#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"

namespace RW
{
	namespace IMP
	{
		typedef struct stRectStruct{
			int iPosX;
			int iPosY;
			int iWidth;
			int iHeight;
		}tstRectStruct;

		typedef struct stInputParams
		{
			int iWidth;
			int iHeight;
			void *pvImg;
		}tstInputParams;

		class cInputBase{
		public:
			cInputBase(){};
			cInputBase(tstInputParams stInput){ _stParams = stInput; };
			cInputBase(cv::cuda::GpuMat gMat){ _gMat = gMat; };
			tstInputParams _stParams;
			cv::cuda::GpuMat _gMat;
		};

		class cOutputBase{
		public:
			cOutputBase(){};
			cOutputBase(cv::cuda::GpuMat *pgMat){ _pgMat = pgMat; };
			cOutputBase(cudaArray *pcuArray){ _pcuArray = pcuArray; };
			cv::cuda::GpuMat *_pgMat;
			cudaArray *_pcuArray;
		};

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