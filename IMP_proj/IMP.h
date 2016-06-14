#pragma once

#include "stdint.h"
#include "opencv2/opencv.hpp"

#include <exception>
#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"
#include "..\ENC_proj\common\inc\dynlink_cuda.h"

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif



namespace RW
{
    namespace IMP
    {
		class cOutputBase{
		public:
			cOutputBase()
			{
				_pgMat = nullptr;
				_bExportImg = false;
			};
			cOutputBase(CUarray cuArray[3], bool bExportImg = false)
			{
				_cuArray[0] = cuArray[0];
				_cuArray[1] = cuArray[1];
				_cuArray[2] = cuArray[2];

				_pgMat = nullptr;
				_bExportImg = bExportImg;
			};
			cOutputBase(cv::cuda::GpuMat *pgMat, bool bExportImg = false)
			{
				_pgMat = pgMat;
				_bExportImg = bExportImg;
			};

			~cOutputBase(){}

			bool _bExportImg;
			cv::cuda::GpuMat *_pgMat;
			CUarray _cuArray[3];
		};

		class cInputBase{
		public:
			typedef struct stImportImg
			{
				void *pvImg;
				int iWidth;
				int iHeight;

				stImportImg(int width, int height, void* data)
				{
					iWidth = width;
					iHeight = height;
					pvImg = data;
				}
				stImportImg()
				{
					pvImg = nullptr;
				}
			}tstImportImg;


			cInputBase()
			{
				_pgMat = nullptr;
				_bImportImg = false;
			};
			cInputBase(tstImportImg stImg, bool bImportImg = false)
			{
				_stImg = stImg;
				_pgMat = nullptr;
				_bImportImg = bImportImg;
			};
			cInputBase(cv::cuda::GpuMat *pgMat, bool bImportImg = false)
			{
				_pgMat = pgMat;
				_bImportImg = bImportImg;
			};

			~cInputBase(){}

			tstImportImg _stImg;
			cv::cuda::GpuMat *_pgMat;
			bool _bImportImg;
		};
	}
}