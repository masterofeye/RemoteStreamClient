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