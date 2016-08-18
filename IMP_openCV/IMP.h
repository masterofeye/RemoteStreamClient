#pragma once

#include "stdint.h"
#include "opencv2/opencv.hpp"

#include <exception>
#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"
#include "..\Common_NVENC\inc\dynlink_cuda.h"

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
			cOutputBase(CUdeviceptr pcuYUV420, bool bExportImg = false)
			{
				_pcuYUV420 = pcuYUV420;
				_pgMat = nullptr;
				_bExportImg = bExportImg;
			}
			//cOutputBase(CUarray cuArrayY, CUarray cuArrayU, CUarray cuArrayV, bool bExportImg = false)
			//{
			//	_cuArrayY = cuArrayY;
			//	_cuArrayUV[0] = cuArrayU;
			//	_cuArrayUV[1] = cuArrayV;

			//	_pgMat = nullptr;
			//	_bExportImg = bExportImg;
			//};
			cOutputBase(cv::cuda::GpuMat *pgMat, bool bExportImg = false)
			{
				_pgMat = pgMat;
				_bExportImg = bExportImg;
			};

			~cOutputBase(){}

			CUdeviceptr _pcuYUV420;
			bool _bExportImg;
			cv::cuda::GpuMat *_pgMat;
			//CUarray _cuArrayUV[2];
			//CUarray _cuArrayY;
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