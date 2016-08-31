#pragma once

#include "stdint.h"
#include "opencv2/opencv.hpp"

#include <exception>
#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"
#include "dynlink_cuda.h"

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
                _pu8Array = nullptr;
			};
			cOutputBase(CUdeviceptr pcuYUV420)
			{
				_pcuYUV420 = pcuYUV420;
				_pgMat = nullptr;
                _pu8Array = nullptr;
            }
            cOutputBase(uint8_t *pu8Array)
            {
                _pu8Array = pu8Array;
                _pgMat = nullptr;
            }
			cOutputBase(cv::cuda::GpuMat *pgMat)
			{
				_pgMat = pgMat;
                _pu8Array = nullptr;
			};

			~cOutputBase(){}

			CUdeviceptr _pcuYUV420;
			bool _bExportImg;
			cv::cuda::GpuMat *_pgMat;
            uint8_t *_pu8Array;
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