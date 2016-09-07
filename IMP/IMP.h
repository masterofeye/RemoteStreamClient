#pragma once

#include "stdint.h"
#include "opencv2/opencv.hpp"

#include <exception>
#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"
#include "dynlink_cuda.h"

#include "..\OpenVXWrapperTest\Pipeline_Config.h"

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
            stBitStream *_pBitstream;
		};

		class cInputBase{
		public:
			cInputBase()
			{
				_pgMat = nullptr;
                _pBitstream = nullptr;
				_bImportImg = false;
			};
			cInputBase(tstBitStream *pBitstream, bool bImportImg = false)
			{
				_pBitstream = pBitstream;
				_pgMat = nullptr;
				_bImportImg = bImportImg;
			};
			cInputBase(cv::cuda::GpuMat *pgMat, bool bImportImg = false)
			{
				_pgMat = pgMat;
				_bImportImg = bImportImg;
			};

			~cInputBase(){}

            tstBitStream *_pBitstream;
			cv::cuda::GpuMat *_pgMat;
            CUdeviceptr _cuDevice;
			bool _bImportImg;
		};
	}
}