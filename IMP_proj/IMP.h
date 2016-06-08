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
        typedef struct stRectStruct{
            int iPosX;
            int iPosY;
            int iWidth;
            int iHeight;
        }tstRectStruct;

        class cInputBase{
        public:
            cInputBase()
            {
				_pgMat = nullptr;
				_pvImg = nullptr;
				_bSetImg2 = false;
				_pInput1 = nullptr;
				_pInput2 = nullptr;
            };
            cInputBase(int iWidth, int iHeight, void *pvImg)
            { 
				_pvImg = pvImg;
				_pgMat = nullptr;
				_iWidth = iWidth;
				_iHeight = iHeight;
			};
			cInputBase(int iWidth, int iHeight, cv::cuda::GpuMat *pgMat)
            { 
                _pgMat = pgMat;
				_pvImg = nullptr;
				_iWidth = iWidth;
				_iHeight = iHeight;
			};
			cInputBase(cInputBase *poInput1, cInputBase *poInput2, bool bSetImg2)
            { 
				_bSetImg2 = bSetImg2;
				_pInput1 = poInput1;
				_pInput2 = poInput2;
            }
            ~cInputBase(){}

			bool _bSetImg2;
            void *_pvImg;
            cv::cuda::GpuMat *_pgMat;
            cInputBase *_pInput1;
            cInputBase *_pInput2;
			int _iWidth;
			int _iHeight; 
        };

        class cOutputBase{
        public:
            cOutputBase()
            {
            };
            cOutputBase(cv::cuda::GpuMat *pgMat)
            {
                _pgMat = pgMat;
            };
            cOutputBase(CUarray *pcuArray)
            {
                _pcuArray = pcuArray;
            };
			~cOutputBase(){}

			cv::cuda::GpuMat *_pgMat;
            CUarray *_pcuArray;
        };

		typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
		{
			stRectStruct *pstFrameRect;
		}tstMyInitialiseControlStruct;

        typedef struct REMOTE_API stMyControlStruct : public CORE::tstControlStruct
		{
			cInputBase *pcInput;
			cOutputBase *pcOutput;
            void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);

		}tstMyControlStruct;

		typedef struct stCropDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
		{
		}tstMyDeinitialiseControlStruct;

	}
}