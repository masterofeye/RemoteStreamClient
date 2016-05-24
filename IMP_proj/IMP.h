#pragma once

#include "stdint.h"
#include "opencv2/opencv.hpp"
#include "opencv2/cudev/common.hpp"
#include <exception>
#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"

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

        typedef struct stInputParams
        {
            int iWidth;
            int iHeight;
            void *pvImg;
        }tstInputParams;

        class cInputBase{
        public:
            cInputBase()
            {
                _pstParams = NULL;
                _pgMat = NULL;
            };
            cInputBase(tstInputParams stInput){ *_pstParams = stInput; };
            cInputBase(cv::cuda::GpuMat gMat){ *_pgMat = gMat; };
            cInputBase(cInputBase oInput1, cInputBase oInput2){ *_pInput1 = oInput1; *_pInput2 = oInput2; }
            tstInputParams *_pstParams;
            cv::cuda::GpuMat *_pgMat;
            cInputBase *_pInput1;
            cInputBase *_pInput2;
            bool _bNeedConversion;
        };

        class cOutputBase{
        public:
            cOutputBase()
            {
                _pgMat = NULL;
                _pcuArray = NULL;
            };
            cOutputBase(cv::cuda::GpuMat *pgMat){ _pgMat = pgMat; };
            cOutputBase(cudaArray *pcuArray){ _pcuArray = pcuArray; };
            cv::cuda::GpuMat *_pgMat;
            cudaArray *_pcuArray;
        };

        typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
        {
            stRectStruct stFrameRect;
        }tstMyInitialiseControlStruct;

        typedef struct stMyControlStruct : public CORE::tstControlStruct
        {
            cInputBase cInput;
            cOutputBase cOutput;
        }tstMyControlStruct;

        typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
        {
        }tstMyDeinitialiseControlStruct;
    }
}