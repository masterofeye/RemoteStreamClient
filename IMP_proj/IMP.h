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
            cInputBase(cInputBase oInput1, cInputBase oInput2){ *_pInput1 = oInput1; *_pInput2 = oInput2; }
            tstInputParams _stParams;
            cv::cuda::GpuMat _gMat;
            cInputBase *_pInput1;
            cInputBase *_pInput2;
        };

        class cOutputBase{
        public:
            cOutputBase(){};
            cOutputBase(cv::cuda::GpuMat *pgMat){ _pgMat = pgMat; };
            cOutputBase(cudaArray *pcuArray){ _pcuArray = pcuArray; };
            cv::cuda::GpuMat *_pgMat;
            cudaArray *_pcuArray;
        };

        typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
        {
            bool bNeedConversion;
            cInputBase cInput;
        }tstMyInitialiseControlStruct;

        typedef struct stMyControlStruct : public CORE::tstControlStruct
        {
            stRectStruct stFrameRect;
        }tstMyControlStruct;

        typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
        {
            bool bNeedConversion;
            cOutputBase cOutput;
        }tstMyDeinitialiseControlStruct;
    }
}