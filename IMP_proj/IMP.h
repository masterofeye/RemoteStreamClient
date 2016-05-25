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

            stInputParams() : iWidth(0), iHeight(0), pvImg(nullptr){}
            ~stInputParams() 
            {
                if (pvImg)
                {
                    delete pvImg;
                    pvImg = nullptr;
                }
            }
        }tstInputParams;

        class cInputBase{
        public:
            cInputBase()
            {
                _pstParams = nullptr;
                _pgMat = nullptr;
            };
            cInputBase(tstInputParams *pstInput)
            { 
                _pstParams = pstInput; 
                _pgMat = nullptr;
            };
            cInputBase(cv::cuda::GpuMat *pgMat)
            { 
                _pstParams = nullptr;
                _pgMat = pgMat;
            };
            cInputBase(cInputBase *poInput1, cInputBase *poInput2)
            { 
                _pInput1 = poInput1; 
                _pInput2 = poInput2; 
            }
            ~cInputBase()
            {
                if (_pstParams)
                {
                    delete _pstParams;
                    _pstParams = nullptr;
                }
                if (_pgMat)
                {
                    delete _pgMat;
                    _pgMat = nullptr;
                }
                if (_pInput1)
                {
                    delete _pInput1;
                    _pInput1 = nullptr;
                }
                if (_pInput2)
                {
                    delete _pInput2;
                    _pInput2 = nullptr;
                }
            }

            tstInputParams *_pstParams;
            cv::cuda::GpuMat *_pgMat;
            cInputBase *_pInput1;
            cInputBase *_pInput2;
        };

        class cOutputBase{
        public:
            cOutputBase()
            {
                _pgMat = nullptr;
                _pcuArray = nullptr;
            };
            cOutputBase(cv::cuda::GpuMat *pgMat)
            {
                _pgMat = pgMat;
                _pcuArray = nullptr;
            };
            cOutputBase(cudaArray *pcuArray)
            {
                _pgMat = nullptr;
                _pcuArray = pcuArray;
            };
            ~cOutputBase()
            {
                if (_pgMat)
                {
                    delete _pgMat;
                    _pgMat = nullptr;
                }
                if (_pcuArray)
                {
                    delete _pcuArray;
                    _pcuArray = nullptr;
                }
            }
            cv::cuda::GpuMat *_pgMat;
            cudaArray *_pcuArray;
        };

        typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
        {
            stRectStruct *pstFrameRect;
        }tstMyInitialiseControlStruct;

        typedef struct stMyControlStruct : public CORE::tstControlStruct
        {
            cInputBase *pcInput;
            cOutputBase *pcOutput;
        }tstMyControlStruct;

        typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
        {
        }tstMyDeinitialiseControlStruct;
    }
}