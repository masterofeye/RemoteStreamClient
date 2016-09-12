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
		class tstInputOutput{
        public:
            tstInputOutput(){};
            ~tstInputOutput(){};

            RW::tstBitStream *pBitstream;
			cv::cuda::GpuMat *pgMat;
            CUdeviceptr cuDevice;

        };
	}
}