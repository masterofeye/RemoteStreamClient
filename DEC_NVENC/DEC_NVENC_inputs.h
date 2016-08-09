#pragma once

// CUDA Header includes
#include "dynlink_cuviddec.h"  // <nvcuvid.h>


#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

namespace RW
{
	namespace DEC
	{
		typedef struct stInputParams
		{
			cudaVideoCreateFlags eVideoCreateFlags;
			bool bUseVsync;
			bool bFrameRepeat;
			int  iRepeatFactor;
			bool bFrameStep;
			bool bUseDisplay;
			bool bUseInterop;
			unsigned int fpsLimit;
            unsigned int nVideoWidth;
            unsigned int nVideoHeight;

            stInputParams() : eVideoCreateFlags(cudaVideoCreateFlags_enum::cudaVideoCreate_PreferCUDA), bFrameRepeat(false), bUseVsync(false), bFrameStep(false), bUseDisplay(true), bUseInterop(true), nVideoHeight(0), nVideoWidth(0) {}
			~stInputParams(){}
		}tstInputParams;
	}
}