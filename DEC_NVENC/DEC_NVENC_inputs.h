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

			stInputParams() : eVideoCreateFlags(cudaVideoCreateFlags_enum::cudaVideoCreate_PreferCUVID), bFrameRepeat(false), bUseVsync(false), bFrameStep(false), bUseDisplay(true), bUseInterop(true) {}
			~stInputParams(){}
		}tstInputParams;
	}
}