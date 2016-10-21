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
        namespace NVENC
        {
            typedef struct stInputParams
            {
                cudaVideoCreateFlags eVideoCreateFlags;
                unsigned int nWidth;
                unsigned int nHeight;
                cudaVideoCodec codec;

                stInputParams() : eVideoCreateFlags(cudaVideoCreateFlags_enum::cudaVideoCreate_PreferCUDA), nHeight(0), nWidth(0), codec(cudaVideoCodec_enum::cudaVideoCodec_H264) {}
                ~stInputParams(){}
            }tstInputParams;
        }
	}
}