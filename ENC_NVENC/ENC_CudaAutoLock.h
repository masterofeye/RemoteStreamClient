#pragma once

#include "NvHWEncoder.h"

namespace RW{
	namespace ENC{
		class ENC_CudaAutoLock
		{
			friend class ENC_CudaInterop;

		private:
			CUcontext m_pCtx;
		public:
			ENC_CudaAutoLock(CUcontext pCtx) :m_pCtx(pCtx) { cuCtxPushCurrent(m_pCtx); };
			~ENC_CudaAutoLock()  { CUcontext cuLast = nullptr; cuCtxPopCurrent(&cuLast); };
		};
	}
}