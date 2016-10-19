#include "IMP_Base.h"
#include <cuda_runtime.h>
#include "opencv2/cudev/common.hpp"

// CUDA kernel
extern "C" void IMP_444ToNV12(uint8_t *pArrayFull, uint8_t *pArrayNV12, int iWidth, int iHeight, size_t iPitchSrc, size_t iPitchDest);

namespace RW{
	namespace IMP{
        tenStatus IMP_Base::GpuMatToGpuNV12(cv::cuda::GpuMat *pgMat, CUdeviceptr *pOutput)
        {
			tenStatus enStatus = tenStatus::nenSuccess;

			int iWidth = pgMat->cols;
			int iHeight = pgMat->rows;

			size_t sPitchOut;
			cudaError err = cudaMallocPitch((void**)pOutput, &sPitchOut, iWidth, iHeight * 3 / 2);
			if (err != cudaSuccess)
			{
				printf("IMP_Base::GpuMatToCpuNV12: cudaMallocPitch failed!");
				return tenStatus::nenError;
			}

			IMP_444ToNV12((uint8_t*)pgMat->data, (uint8_t*)*pOutput, iWidth, iHeight, pgMat->step, sPitchOut);

			err = cudaDeviceSynchronize();
			if (err)
			{
				printf("IMP_444ToNV12: Device synchronize failed! Error = %d\n", err);
				return tenStatus::nenError;
			}
			err = cudaGetLastError();
			if (err)
			{
				printf("IMP_444ToNV12: kernel() failed to launch error = %d\n", err);
				return tenStatus::nenError;
			}

#ifdef TEST
			size_t sSize = (size_t)(pgMat->cols * pgMat->rows * 3 / 2);
			uint8_t *pBuffer = new uint8_t[sSize];
			err = cudaMemcpy2D(pBuffer, pgMat->cols, (void*)*pOutput, sPitchOut, pgMat->cols, pgMat->rows * 3 / 2, cudaMemcpyDeviceToHost);
			static int count;
			WriteBufferToFile(pBuffer, sSize, "Server_GpuMatToGpuNV12", count);
			delete[] pBuffer;
#endif

			return enStatus;
		}
        tenStatus IMP_Base::GpuMatToCpuNV12(cv::cuda::GpuMat *pgMat, uint8_t *pOutput)
        {
            tenStatus enStatus = tenStatus::nenSuccess;
            if (!pOutput)
            {
                printf("IMP_Base::tensProcessOutput: pOutput is empty!");
                return tenStatus::nenError;
            }

            CUdeviceptr arrYUV;
            size_t pitch;

            int iWidth = pgMat->cols;
            int iHeight = pgMat->rows * 3 / 2;

            cudaError err = cudaMallocPitch((void**)&arrYUV, &pitch, iWidth, iHeight);
            if (err != cudaSuccess)
            {
                printf("IMP_Base::GpuMatToCpuNV12: cudaMallocPitch failed!");
                return tenStatus::nenError;
            }

            GpuMatToGpuNV12(pgMat, &arrYUV);

            cudaMemcpy2D(pOutput, iWidth, (void*)arrYUV, pitch, iWidth, iHeight, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                printf("IMP_Base::GpuMatToCpuNV12: cudaMemcpy2D failed!");
                return tenStatus::nenError;
            }

            cudaFree((void*)arrYUV);

            return enStatus;
        }
	}
}