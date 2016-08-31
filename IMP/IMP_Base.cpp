#include "IMP_Base.h"
#include <cuda_runtime.h>
#include "opencv2/cudev/common.hpp"

// CUDA kernel
extern "C" void IMP_444To420(uint8_t *pArrayFull, uint8_t *pArrayYUV420, int iWidth, int iHeight, size_t pitchY);

namespace RW{
	namespace IMP{

        tenStatus IMP_Base::GpuMatToGpuYUV(cv::cuda::GpuMat *pgMat, CUdeviceptr *pOutput)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			if (!pOutput)
			{
                printf("IMP_Base::GpuMatToGpuYUV: Output is empty!");
				return tenStatus::nenError;
			}

			int iWidth = pgMat->cols;
			int iHeight = pgMat->rows;

			size_t pitchY;
			uint8_t* arrayY;
			uint8_t* arrayYUV420 = (uint8_t *)*pOutput;

            cudaError err = cudaMallocPitch((void**)&arrayY, &pitchY, iWidth, 1);
			if (err != cudaSuccess) return tenStatus::nenError;

            IMP_444To420(pgMat->data, arrayYUV420, iWidth, iHeight, pitchY);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("IMP_444To420: kernel() failed to launch error = %d\n", err);
                return tenStatus::nenError;
            }
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("IMP_444To420: Device synchronize failed! Error = %d\n", err);
                return tenStatus::nenError;
            }

            cudaFree(arrayY);

            *pOutput = (CUdeviceptr)arrayYUV420;

			return enStatus;
		}
        tenStatus IMP_Base::GpuMatToCpuYUV(cv::cuda::GpuMat *pgMat, uint8_t *pOutput)
        {
            tenStatus enStatus = tenStatus::nenSuccess;
            if (!pOutput)
            {
                printf("IMP_Base::tensProcessOutput: pOutput is empty!");
                return tenStatus::nenError;
            }

            CUdeviceptr arrYUV;
            size_t pitch;

            cudaError err = cudaMallocPitch((void**)&arrYUV, &pitch, pgMat->cols, pgMat->rows * 3 / 2);

            GpuMatToGpuYUV(pgMat, &arrYUV);

            int iWidth = pgMat->cols;
            int iHeight = pgMat->rows * 3/2;

            cudaMemcpy2D(pOutput, iWidth, (void*)arrYUV, pitch, iWidth, iHeight, cudaMemcpyDeviceToHost);

            cudaFree((void*)arrYUV);

            return enStatus;
        }

	}
}