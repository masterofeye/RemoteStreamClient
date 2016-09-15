#include "IMP_Base.h"
#include <cuda_runtime.h>
#include "opencv2/cudev/common.hpp"

// CUDA kernel
extern "C" void IMP_444To420(uint8_t *pArrayFull, uint8_t *pArrayYUV420, int iWidth, int iHeight, size_t pitchY);
extern "C" void IMP_444ToNV12(uint8_t *pArrayFull, uint8_t *pArrayYUV420, int iWidth, int iHeight, size_t pitchY);

namespace RW{
	namespace IMP{

        tenStatus IMP_Base::GpuMatToGpuYUV(cv::cuda::GpuMat *pgMat, CUdeviceptr *pOutput)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

			int iWidth = pgMat->cols;
			int iHeight = pgMat->rows;

            cudaError err = cudaMallocPitch((void**)pOutput, &m_sPitch, iWidth, iHeight * 3 / 2);
            if (err != cudaSuccess)
            {
                printf("IMP_Base::GpuMatToCpuYUV: cudaMallocPitch failed!");
                return tenStatus::nenError;
            }
            CUdeviceptr cuInput;
            err = cudaMallocPitch((void**)&cuInput, &m_sPitch, iWidth, iHeight * 3);
            if (err != cudaSuccess)
            {
                printf("IMP_Base::GpuMatToCpuYUV: cudaMallocPitch failed!");
                return tenStatus::nenError;
            }
            err = cudaMemcpy2D((void*)cuInput, m_sPitch, pgMat->data, iWidth, iWidth, iHeight * 3, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                printf("IMP_Base::GpuMatToCpuYUV: cudaMemcpy2D failed!");
                return tenStatus::nenError;
            }


            IMP_444To420((uint8_t*)cuInput, (uint8_t*)*pOutput, iWidth, iHeight, m_sPitch);

            err = cudaDeviceSynchronize();
            if (err)
            {
                printf("IMP_444To420: Device synchronize failed! Error = %d\n", err);
                return tenStatus::nenError;
            }
            err = cudaGetLastError();
            if (err)
            {
                printf("IMP_444To420: kernel() failed to launch error = %d\n", err);
                return tenStatus::nenError;
            }
            cudaFree((void*)cuInput);

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

            int iWidth = pgMat->cols;
            int iHeight = pgMat->rows * 3 / 2;

            enStatus = GpuMatToGpuYUV(pgMat, &arrYUV);
            if (enStatus != tenStatus::nenSuccess)
            {
                printf("IMP_Base::GpuMatToCpuYUV: GpuMatToGpuYUV failed!");
                return tenStatus::nenError;
            }

            cudaError err = cudaMemcpy2D(pOutput, iWidth, (void*)arrYUV, m_sPitch, iWidth, iHeight, cudaMemcpyDeviceToHost);
            if (err)
            {
                printf("IMP_Base::GpuMatToCpuYUV: cudaMemcpy2D failed!");
                return tenStatus::nenError;
            }

            cudaFree((void*)arrYUV);

            return enStatus;
        }

        tenStatus IMP_Base::GpuMatToGpuNV12(cv::cuda::GpuMat *pgMat, CUdeviceptr *pOutput)
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

            IMP_444ToNV12(pgMat->data, arrayYUV420, iWidth, iHeight, pitchY);
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

            //cv::Mat mat(iHeight * 3 / 2, iWidth, CV_8U);
            //err = cudaMemcpy2D(mat.data, iWidth, arrayYUV420, pitchY, iWidth, iHeight * 3 / 2, cudaMemcpyDeviceToHost);
            //if (err != cudaSuccess) return tenStatus::nenError;

            cudaFree(arrayY);

            *pOutput = (CUdeviceptr)arrayYUV420;

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