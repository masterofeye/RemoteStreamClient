#ifndef IMP_COPYTO420_CU
#define IMP_COPYTO420_CU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


__global__ void kernel(uint8_t *pArrayFull, uint8_t *pYUV420, int iWidth, int iHeight)
{
	int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
	int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
	int iPos = iPosY * iWidth + iPosX;

	int iPosIn = iPosY * 3 * iWidth + 3 * iPosX;

	pYUV420[iPos] = pArrayFull[iPosIn];

	if ((iPosX % 2 == 0) && (iPosY % 2 == 0))
	{
		int iPosUV = iWidth * iHeight + iPosY / 2 * iWidth + iPosX;
		pYUV420[iPosUV] = pArrayFull[iPosIn + 1];
		pYUV420[iPosUV + 1] = pArrayFull[iPosIn + 2];
	}
}

extern "C" void IMP_CopyTo420(uint8_t *pArrayFull, uint8_t *pArrayYUV420, int iWidth, int iHeight, size_t pitchY)
{
	cudaError_t error = cudaSuccess;

	dim3 block(24, 16, 1);
	dim3 grid(iWidth / block.x, iHeight / block.y, 1);
	kernel<<<grid, block>>>(pArrayFull, pArrayYUV420, (int)pitchY, iHeight);
	
	error = cudaGetLastError();   
	if (error != cudaSuccess)
	{
        printf("IMP_CopyTo420: kernel() failed to launch error = %d\n", error);
        return;
	}
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        printf("IMP_CopyTo420: Device synchronize failed! Error = %d\n", error);
        return;
    }
}

#endif