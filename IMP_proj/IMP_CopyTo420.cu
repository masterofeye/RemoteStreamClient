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
        int iOffsetU = iWidth * iHeight;
        int iOffsetV = iWidth * iHeight * 5 / 4;
        int iPosX_UV = (iPosY / 4 * iWidth) + (iPosX / 2);
        if (iPosY % 4 == 0) {
            iPosX += (iWidth / 2);
        }

        pYUV420[iOffsetU + iPosX_UV] = pArrayFull[iPosIn + 1];
        pYUV420[iOffsetV + iPosX_UV] = pArrayFull[iPosIn + 2];

    }
}

extern "C" void IMP_CopyTo420(uint8_t *pArrayFull, uint8_t *pArrayYUV420, int iWidth, int iHeight, size_t pitchY)
{
	cudaError_t error = cudaSuccess;

	dim3 block(32, 16, 1);
	dim3 grid(iWidth / block.x, iHeight / block.y, 1);
    kernel << <grid, block >> >(pArrayFull, pArrayYUV420, (int)pitchY, iHeight);
}

#endif