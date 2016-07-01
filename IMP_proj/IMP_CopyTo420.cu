#ifndef IMP_COPYTO420_CU
#define IMP_COPYTO420_CU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


__global__ void kernel(uint8_t *pArrayFull, uint8_t *pYUV420, int iHeight, int iPitch)
{
	int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
	int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
    int iPos = iPosY * iPitch + iPosX;

    int iPosIn = iPosY * 3 * iPitch + 3 * iPosX;

	pYUV420[iPos] = pArrayFull[iPosIn];

    if ((iPosX % 2 == 0) && (iPosY % 2 == 0))
    {
        int iOffsetU = iPitch * iHeight;
        int iOffsetV = iPitch * iHeight * 5 / 4;
        int iPosX_UV = (iPosX / 2);
        if (iPosY % 4 == 0) {
            iPosX_UV += (iPosY / 4 * iPitch);
        }
        else {
            iPosX_UV += ((iPosY - 2) / 4 * iPitch) + (iPitch / 2);
        }

        pYUV420[iOffsetU + iPosX_UV] = pArrayFull[iPosIn + 1];
        pYUV420[iOffsetV + iPosX_UV] = pArrayFull[iPosIn + 2];

    }
}

extern "C" void IMP_CopyTo420(uint8_t *pArrayFull, uint8_t *pArrayYUV420, int iWidth, int iHeight, size_t pitchY)
{
	dim3 block(32, 16, 1);
	dim3 grid(iWidth / block.x, iHeight / block.y, 1);
    kernel << <grid, block >> >(pArrayFull, pArrayYUV420, iHeight, (int)pitchY);
}

#endif