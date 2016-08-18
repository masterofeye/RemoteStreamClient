#ifndef IMP_420To444_CU
#define IMP_420To444_CU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


__global__ void kernel420To444(uint8_t *pYUV420, uint8_t *pArrayFull, int iHeight, int iPitch)
{
    int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
    int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
    int iPos = iPosY * iPitch + iPosX;
    int iOffset = iHeight * iPitch;

    if (iPos < 0)
    {
        return;
    }
    else if (iPos < iHeight * iPitch) // Y channel
    {
        pArrayFull[iPos] = pYUV420[iPos];
    }
    else if (iPos > iHeight * iPitch)
    {
        if (iPosX % 2 == 0) // U channel
        {
            pArrayFull[iOffset + (iPosY - iHeight) * iPitch + iPosX];
            pArrayFull[iOffset + (iPosY - iHeight) * iPitch + iPosX*2];
            pArrayFull[iOffset + (iPosY - iHeight)*2 * iPitch + iPosX];
            pArrayFull[iOffset + (iPosY - iHeight)*2 * iPitch + iPosX * 2];
        }
        else // V channel
        {
            pArrayFull[2 * iOffset + (iPosY - iHeight) * iPitch + (iPosX - 1)];
            pArrayFull[2 * iOffset + (iPosY - iHeight) * iPitch + (iPosX - 1) * 2];
            pArrayFull[2 * iOffset + (iPosY - iHeight) * 2 * iPitch + (iPosX - 1)];
            pArrayFull[2 * iOffset + (iPosY - iHeight) * 2 * iPitch + (iPosX - 1) * 2];
        }
    }
}

extern "C" void IMP_420To444(uint8_t *pArrayYUV420, uint8_t *pArrayFull, int iWidth, int iHeight, size_t pitchY)
{
    dim3 block(32, 16, 1);
    dim3 grid(iWidth / block.x, iHeight / block.y, 1);

    //interleaved to plane
    kernel420To444 << <grid, block >> >(pArrayYUV420, pArrayFull, iHeight, (int)iWidth);
}

#endif