#ifndef IMP_420To444_CU
#define IMP_420To444_CU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


__global__ void kernel420To444(uint8_t *pYUV420, uint8_t *pArrayFull, int iWidth, int iHeight, int iPitch)
{
    int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
    int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
    int iPos = iPosY * iPitch + iPosX;
    int iOffset = iHeight * iPitch;

    if (iPos < 0)
    {
        return;
    }
    else if (iPos < iHeight * iPitch)
    {
        if (iPosX < iWidth)
        {
            pArrayFull[iPos] = pYUV420[iPos];
        }
        if (iPosY < iHeight / 2)
        {
            if (iPosX < iWidth / 2){
                pArrayFull[iOffset + (4 * iPosY) * iPitch + 2 * iPosX] = pYUV420[iOffset + iPos];
                pArrayFull[iOffset + (4 * iPosY) * iPitch + 2 * iPosX + 1] = pYUV420[iOffset + iPos];
                pArrayFull[iOffset + (4 * iPosY + 1) * iPitch + 2 * iPosX] = pYUV420[iOffset + iPos];
                pArrayFull[iOffset + (4 * iPosY + 1) * iPitch + 2 * iPosX + 1] = pYUV420[iOffset + iPos];
            }
            else if ((iPosX > iPitch / 2) && (iPosX < (iPitch - (iPitch/2 - iWidth/2))))
            {
                pArrayFull[iOffset + (4 * iPosY + 2) * iPitch + 2 * iPosX] = pYUV420[iOffset + iPos];
                pArrayFull[iOffset + (4 * iPosY + 2) * iPitch + 2 * iPosX + 1] = pYUV420[iOffset + iPos];
                pArrayFull[iOffset + (4 * iPosY + 3) * iPitch + 2 * iPosX] = pYUV420[iOffset + iPos];
                pArrayFull[iOffset + (4 * iPosY + 3) * iPitch + 2 * iPosX + 1] = pYUV420[iOffset + iPos];
            }
        }
    }
}

extern "C" void IMP_420To444(uint8_t *pArrayYUV420, uint8_t *pArrayFull, int iWidth, int iHeight, size_t pitchY)
{
    dim3 block(32, 16, 1);
    dim3 grid(pitchY / block.x, iHeight / block.y, 1);

    //interleaved to plane
    kernel420To444 << <grid, block >> >(pArrayYUV420, pArrayFull, iWidth, iHeight, (int)pitchY);
}

#endif