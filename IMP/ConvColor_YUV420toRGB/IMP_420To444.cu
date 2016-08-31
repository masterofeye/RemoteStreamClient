#ifndef IMP_420To444_CU
#define IMP_420To444_CU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


__global__ void kernel420To444(uint8_t *pYUV420Array, uint8_t *pArrayFull, int iWidth, int iHeight, int iPitch)
{
    int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
    int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
    int iPos420 = iPosY * iPitch + iPosX;
    int iOffset = iHeight * iPitch;
    int iDepth = 3;

    if (iPosX < iWidth)
    {
        //            |        y Pos        | + |   x Pos    | + |z|
        int iY = iPosY * iDepth * iWidth;
        int iX = iPosX * iDepth;
        int iPos444 = iY + iX + 0;

        pArrayFull[iPos444] = pYUV420Array[iPos420];

        if (iPosY < iHeight / 2)
        {
            int iY1 = 2 * iPosY * iDepth * iWidth;
            int iY2 = ((2 * iPosY) + 1) * iDepth * iWidth;

            if (iPosX % 2 == 0){
                int iX1 = iPosX * iDepth;
                int iX2 = (iPosX + 1) * iDepth;

                pArrayFull[iY + iX + 1] = pYUV420Array[iOffset + iPos420];
                //pArrayFull[iY1 + iX2 + 1] = pYUV420Array[iOffset + iPos420];
                //pArrayFull[iY2 + iX1 + 1] = pYUV420Array[iOffset + iPos420];
                //pArrayFull[iY2 + iX2 + 1] = pYUV420Array[iOffset + iPos420];
            }
            else {
                int iX1 = (iPosX - 1) * iDepth;
                int iX2 = iPosX * iDepth;

                pArrayFull[iY + iX + 2] = pYUV420Array[iOffset + iPos420];
                //pArrayFull[iY1 + iX2 + 2] = pYUV420Array[iOffset + iPos420];
                //pArrayFull[iY2 + iX1 + 2] = pYUV420Array[iOffset + iPos420];
                //pArrayFull[iY2 + iX2 + 2] = pYUV420Array[iOffset + iPos420];
            }
        }
    }
}

extern "C" void IMP_420To444(uint8_t *pYUV420Array, uint8_t *pArrayFull, int iWidth, int iHeight, size_t pitchY)
{
    dim3 block(32, 16, 1);
    dim3 grid(pitchY / block.x, iHeight / block.y, 1);

    //interleaved to plane
    kernel420To444 << <grid, block >> >(pYUV420Array, pArrayFull, iWidth, iHeight, (int)pitchY);
}

#endif