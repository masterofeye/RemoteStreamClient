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
    //              |        y Pos        | + |   x Pos    | + |z|
    int iPos444 = iPosY * iDepth * iWidth + iDepth * iPosX +  0; 

    if (iPosX < iWidth)
    {
        pArrayFull[iPos444] = pYUV420[iPos420];
        //int val = pYUV420Array[0];

        if (iPosY < iHeight / 2)
        {
            if (iPosX % 2 == 0){
                pArrayFull[(2 * iPosY) * iDepth * iWidth + iDepth * iPosX + 1] = pYUV420[iOffset + iPos420];
                pArrayFull[(2 * iPosY) * iDepth * iWidth + iDepth * (iPosX + 1) + 1] = pYUV420[iOffset + iPos420];
                pArrayFull[((2 * iPosY) + 1) * iDepth * iWidth + iDepth * iPosX + 1] = pYUV420[iOffset + iPos420];
                pArrayFull[((2 * iPosY) + 1) * iDepth * iWidth + iDepth * (iPosX + 1) + 1] = pYUV420[iOffset + iPos420];
            }
            else 
            {
                pArrayFull[(2 * iPosY) * iDepth * iWidth + iDepth * iPosX + 2] = pYUV420[iOffset + iPos420];
                pArrayFull[(2 * iPosY) * iDepth * iWidth + iDepth * (iPosX + 1) + 2] = pYUV420[iOffset + iPos420];
                pArrayFull[((2 * iPosY) + 1) * iDepth * iWidth + iDepth * iPosX + 2] = pYUV420[iOffset + iPos420];
                pArrayFull[((2 * iPosY) + 1) * iDepth * iWidth + iDepth * (iPosX + 1) + 2] = pYUV420[iOffset + iPos420];
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