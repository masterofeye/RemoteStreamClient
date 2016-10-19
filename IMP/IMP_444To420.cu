#include <stdint.h>

__global__ void kernel444To420(uint8_t *pArrayFull, uint8_t *pYUV420, int iWidth, int iHeight, int iPitchSrc, int iPitchDest)
{
	int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
	int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
	int iPosIn = iPosY * iPitchSrc + 3 * iPosX;
	int iPos = iPosY * iPitchDest + iPosX;

    pYUV420[iPos] = pArrayFull[iPosIn];

    if ((iPosX % 2 == 0) && (iPosY % 2 == 0))
    {
		int iOffsetU = iPitchDest * iHeight;
		int iOffsetV = iPitchDest / 4 * iHeight * 5;
		int iPosX_UV = (iPosX / 2);
		int iPosY_UV = (iPosY / 2) / 2;
		int iPosUV = iPosY_UV * iPitchDest + iPosX_UV + ((iPosY / 2) % 2) * iWidth / 2;
		pYUV420[iOffsetU + iPosUV] = pArrayFull[iPosIn + 1];
        pYUV420[iOffsetV + iPosUV] = pArrayFull[iPosIn + 2];
    }
}

extern "C" void IMP_444To420(uint8_t *pArrayFull, uint8_t *pArrayYUV420, int iWidth, int iHeight, size_t iPitchSrc, size_t iPitchDest)
{
	dim3 block(16, 16, 1);
	dim3 grid(iWidth / block.x, iHeight / block.y, 1);

    //plane to plane
	kernel444To420 << <grid, block >> >(pArrayFull, pArrayYUV420, iWidth, iHeight, (int)iPitchSrc, (int)iPitchDest);
}
