#include <stdint.h>

__global__ void kernel444ToNV12_Y(uint8_t *pArrayFull, uint8_t *pNV12_Y, int iWidth, int iHeight, int iPitchSrc, int iPitchDest)
{
	int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
	int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
	int src = iPosY * iPitchSrc + iPosX * 3;
	int dest = iPosY * iPitchDest + iPosX;
	pNV12_Y[dest] = pArrayFull[src]; // Y
}

__global__ void kernel444ToNV12_UV(uint8_t *pArrayFull, uint8_t *pNV12_UV, int iWidth, int iHeight, int iPitchSrc, int iPitchDest)
{
	int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
	int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
	int src = iPosY * 2 * iPitchSrc + iPosX * 2 * 3;
	int dest = iPosY * iPitchDest + iPosX * 2;
    pNV12_UV[dest] = pArrayFull[src + 1]; // U
	pNV12_UV[dest + 1] = pArrayFull[src + 2]; // V
}

extern "C" void IMP_444ToNV12(uint8_t *pArrayFull, uint8_t *pArrayNV12, int iWidth, int iHeight, size_t iPitchSrc, size_t iPitchDest)
{
	dim3 block(64, 8, 1);

	dim3 grid_Y(iWidth / block.x, iHeight / block.y, 1);
	kernel444ToNV12_Y << <grid_Y, block >> >(pArrayFull, pArrayNV12, iWidth, iHeight, (int)iPitchSrc, (int)iPitchDest);

	dim3 grid_UV(iWidth / 2 / block.x, iHeight / 2 / block.y, 1);
	kernel444ToNV12_UV << <grid_UV, block >> >(pArrayFull, pArrayNV12 + iHeight * iPitchDest, iWidth / 2, iHeight / 2, (int)iPitchSrc, (int)iPitchDest);
}
