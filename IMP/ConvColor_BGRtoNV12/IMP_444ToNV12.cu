#include <stdint.h>

__global__ void kernel444ToNV12_Y(const uint8_t *pArrayFull, uint8_t *pNV12_Y, int iWidth, int iHeight, int iPitchSrc, int iPitchDest)
{
	int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
	int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
	if (iPosX >= iWidth || iPosY >= iHeight)
		return;
	int src = iPosY * iPitchSrc + iPosX * 3;
	int dest = iPosY * iPitchDest + iPosX;
	pNV12_Y[dest] = pArrayFull[src]; // Y
}

__global__ void kernel444ToNV12_UV(const uint8_t *pArrayFull, uint8_t *pNV12_UV, int iWidth, int iHeight, int iPitchSrc, int iPitchDest)
{
	int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
	int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
	if (iPosX >= iWidth || iPosY >= iHeight)
		return;
	int src = iPosY * 2 * iPitchSrc + iPosX * 2 * 3;
	int dest = iPosY * iPitchDest + iPosX * 2;
    pNV12_UV[dest] = pArrayFull[src + 1]; // U
	pNV12_UV[dest + 1] = pArrayFull[src + 2]; // V
}

// Divide n by d and round up
#define DIV(n, d) (((n) + (d) - 1) / (d))

extern "C" void IMP_444ToNV12(const uint8_t *pArrayFull, uint8_t *pArrayNV12, int iWidth, int iHeight, size_t iPitchSrc, size_t iPitchDest)
{
	if (iWidth < 1 || iHeight < 1 || iPitchSrc < 1 || iPitchDest < 1)
		return;

	dim3 block(32, 16);

	dim3 grid_Y(DIV(iWidth, block.x), DIV(iHeight, block.y));
	kernel444ToNV12_Y << <grid_Y, block >> >(pArrayFull, pArrayNV12, iWidth, iHeight, (int)iPitchSrc, (int)iPitchDest);

	int offsetUV = iHeight * iPitchDest;
	iWidth /= 2; // No DIV here. UV needs two bytes, so it's not possible to round up if the width is odd.
	iHeight = DIV(iHeight, 2);
	dim3 grid_UV(DIV(iWidth, block.x), DIV(iHeight, block.y));
	kernel444ToNV12_UV << <grid_UV, block >> >(pArrayFull, pArrayNV12 + offsetUV, iWidth, iHeight, (int)iPitchSrc, (int)iPitchDest);
}
