#ifndef COPY420_CU
#define COPY420_CU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void kernel(unsigned char *pArrayFull, unsigned char *pArrayU, unsigned char *pArrayV, int width, int height)
{
	int iIndexX = blockIdx.x * blockDim.x + threadIdx.x;
	int iIndexY = blockIdx.y * blockDim.y + threadIdx.y;

	int iIndexUVX = (2 * blockIdx.x * blockDim.x) + (2 * threadIdx.x);
	int iIndexUVY = (2 * blockIdx.y * blockDim.y) + (2 * threadIdx.y);

	pArrayU[iIndexY * gridDim.x + iIndexX] = pArrayFull[width * height + iIndexUVY * gridDim.x * blockDim.x + iIndexUVX];
	pArrayV[iIndexY * gridDim.x + iIndexX] = pArrayFull[2 * width * height + iIndexUVY * gridDim.x * blockDim.x + iIndexUVX];
}

extern "C" void Copy420(unsigned char *pArrayFull, unsigned char *pArrayU, unsigned char *pArrayV, int width, int height)
{
	cudaError_t error = cudaSuccess;

	dim3 block(128, 128, 1);
	dim3 grid((width / 2) / block.x, (height / 2) / block.y, 1);

	//unsigned int block = 8 * 8;
	//unsigned int grid = ((width / 2) * (height / 2)) / block;

	kernel<<<grid, block>>>(pArrayFull, pArrayU, pArrayV, width, height);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("kernel() failed to launch error = %d\n", error);
	}
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess)
	{
		printf("Device synchronize failed! Error = %d\n", error);
	}

}

#endif