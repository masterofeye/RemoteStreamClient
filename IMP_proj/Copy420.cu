#ifndef COPY420_CU
#define COPY420_CU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


__global__ void kernel(unsigned char *pArrayFull, unsigned char *pArrayU, unsigned char *pArrayV, int width, int height)
{

	int iIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int iIndexU = (width * height) + (2 * blockIdx.x * blockDim.x) + (2 * threadIdx.x);
	int iIndexV = (2 * width * height) + (2 * blockIdx.x * blockDim.x) + (2 * threadIdx.x);

	pArrayU[iIndex] = pArrayFull[iIndexU];
	pArrayV[iIndex] = pArrayFull[iIndexV]; 
}

extern "C" void Copy420(unsigned char *pArrayFull, unsigned char *pArrayU, unsigned char *pArrayV, int width, int height)
{
	cudaError_t error = cudaSuccess;

	kernel<<<width / 2, height / 2>>>(pArrayFull, pArrayU, pArrayV, width, height);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("kernel() failed to launch error = %d\n", error);
	}

}

#endif