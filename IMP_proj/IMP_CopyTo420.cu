#ifndef IMP_COPYTO420_CU
#define IMP_COPYTO420_CU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


__global__ void kernel(uint8_t *pArrayFull, uint8_t *pArrayY, uint8_t *pArrayU, uint8_t *pArrayV, int width, int height)
{
	int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
	int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
	int iPos = iPosY * width + iPosX;
	if (iPos > width * height) printf("kernel(...): Position is out of range!");

	if ((iPosX % 2 == 0) && (iPosY % 2 == 0))
	{
		int iPos2 = iPosY / 2 * width / 2 + iPosX / 2;
		pArrayU[iPos2] = pArrayFull[width * height + iPos];
		pArrayV[iPos2] = pArrayFull[2 * width * height + iPos];
	}

	pArrayY[iPos] = pArrayFull[iPos];
	//FILE *pFile;
	//pFile = fopen("C:\\dummy\\matrix.dat", "wa");
	//fprintf(pFile, "%d\t%d\t%d\t%d\t%d\t%d\t%d", blockIdx.x, threadIdx.x, iPosX, blockIdx.y, threadIdx.y, iPosY, iPos);
	//fclose(pFile);
}

extern "C" void IMP_CopyTo420(uint8_t *pArrayFull, uint8_t *pArrayY, uint8_t *pArrayU, uint8_t *pArrayV, int width, int height)
{
	cudaError_t error = cudaSuccess;

	dim3 block(32, 16, 1);

	dim3 grid(width / block.x, height / block.y, 1);
	kernel<<<grid, block>>>(pArrayFull, pArrayY, pArrayU, pArrayV, width, height);

	error = cudaDeviceSynchronize();
	if (error != cudaSuccess)
	{
		printf("Device synchronize failed! Error = %d\n", error);
	}
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("kernel() failed to launch error = %d\n", error);
	}

}

#endif