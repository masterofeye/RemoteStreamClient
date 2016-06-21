#ifndef IMP_COPYTO420_CU
#define IMP_COPYTO420_CU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


__global__ void kernel(uint8_t *pArrayFull, uint8_t *pArrayY, uint8_t *pArrayU, uint8_t *pArrayV, int width, int height)
{
	/********************
	pgMat->data is organized like this:
	YUVYUVYUVYUVYUVYUVYUVYUV
	YUVYUVYUVYUVYUVYUVYUVYUV
	YUVYUVYUVYUVYUVYUVYUVYUV
	YUVYUVYUVYUVYUVYUVYUVYUV
	YUVYUVYUVYUVYUVYUVYUVYUV
	YUVYUVYUVYUVYUVYUVYUVYUV
	YUVYUVYUVYUVYUVYUVYUVYUV
	YUVYUVYUVYUVYUVYUVYUVYUV

	but we want to have 
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	UUUU
	UUUU
	UUUU
	UUUU
	VVVV
	VVVV
	VVVV
	VVVV
	********************/

	int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
	int iPosX = blockIdx.x * blockDim.x + threadIdx.x;

	int iPosXIn = 3 * blockIdx.x * blockDim.x + 3 * threadIdx.x;
	int iPosIn = iPosY * 3 * width + iPosXIn;

	int iPos = iPosY * width + iPosX;
	pArrayY[iPos] = pArrayFull[iPosIn];

	if ((iPosX % 2 == 0) && (iPosY % 2 == 0))
	{
		int iPos2 = iPosY / 2 * width / 2 + iPosX / 2;
		pArrayU[iPos2] = pArrayFull[iPosIn + 1];
		pArrayV[iPos2] = pArrayFull[iPosIn + 2];
	}

	/************************************
	As a next step I should organize it like NV12 to spare out the converting step later (in ENC)
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	YYYYYYYY
	UVUVUVUV
	UVUVUVUV
	UVUVUVUV
	UVUVUVUV
	*************************************/
}

extern "C" void IMP_CopyTo420(uint8_t *pArrayFull, uint8_t *pArrayY, uint8_t *pArrayU, uint8_t *pArrayV, int width, int height)
{
	cudaError_t error = cudaSuccess;

	dim3 block(24, 16, 1);
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

	//cudaDeviceReset();
}

#endif