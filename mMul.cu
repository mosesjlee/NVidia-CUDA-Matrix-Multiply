#include <stdio.h>
#include "matrixMul.h"

///////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: P = M * N
///////////////////////////////////////////////////////////////////////////////

__global__ void
matrixMulKernelGlobal( float* Md, float* Nd, float* Pd, int width)
{
	//ThreadID index
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int k;
	float Psub = 0.0f;

	if(col < width && row < width) {
		for(k = 0;  k < width; k++){
			Psub += Md[row * width + k] * Nd[k * width + col];
		}
		//Write value into respective location
		Pd[row * width + col] = Psub;
	}
}

__global__ void
matrixMulKernelShared( float* Md, float* Nd, float* Pd, int width)
{
	__shared__ float s_M[THREAD_BLOCK][THREAD_BLOCK];
	__shared__ float s_N[THREAD_BLOCK][THREAD_BLOCK];

	//To access right indices in Global memory
	int row = THREAD_BLOCK * blockIdx.y + threadIdx.y;
	int col = THREAD_BLOCK * blockIdx.x + threadIdx.x; 
	int k;
	int m;

	float Psub = 0.0f;

	if(row < width && col < width){
		for (m = 0; m < width/THREAD_BLOCK; m++){
			s_M[threadIdx.y][threadIdx.x] = Md[row*width + m*THREAD_BLOCK + threadIdx.x];
			s_N[threadIdx.y][threadIdx.x] = Nd[(m*THREAD_BLOCK + threadIdx.y)*width + col];

			for(k = 0; k < THREAD_BLOCK; ++k){
				Psub += s_M[threadIdx.y][k] * s_N[k][threadIdx.x];
			}
			__syncthreads();
		}
	}
	Pd[row * width + col] = Psub;

}
