#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
// Calculate ndvi of two channels d_A and d_B on GPU and store result in d_C
///////////////////////////////////////////////////////////////////////////////

__global__ void ndviGPU(
    float *d_C,
    short int *d_A,
    short int *d_B,
    int width,
    int height
){

    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(xIndex < width && yIndex < height)
	{
		unsigned int i = yIndex * (width) + xIndex;
		
		d_C[i] =  __fdividef( (float)(d_A[i] - d_B[i]), (float)(d_A[i] + d_B[i]) );
	}
}

