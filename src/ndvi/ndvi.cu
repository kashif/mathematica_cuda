/*
 * This sample calculates Normalized Vegitation Index (NDVI) of a 
 * given pair of near infra red and red
 */

#include <stdio.h>
#include <cutil_inline.h>
#include <mathlink.h>

///////////////////////////////////////////////////////////////////////////////
// Calculate NDVI on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C"
void ndvi(void);

///////////////////////////////////////////////////////////////////////////////
// Calculate NDVI on GPU
///////////////////////////////////////////////////////////////////////////////
#include "ndvi_kernel.cu"


///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{   
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    int result = MLMain(argc, argv);
    cutilExit(argc, argv);
    return result;
}

void ndvi(void)
{
    short int *h_A, *h_B;
    float *h_C_GPU;
    short int *d_A, *d_B;
    float *d_C;

    char **heads_A, **heads_B;
    int *dims_A, *dims_B;
    int rank_A, rank_B;

    if(! MLGetInteger16Array(stdlink, &h_A, &dims_A, &heads_A, &rank_A))
    {
        return;
    }
    
    if(! MLGetInteger16Array(stdlink, &h_B, &dims_B, &heads_B, &rank_B))
    {
        return;
    }
    
    printf("dims0 = %d, dims1 = %d\n",dims_A[0],dims_A[1]);
    
    //Initializing data
    h_C_GPU = (float *)malloc(dims_A[0]*dims_A[1]*sizeof(float));

    //Allocating GPU memory
    cutilSafeCall( cudaMalloc((void **)&d_A, dims_A[0]*dims_A[1]*sizeof(short int)) );
	//printf("array size %d\n", dims_A[0]*dims_A[1]*sizeof(short int));
    cutilSafeCall( cudaMalloc((void **)&d_B, dims_A[0]*dims_A[1]*sizeof(short int)) );
    cutilSafeCall( cudaMalloc((void **)&d_C, dims_A[0]*dims_A[1]*sizeof(float)) );

    //Copy data to GPU memory for further processing 
    cutilSafeCall( cudaMemcpy(d_A, h_A, dims_A[0]*dims_A[1]*sizeof(short int),cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy(d_B, h_B, dims_A[0]*dims_A[1]*sizeof(short int),cudaMemcpyHostToDevice) );

    cutilSafeCall( cudaThreadSynchronize() );

    dim3 grid(ceil((float)dims_A[0]/(float)16.0f), ceil((float) dims_A[1]/32.0f), 1);
    dim3 threads(ceil( dims_A[0]/(float)grid.x), ceil( dims_A[1]/(float)grid.y), 1);

    ndviGPU<<<grid, threads>>>(d_C, d_A, d_B, dims_A[0], dims_A[1]);
    //cutilCheckMsg("ndviGPU() execution failed\n");
    cutilSafeCall( cudaThreadSynchronize() );

    //Release d_A and d_B
    cutilSafeCall( cudaFree(d_B) );
    cutilSafeCall( cudaFree(d_A) );

    printf("dims0 = %d, dims1 = %d\n",dims_A[0],dims_A[1]);

    //Read back GPU results into h_C_GPU
    cutilSafeCall( cudaMemcpy(h_C_GPU, d_C, dims_A[0]*dims_A[1]*sizeof(float), cudaMemcpyDeviceToHost) );

    //Release d_C
    cutilSafeCall( cudaFree(d_C) );

    //Return result
    MLPutReal32List(stdlink, h_C_GPU, dims_A[0]*dims_A[1]);

    //Release h_A and h_B
    MLReleaseInteger16Array(stdlink, h_A, dims_A, heads_A, rank_A);
    MLReleaseInteger16Array(stdlink, h_B, dims_B, heads_B, rank_B);

    cudaThreadExit();
}

