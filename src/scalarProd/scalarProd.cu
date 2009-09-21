/*
 * This sample calculates scalar products of a 
 * given set of input vector pairs
 */

#include <stdio.h>
#include <cutil_inline.h>
#include <mathlink.h>

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C"
void scalarProd(void);

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
///////////////////////////////////////////////////////////////////////////////
#include "scalarProd_kernel.cu"


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

    return MLMain(argc, argv);
}

void scalarProd(void)
{
    float *h_A, *h_B, *h_C_GPU;
    float *d_A, *d_B, *d_C;

    char **heads_A, **heads_B;
    int *dims_A, *dims_B;
    int rank_A, rank_B;

    if(! MLGetReal32Array(stdlink, &h_A, &dims_A, &heads_A, &rank_A))
    {
        return;
    }
    
    if(! MLGetReal32Array(stdlink, &h_B, &dims_B, &heads_B, &rank_B))
    {
        return;
    }
    
    //Initializing data
    h_C_GPU = (float *)malloc(dims_A[0]*sizeof(float));

    //Allocating GPU memory
    cutilSafeCall( cudaMalloc((void **)&d_A, dims_A[0]*dims_A[1]*sizeof(float)) );
    cutilSafeCall( cudaMalloc((void **)&d_B, dims_A[0]*dims_A[1]*sizeof(float)) );
    cutilSafeCall( cudaMalloc((void **)&d_C, dims_A[0]*sizeof(float)) );

    //Copy options data to GPU memory for further processing 
    cutilSafeCall( cudaMemcpy(d_A, h_A, dims_A[0]*dims_A[1]*sizeof(float),cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy(d_B, h_B, dims_A[0]*dims_A[1]*sizeof(float),cudaMemcpyHostToDevice) );

    cutilSafeCall( cudaThreadSynchronize() );
    scalarProdGPU<<<128, 256>>>(d_C, d_A, d_B, dims_A[0], dims_A[1]);
    cutilCheckMsg("scalarProdGPU() execution failed\n");
    cutilSafeCall( cudaThreadSynchronize() );

    //Read back GPU results to compare them to CPU results
    cutilSafeCall( cudaMemcpy(h_C_GPU, d_C, dims_A[0]*sizeof(float), cudaMemcpyDeviceToHost) );

    MLPutReal32List(stdlink, h_C_GPU, dims_A[0]);

    cutilSafeCall( cudaFree(d_C) );
    cutilSafeCall( cudaFree(d_B) );
    cutilSafeCall( cudaFree(d_A) );

    MLReleaseReal32Array(stdlink, h_A, dims_A, heads_A, rank_A);
    MLReleaseReal32Array(stdlink, h_B, dims_B, heads_B, rank_B);

    cudaThreadExit();
}
