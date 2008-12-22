/*
 * This sample calculates scalar products of a 
 * given set of input vector pairs
 */
#include <stdio.h>

#include <cutil.h>

#include <mathlink.h>

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C"
float scalarProd(void);

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
///////////////////////////////////////////////////////////////////////////////
#include "scalarProd_kernel.cu"


///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{   

    CUT_DEVICE_INIT(argc, argv);
	int result = MLMain(argc, argv);
    CUT_EXIT(argc, argv);

    return result;
}

float scalarProd(void)
{
    float *h_A;
    float *h_B;
    float *h_C_GPU;
    float *d_A, *d_B, *d_C;
    char **heads;
	int* dims;
    int rank;

    if(! MLGetReal32Array(stdlink, &h_A, &dims, &heads, &rank))
    {
        return 0;
    }
    
    if(! MLGetReal32Array(stdlink, &h_B, &dims, &heads, &rank))
    {
        return 0;
    }
    
    h_C_GPU = (float *)malloc(dims[0]*sizeof(float));
    

    CUDA_SAFE_CALL( cudaMalloc((void **)&d_A, 
                                dims[0]*dims[1]*sizeof(float)) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_B, 
                                dims[0]*dims[1]*sizeof(float)) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_C, dims[0]*sizeof(float)) );

    //Copy options data to GPU memory for further processing 
    CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, dims[0]*dims[1]*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_B, h_B, dims[0]*dims[1]*sizeof(float), cudaMemcpyHostToDevice) );


    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    scalarProdGPU<<<128, 256>>>(d_C, d_A, d_B, dims[0], dims[1]);
    CUT_CHECK_ERROR("scalarProdGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    //Read back GPU results to compare them to CPU results
    CUDA_SAFE_CALL( cudaMemcpy(h_C_GPU, d_C, dims[0]*sizeof(float), cudaMemcpyDeviceToHost) );

    MLPutReal32List(stdlink, h_C_GPU, dims[0]);

    CUDA_SAFE_CALL( cudaFree(d_C) );
    CUDA_SAFE_CALL( cudaFree(d_B) );
    CUDA_SAFE_CALL( cudaFree(d_A) );

    MLReleaseReal32Array(stdlink, h_A, dims, heads, rank);
    MLReleaseReal32Array(stdlink, h_B, dims, heads, rank);
}
