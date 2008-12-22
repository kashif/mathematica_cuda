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

    CUT_DEVICE_INIT(argc, argv);
	int result = MLMain(argc, argv);
    CUT_EXIT(argc, argv);

    return result;
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
    
    h_C_GPU = (float *)malloc(dims_A[0]*sizeof(float));
    

    CUDA_SAFE_CALL( cudaMalloc((void **)&d_A, 
                                dims_A[0]*dims_A[1]*sizeof(float)) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_B, 
                                dims_A[0]*dims_A[1]*sizeof(float)) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_C, dims_A[0]*sizeof(float)) );

    //Copy options data to GPU memory for further processing 
    CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, dims_A[0]*dims_A[1]*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_B, h_B, dims_A[0]*dims_A[1]*sizeof(float), cudaMemcpyHostToDevice) );


    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    scalarProdGPU<<<128, 256>>>(d_C, d_A, d_B, dims_A[0], dims_A[1]);
    CUT_CHECK_ERROR("scalarProdGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    //Read back GPU results to compare them to CPU results
    CUDA_SAFE_CALL( cudaMemcpy(h_C_GPU, d_C, dims_A[0]*sizeof(float), cudaMemcpyDeviceToHost) );

    MLPutReal32List(stdlink, h_C_GPU, dims_A[0]);

    CUDA_SAFE_CALL( cudaFree(d_C) );
    CUDA_SAFE_CALL( cudaFree(d_B) );
    CUDA_SAFE_CALL( cudaFree(d_A) );

    MLReleaseReal32Array(stdlink, h_A, dims_A, heads_A, rank_A);
    MLReleaseReal32Array(stdlink, h_B, dims_B, heads_B, rank_B);
}
