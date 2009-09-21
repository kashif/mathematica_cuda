// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cufft.h>
#include <cutil_inline.h>

// includes, mathlink
#include <mathlink.h>

// Complex data type
typedef float2 Complex;

///////////////////////////////////////////////////////////////////////////////
// Showing the use of CUFFT for fast convolution using FFT.
///////////////////////////////////////////////////////////////////////////////
extern "C"
void cuFourier1D(double*, long);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{   
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    return MLMain(argc, argv);
}

void cuFourier1D (double *h_A, long n)
{
    double norm = 1.0/sqrt((double) n);
    long mem_size = sizeof(Complex) * n;
    
    // Allocate host memory for the signal
    Complex* h_signal = (Complex*)malloc(mem_size);
    
    // Initalize the memory for the signal
    for (long i = 0; i < n; ++i) {
        h_signal[i].x = (float)h_A[i];
        h_signal[i].y = 0.0f;
    }
    
    // Allocate device memory for signal
    Complex* d_signal;
    cutilSafeCall(cudaMalloc((void**)&d_signal, mem_size));
    // Copy host memory to device
    cutilSafeCall(cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));
                              
    // CUFFT plan
    cufftHandle plan;
    cufftSafeCall(cufftPlan1d(&plan, n, CUFFT_C2C, 1));
    
    // Transform signal
    cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE));
    
    // Copy device memory to host
    Complex* h_convolved_signal = h_signal;
    cutilSafeCall(cudaMemcpy(h_convolved_signal, d_signal, mem_size, cudaMemcpyDeviceToHost));

    // Release d_signal
    cutilSafeCall(cudaFree(d_signal));
    
    // Destroy CUFFT context
    cufftSafeCall(cufftDestroy(plan));
    
    // Return transformed signal to Mathematica as a Complex List
    MLPutFunction(stdlink, "Map", 2);
    MLPutFunction(stdlink, "Function", 2);
    MLPutFunction(stdlink, "List", 1);
    MLPutSymbol(stdlink, "x");
    MLPutFunction(stdlink, "Apply", 2);
    MLPutSymbol(stdlink, "Complex");
    MLPutSymbol(stdlink, "x");
    MLPutFunction(stdlink, "Partition", 2);
    MLPutFunction(stdlink, "Times", 2);
    MLPutReal(stdlink, norm);
    MLPutReal32List(stdlink, (float*)h_convolved_signal, 2*n);
    MLPutInteger(stdlink, 2);
    
    // Cleanup memory
    free(h_signal);
    
    cudaThreadExit();
}
