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

    int result = MLMain(argc, argv);
    cutilExit(argc, argv);
    return result;
}

void cuFourier1D (double *h_A, long n)
{
    float norm = 1.0/sqrt((float) n);
    int mem_size = sizeof(Complex) * n;
    
    // Allocate host memory for the signal
    Complex* h_signal = (Complex*)malloc(sizeof(Complex) * n);
    
    // Initalize the memory for the signal
    for (long i = 0; i < n; ++i) {
        h_signal[i].x = (float)h_A[i];
        h_signal[i].y = 0.0f;
    }
    
    // Allocate device memory for signal
    Complex* d_signal;
    cutilSafeCall(cudaMalloc((void**)&d_signal, mem_size));
    // Copy host memory to device
    cutilSafeCall(cudaMemcpy(d_signal, h_signal, mem_size,
                              cudaMemcpyHostToDevice));
                              
    // CUFFT plan
    cufftHandle plan;
    cufftSafeCall(cufftPlan1d(&plan, n, CUFFT_C2C, 1));
    
    // Transform signal
    cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));
    
    // Copy device memory to host
    Complex* h_convolved_signal = h_signal;
    cutilSafeCall(cudaMemcpy(h_convolved_signal, d_signal, mem_size,
                              cudaMemcpyDeviceToHost));
    
    // Destroy CUFFT context
    cufftSafeCall(cufftDestroy(plan));
    
    // Return transformed signal to Mathematica as a Complex List
    MLPutFunction(stdlink,"List",n);
    for (long i = 0; i < n; i++) {
        MLPutFunction(stdlink,"Complex",2);
        MLPutFloat(stdlink,h_convolved_signal[i].x*norm);
        MLPutFloat(stdlink,h_convolved_signal[i].y*norm);
    }
    
    // Cleanup memory
    free(h_signal);
    cutilSafeCall(cudaFree(d_signal));
    
    cudaThreadExit();
}