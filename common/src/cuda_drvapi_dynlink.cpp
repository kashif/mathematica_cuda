#include <dynlink/cuda_drvapi_dynlink.h>


tcuInit                         *cuInit;
tcuDeviceGet                    *cuDeviceGet;
tcuDeviceGetCount               *cuDeviceGetCount;
tcuDeviceGetName                *cuDeviceGetName;
tcuDeviceComputeCapability      *cuDeviceComputeCapability;
tcuDeviceTotalMem               *cuDeviceTotalMem;
tcuDeviceGetProperties          *cuDeviceGetProperties;
tcuDeviceGetAttribute           *cuDeviceGetAttribute;
tcuCtxCreate                    *cuCtxCreate;
tcuCtxDestroy                   *cuCtxDestroy;
tcuCtxAttach                    *cuCtxAttach;
tcuCtxDetach                    *cuCtxDetach;
tcuCtxPushCurrent               *cuCtxPushCurrent;
tcuCtxPopCurrent                *cuCtxPopCurrent;
tcuCtxGetDevice                 *cuCtxGetDevice;
tcuCtxSynchronize               *cuCtxSynchronize;
tcuModuleLoad                   *cuModuleLoad;
tcuModuleLoadData               *cuModuleLoadData;
tcuModuleLoadDataEx             *cuModuleLoadDataEx;
tcuModuleLoadFatBinary          *cuModuleLoadFatBinary;
tcuModuleUnload                 *cuModuleUnload;
tcuModuleGetFunction            *cuModuleGetFunction;
tcuModuleGetGlobal              *cuModuleGetGlobal;
tcuModuleGetTexRef              *cuModuleGetTexRef;
tcuMemGetInfo                   *cuMemGetInfo;
tcuMemAlloc                     *cuMemAlloc;
tcuMemAllocPitch                *cuMemAllocPitch;
tcuMemFree                      *cuMemFree;
tcuMemGetAddressRange           *cuMemGetAddressRange;
tcuMemAllocHost                 *cuMemAllocHost;
tcuMemFreeHost                  *cuMemFreeHost;
tcuMemcpyHtoD                   *cuMemcpyHtoD;
tcuMemcpyDtoH                   *cuMemcpyDtoH;
tcuMemcpyDtoD                   *cuMemcpyDtoD;
tcuMemcpyDtoA                   *cuMemcpyDtoA;
tcuMemcpyAtoD                   *cuMemcpyAtoD;
tcuMemcpyHtoA                   *cuMemcpyHtoA;
tcuMemcpyAtoH                   *cuMemcpyAtoH;
tcuMemcpyAtoA                   *cuMemcpyAtoA;
tcuMemcpy2D                     *cuMemcpy2D;
tcuMemcpy2DUnaligned            *cuMemcpy2DUnaligned;
tcuMemcpy3D                     *cuMemcpy3D;
tcuMemcpyHtoDAsync              *cuMemcpyHtoDAsync;
tcuMemcpyDtoHAsync              *cuMemcpyDtoHAsync;
tcuMemcpyHtoAAsync              *cuMemcpyHtoAAsync;
tcuMemcpyAtoHAsync              *cuMemcpyAtoHAsync;
tcuMemcpy2DAsync                *cuMemcpy2DAsync;
tcuMemcpy3DAsync                *cuMemcpy3DAsync;
tcuMemsetD8                     *cuMemsetD8;
tcuMemsetD16                    *cuMemsetD16;
tcuMemsetD32                    *cuMemsetD32;
tcuMemsetD2D8                   *cuMemsetD2D8;
tcuMemsetD2D16                  *cuMemsetD2D16;
tcuMemsetD2D32                  *cuMemsetD2D32;
tcuFuncSetBlockShape            *cuFuncSetBlockShape;
tcuFuncSetSharedSize            *cuFuncSetSharedSize;
tcuArrayCreate                  *cuArrayCreate;
tcuArrayGetDescriptor           *cuArrayGetDescriptor;
tcuArrayDestroy                 *cuArrayDestroy;
tcuArray3DCreate                *cuArray3DCreate;
tcuArray3DGetDescriptor         *cuArray3DGetDescriptor;
tcuTexRefCreate                 *cuTexRefCreate;
tcuTexRefDestroy                *cuTexRefDestroy;
tcuTexRefSetArray               *cuTexRefSetArray;
tcuTexRefSetAddress             *cuTexRefSetAddress;
tcuTexRefSetFormat              *cuTexRefSetFormat;
tcuTexRefSetAddressMode         *cuTexRefSetAddressMode;
tcuTexRefSetFilterMode          *cuTexRefSetFilterMode;
tcuTexRefSetFlags               *cuTexRefSetFlags;
tcuTexRefGetAddress             *cuTexRefGetAddress;
tcuTexRefGetArray               *cuTexRefGetArray;
tcuTexRefGetAddressMode         *cuTexRefGetAddressMode;
tcuTexRefGetFilterMode          *cuTexRefGetFilterMode;
tcuTexRefGetFormat              *cuTexRefGetFormat;
tcuTexRefGetFlags               *cuTexRefGetFlags;
tcuParamSetSize                 *cuParamSetSize;
tcuParamSeti                    *cuParamSeti;
tcuParamSetf                    *cuParamSetf;
tcuParamSetv                    *cuParamSetv;
tcuParamSetTexRef               *cuParamSetTexRef;
tcuLaunch                       *cuLaunch;
tcuLaunchGrid                   *cuLaunchGrid;
tcuLaunchGridAsync              *cuLaunchGridAsync;
tcuEventCreate                  *cuEventCreate;
tcuEventRecord                  *cuEventRecord;
tcuEventQuery                   *cuEventQuery;
tcuEventSynchronize             *cuEventSynchronize;
tcuEventDestroy                 *cuEventDestroy;
tcuEventElapsedTime             *cuEventElapsedTime;
tcuStreamCreate                 *cuStreamCreate;
tcuStreamQuery                  *cuStreamQuery;
tcuStreamSynchronize            *cuStreamSynchronize;
tcuStreamDestroy                *cuStreamDestroy;

#include <stdio.h>
#define QUOTE(x)        #x


#if defined(_WIN32) || defined(_WIN64)

#include <Windows.h>


#define LOAD_LIBRARY()                                          \
    LPCSTR DllName = "nvcuda.dll";                              \
    HMODULE CudaDrvLib = LoadLibrary(DllName);                  \
    if (CudaDrvLib == NULL)                                     \
    {                                                           \
        return CUDA_ERROR_UNKNOWN;                              \
    }

#define GET_PROC(name)                                          \
    name = (t##name *)GetProcAddress(CudaDrvLib, QUOTE(name));  \
    if (name == NULL) return CUDA_ERROR_UNKNOWN


#elif defined(__unix__)

#include <dlfcn.h>

#define LOAD_LIBRARY()                                          \
    void* CudaDrvLib = dlopen("libcuda.so", RTLD_LAZY);         \
    if (CudaDrvLib == NULL)                                     \
    {                                                           \
        return CUDA_ERROR_UNKNOWN;                              \
    }

#define GET_PROC(name)                                          \
    name = (t##name *)dlsym(CudaDrvLib, QUOTE(name));           \
    if (name == NULL) return CUDA_ERROR_UNKNOWN

#endif


CUresult CUDAAPI cuDriverAPIdynload(void)
{
    LOAD_LIBRARY();
    GET_PROC(cuInit);
    GET_PROC(cuDeviceGet);
    GET_PROC(cuDeviceGetCount);
    GET_PROC(cuDeviceGetName);
    GET_PROC(cuDeviceComputeCapability);
    GET_PROC(cuDeviceTotalMem);
    GET_PROC(cuDeviceGetProperties);
    GET_PROC(cuDeviceGetAttribute);
    GET_PROC(cuCtxCreate);
    GET_PROC(cuCtxDestroy);
    GET_PROC(cuCtxAttach);
    GET_PROC(cuCtxDetach);
    GET_PROC(cuCtxPushCurrent);
    GET_PROC(cuCtxPopCurrent);
    GET_PROC(cuCtxGetDevice);
    GET_PROC(cuCtxSynchronize);
    GET_PROC(cuModuleLoad);
    GET_PROC(cuModuleLoadData);
    GET_PROC(cuModuleLoadDataEx);
    GET_PROC(cuModuleLoadFatBinary);
    GET_PROC(cuModuleUnload);
    GET_PROC(cuModuleGetFunction);
    GET_PROC(cuModuleGetGlobal);
    GET_PROC(cuModuleGetTexRef);
    GET_PROC(cuMemGetInfo);
    GET_PROC(cuMemAlloc);
    GET_PROC(cuMemAllocPitch);
    GET_PROC(cuMemFree);
    GET_PROC(cuMemGetAddressRange);
    GET_PROC(cuMemAllocHost);
    GET_PROC(cuMemFreeHost);
    GET_PROC(cuMemcpyHtoD);
    GET_PROC(cuMemcpyDtoH);
    GET_PROC(cuMemcpyDtoD);
    GET_PROC(cuMemcpyDtoA);
    GET_PROC(cuMemcpyAtoD);
    GET_PROC(cuMemcpyHtoA);
    GET_PROC(cuMemcpyAtoH);
    GET_PROC(cuMemcpyAtoA);
    GET_PROC(cuMemcpy2D);
    GET_PROC(cuMemcpy2DUnaligned);
    GET_PROC(cuMemcpy3D);
    GET_PROC(cuMemcpyHtoDAsync);
    GET_PROC(cuMemcpyDtoHAsync);
    GET_PROC(cuMemcpyHtoAAsync);
    GET_PROC(cuMemcpyAtoHAsync);
    GET_PROC(cuMemcpy2DAsync);
    GET_PROC(cuMemcpy3DAsync);
    GET_PROC(cuMemsetD8);
    GET_PROC(cuMemsetD16);
    GET_PROC(cuMemsetD32);
    GET_PROC(cuMemsetD2D8);
    GET_PROC(cuMemsetD2D16);
    GET_PROC(cuMemsetD2D32);
    GET_PROC(cuFuncSetBlockShape);
    GET_PROC(cuFuncSetSharedSize);
    GET_PROC(cuArrayCreate);
    GET_PROC(cuArrayGetDescriptor);
    GET_PROC(cuArrayDestroy);
    GET_PROC(cuArray3DCreate);
    GET_PROC(cuArray3DGetDescriptor);
    GET_PROC(cuTexRefCreate);
    GET_PROC(cuTexRefDestroy);
    GET_PROC(cuTexRefSetArray);
    GET_PROC(cuTexRefSetAddress);
    GET_PROC(cuTexRefSetFormat);
    GET_PROC(cuTexRefSetAddressMode);
    GET_PROC(cuTexRefSetFilterMode);
    GET_PROC(cuTexRefSetFlags);
    GET_PROC(cuTexRefGetAddress);
    GET_PROC(cuTexRefGetArray);
    GET_PROC(cuTexRefGetAddressMode);
    GET_PROC(cuTexRefGetFilterMode);
    GET_PROC(cuTexRefGetFormat);
    GET_PROC(cuTexRefGetFlags);
    GET_PROC(cuParamSetSize);
    GET_PROC(cuParamSeti);
    GET_PROC(cuParamSetf);
    GET_PROC(cuParamSetv);
    GET_PROC(cuParamSetTexRef);
    GET_PROC(cuLaunch);
    GET_PROC(cuLaunchGrid);
    GET_PROC(cuLaunchGridAsync);
    GET_PROC(cuEventCreate);
    GET_PROC(cuEventRecord);
    GET_PROC(cuEventQuery);
    GET_PROC(cuEventSynchronize);
    GET_PROC(cuEventDestroy);
    GET_PROC(cuEventElapsedTime);
    GET_PROC(cuStreamCreate);
    GET_PROC(cuStreamQuery);
    GET_PROC(cuStreamSynchronize);
    GET_PROC(cuStreamDestroy);
    return CUDA_SUCCESS;
}

#undef QUOTE
#undef LOAD_LIBRARY
#undef GET_PROC
