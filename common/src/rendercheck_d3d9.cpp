/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

//
//  Utility funcs to wrap up savings a surface or the back buffer as a PPM file
//	In addition, wraps up a threshold comparision of two PPMs.
//
//	These functions are designed to be used to implement an automated QA testing for SDK samples.
//
//	Author: Bryan Dudash
//  Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include <cutil.h>
#include <rendercheck_d3d9.h>
#include <d3dx9.h>

// originally copied from checkrender_gl.cpp and slightly modified
bool CheckRenderD3D9::PPMvsPPM( const char *src_file, const char *ref_file, const char *exec_path, 
                                const float epsilon, const float threshold )
{
    char *ref_file_path = cutFindFilePath(ref_file, exec_path);
    if (ref_file_path == NULL) {
        printf("CheckRenderD3D9::PPMvsPPM unable to find <%s> in <%s> Aborting comparison!\n", ref_file, exec_path);
        printf(">>> Check info.xml and [project//data] folder <%s> <<<\n", ref_file);
        printf("Aborting comparison!\n");
        printf("  FAILED!\n");
        return false;
    }

    return (cutComparePPM( src_file, ref_file_path, epsilon, threshold, true ) == CUTTrue);
};

HRESULT CheckRenderD3D9::BackbufferToPPM(IDirect3DDevice9*pDevice, const char *zFileName)
{
	IDirect3DSurface9 *pSurface = NULL;	

	if(FAILED(pDevice->GetBackBuffer(0,0,D3DBACKBUFFER_TYPE_MONO, &pSurface)))
	{
		printf("Unable to get the back buffer.  Aborting...\n");
		return E_FAIL;
	}

	//D3DXSaveSurfaceToFile("C:\\bing.dds",D3DXIFF_DDS,pSurface,NULL,NULL);

	HRESULT hr = S_OK;
	hr = SurfaceToPPM(pDevice,pSurface,zFileName);

	pSurface->Release();

	return hr;
}

HRESULT CheckRenderD3D9::SurfaceToPPM(IDirect3DDevice9*pDevice, IDirect3DSurface9 *pSurface, const char *zFileName)
{
	D3DSURFACE_DESC pDesc;
	pSurface->GetDesc(&pDesc);

	// $$ For now only support common 8bit formats.  TODO: support for more complex formats via conversion?
	if(!(pDesc.Format == D3DFMT_A8R8G8B8 || pDesc.Format == D3DFMT_X8R8G8B8))
	{
		return E_INVALIDARG;
	}

	IDirect3DTexture9 *pTargetTex = NULL;
	if(FAILED(pDevice->CreateTexture(pDesc.Width,pDesc.Height,1,D3DUSAGE_DYNAMIC,pDesc.Format,D3DPOOL_SYSTEMMEM,&pTargetTex,NULL)))
	{
		printf("Unable to create texture for surface transfer! Aborting...\n");
		return E_FAIL;
	}

	IDirect3DSurface9 *pTargetSurface = NULL;
	if(FAILED(pTargetTex->GetSurfaceLevel(0,&pTargetSurface)))
	{
		printf("Unable to get surface for surface transfer! Aborting...\n");
		return E_FAIL;
	}

	// This is required because we cannot lock a D3DPOOL_DEAULT surface directly.  So, we copy to our sysmem surface.
	if(FAILED(pDevice->GetRenderTargetData(pSurface,pTargetSurface)))
	{
		printf("Unable to GetRenderTargetData() for surface transfer! Aborting...\n");
		return E_FAIL;
	}

	D3DLOCKED_RECT lockedRect;
	HRESULT hr = pTargetSurface->LockRect(&lockedRect,NULL,0);

	// Need to convert from dx pitch to pitch=width
	//
	// $ PPM is BGR and not RGB it seems. Saved image looks "funny" in viewer(red and blue swapped), but since ref will be dumped using same method, this is ok.
	//		however, if we want the saved image to be properly colored, then we can swizzle the color bytes here.
	unsigned char *pPPMData = new unsigned char[pDesc.Width*pDesc.Height*4];
	for(unsigned int iHeight = 0;iHeight<pDesc.Height;iHeight++)
	{
#if 1 // swizzle to implment RGB to BGR conversion.
		for(unsigned int iWidth = 0;iWidth< pDesc.Width;iWidth++)
		{
			DWORD color = *(DWORD *)((unsigned char*)(lockedRect.pBits)+iHeight*lockedRect.Pitch + iWidth*4);

			// R<->B, [7:0] <-> [23:16], swizzle
			color = ((color&0xFF)<<16) | (color&0xFF00) | ((color&0xFF0000)>>16) | (color&0xFF000000);

			memcpy(&(pPPMData[(iHeight*pDesc.Width + iWidth)*4]),(unsigned char*)&color,4);
		}
#else
		memcpy(&(pPPMData[iHeight*pDesc.Width*4]),(unsigned char*)(lockedRect.pBits)+iHeight*lockedRect.Pitch,pDesc.Width*4);
#endif
	}

	pTargetSurface->UnlockRect();

	// Prepends the PPM header info and bumps byte data afterwards
	cutSavePPM4ub(zFileName, pPPMData, pDesc.Width, pDesc.Height);

	delete [] pPPMData;
	pTargetSurface->Release();
	pTargetTex->Release();
	
	return S_OK;
}