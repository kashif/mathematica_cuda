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
//  Utility funcs to wrap up saving a surface or the back buffer as a PPM file
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
#include <rendercheck_d3d10.h>

HRESULT CheckRenderD3D10::ActiveRenderTargetToPPM(ID3D10Device *pDevice, const char *zFileName)
{
	ID3D10RenderTargetView *pRTV = NULL;
	pDevice->OMGetRenderTargets(1,&pRTV,NULL);

	ID3D10Resource *pSourceResource = NULL;
	pRTV->GetResource(&pSourceResource);

	return ResourceToPPM(pDevice,pSourceResource,zFileName);
}

HRESULT CheckRenderD3D10::ResourceToPPM(ID3D10Device*pDevice, ID3D10Resource *pResource, const char *zFileName)
{
	D3D10_RESOURCE_DIMENSION rType;
	pResource->GetType(&rType);

	if(rType != D3D10_RESOURCE_DIMENSION_TEXTURE2D)
	{
		printf("SurfaceToPPM: pResource is not a 2D texture! Aborting...\n");
		return E_FAIL;
	}

	ID3D10Texture2D * pSourceTexture = (ID3D10Texture2D *)pResource;
	ID3D10Texture2D * pTargetTexture = NULL;

	D3D10_TEXTURE2D_DESC desc;
	pSourceTexture->GetDesc(&desc);
	desc.BindFlags = 0;
	desc.CPUAccessFlags = D3D10_CPU_ACCESS_READ;
	desc.Usage = D3D10_USAGE_STAGING;
		
	if(FAILED(pDevice->CreateTexture2D(&desc,NULL,&pTargetTexture)))
	{
		printf("SurfaceToPPM: Unable to create target Texture resoruce! Aborting... \n");
		return E_FAIL;
	}

	pDevice->CopyResource(pTargetTexture,pSourceTexture);

	D3D10_MAPPED_TEXTURE2D mappedTex2D;
	pTargetTexture->Map(0,D3D10_MAP_READ,0,&mappedTex2D);
	
	// Need to convert from dx pitch to pitch=width
	unsigned char *pPPMData = new unsigned char[desc.Width*desc.Height*4];
	for(unsigned int iHeight = 0;iHeight<desc.Height;iHeight++)
	{
		memcpy(&(pPPMData[iHeight*desc.Width*4]),(unsigned char*)(mappedTex2D.pData)+iHeight*mappedTex2D.RowPitch,desc.Width*4);
	}

	pTargetTexture->Unmap(0);

	// Prepends the PPM header info and bumps byte data afterwards
	cutSavePPM4ub(zFileName, pPPMData, desc.Width, desc.Height);

	delete [] pPPMData;
	pTargetTexture->Release();

	return S_OK;
}

bool CheckRenderD3D10::PPMvsPPM( const char *src_file, const char *ref_file, const char *exec_path, 
                                 const float epsilon, const float threshold )
{
    char *ref_file_path = cutFindFilePath(ref_file, exec_path);
    if (ref_file_path == NULL) {
        printf("CheckRenderD3D10::PPMvsPPM unable to find <%s> in <%s> Aborting comparison!\n", ref_file, exec_path);
        printf(">>> Check info.xml and [project//data] folder <%s> <<<\n", ref_file);
        printf("Aborting comparison!\n");
        printf("  FAILED!\n");
        return false;
    }

    return cutComparePPM(src_file,ref_file_path,epsilon,threshold,true) == CUTTrue;
}