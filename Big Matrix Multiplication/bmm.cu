#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include "gpuerrors.h"
#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY is used to set number of threads in a CUDA block 
#define TILEX 32
#define TILEY 32

dim3 getDimGrid(const int m, const int n) {
	
	dim3 dimGrid(n/TILEY,n/TILEX);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY);
	return dimBlock;
}
__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {
	

		
	__shared__ float as[TILEY][TILEY];
	__shared__ float bs[TILEY][TILEY];
	
	int i = TILEY * by + ty ; 
	int j = TILEY * bx + tx ; 
	
	float t = 0 ; 

	for(int k = 0 ; k < n/TILEX ; ++ k){

		as[ty][tx] = ad[ i * n  + tx + TILEX * k];
		bs[ty][tx] = bd[ j + (ty + TILEX * k)*n];
		
		__syncthreads(); 
		
		for(int s = 0 ; s < TILEY ; ++s)
			t+= as[ty][s] * bs[s][tx] ;

		__syncthreads();
	} 

	cd[i * n + j ] = t ; 
	}
	

//-----------------------------------------------------------------------------
void gpuKernel(const float* const a, const float* const b, float* c, const int m, const int n) {
	
	// you need to modify this function because the matrices do not fit in GPU for large values of m.
	
    if(m<14){
	float* ad;
	float* bd;
	float* cd;
    HANDLE_ERROR(cudaMalloc((void**)&ad, n*n * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&bd, n*n * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&cd, n*n * sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(ad, a, n*n * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(bd, b, n*n * sizeof(float), cudaMemcpyHostToDevice));

	dim3 dimGrid = getDimGrid(m,n); 
	dim3 dimBlock = getDimBlock(m,n); 
	
	kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, m, n); 
	
	HANDLE_ERROR(cudaMemcpy(c, cd, n*n * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(ad));
    HANDLE_ERROR(cudaFree(bd));
    HANDLE_ERROR(cudaFree(cd));
	}
	if(m == 14){
		
		float* ad;
		float* bd;
		float* cd;
		
	    HANDLE_ERROR(cudaMalloc((void**)&ad, n*n/4 * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&bd, n*n/4 * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cd, n*n/4 * sizeof(float)));
		
		
		float * ctemp = (float*)malloc(n*n * sizeof(float)/4);
		float * atemp = (float*)malloc(n*n * sizeof(float)/4);
		float * btemp = (float*)malloc(n*n * sizeof(float)/4);

		
		
		dim3 dimGrid = getDimGrid(m,n/2); 
		dim3 dimBlock = getDimBlock(m,n); 
		
		int k = 0 ; 
		int s = 0 ; 
		//a1
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						atemp [(o-n/2*k)*n/2 + (p-n/2*s)]= a[o*n+p];
					}
			}
		//b1
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						btemp [(o-n/2*k)*n/2 + (p-n/2*s)]= b[o*n+p];
					}
			}	
		HANDLE_ERROR(cudaMemcpy(ad, atemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(bd, btemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, m, n/2); 
		HANDLE_ERROR(cudaMemcpy(ctemp, cd, n*n* sizeof(float)/4, cudaMemcpyDeviceToHost));
		//c1
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						c[o*n+p] = ctemp[(o-n/2*k)*n/2 + (p-n/2*s)];
					}
				}
		s = 1 ; 
		//b2
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						btemp [(o-n/2*k)*n/2 + (p-n/2*s)]= b[o*n+p];
					}
			}
			
		HANDLE_ERROR(cudaMemcpy(bd, btemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, m, n/2); 
		HANDLE_ERROR(cudaMemcpy(ctemp, cd, n*n* sizeof(float)/4, cudaMemcpyDeviceToHost));
			//c2
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						c[o*n+p] = ctemp[(o-n/2*k)*n/2 + (p-n/2*s)];
					}
				}
		
		//a3
		k=1 ; 
		s = 0 ; 
	    for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						atemp [(o-n/2*k)*n/2 + (p-n/2*s)]= a[o*n+p];
					}
			}
			
		HANDLE_ERROR(cudaMemcpy(ad, atemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, m, n/2); 
		HANDLE_ERROR(cudaMemcpy(ctemp, cd, n*n* sizeof(float)/4, cudaMemcpyDeviceToHost));
		//c4
		s = 1 ;
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						c[o*n+p] = ctemp[(o-n/2*k)*n/2 + (p-n/2*s)];
					}
				}
		//b1
		k = 0 ; 
		s = 0 ; 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						btemp [(o-n/2*k)*n/2 + (p-n/2*s)]= b[o*n+p];
					}
			}
			
		HANDLE_ERROR(cudaMemcpy(bd, btemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, m, n/2); 
		HANDLE_ERROR(cudaMemcpy(ctemp, cd, n*n* sizeof(float)/4, cudaMemcpyDeviceToHost));
	    //c3
		k = 1 ; 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						c[o*n+p] = ctemp[(o-n/2*k)*n/2 + (p-n/2*s)];
					}
				}
		//a2 
		k= 0 ; 
		s = 1 ; 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						atemp [(o-n/2*k)*n/2 + (p-n/2*s)]= a[o*n+p];
					}
			}
		//b3
		k = 1 ; 
		s = 0 ; 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						btemp [(o-n/2*k)*n/2 + (p-n/2*s)]= b[o*n+p];
					}
			}
			
		HANDLE_ERROR(cudaMemcpy(bd, btemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(ad, atemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, m, n/2); 
		HANDLE_ERROR(cudaMemcpy(ctemp, cd, n*n* sizeof(float)/4, cudaMemcpyDeviceToHost));
		//c1 
		s = 0 ; 
		k = 0 ; 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						c[o*n+p] += ctemp[(o-n/2*k)*n/2 + (p-n/2*s)];
					}
				}
	    // b4 
		s = 1 ; 
		k = 1 ; 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						btemp [(o-n/2*k)*n/2 + (p-n/2*s)]= b[o*n+p];
					}
			}
			
		HANDLE_ERROR(cudaMemcpy(bd, btemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, m, n/2); 
		HANDLE_ERROR(cudaMemcpy(ctemp, cd, n*n* sizeof(float)/4, cudaMemcpyDeviceToHost));
	    //c2 
		k = 0 ; 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						c[o*n+p] += ctemp[(o-n/2*k)*n/2 + (p-n/2*s)];
					}
				}
		//a4
		k= 1 ; 
		s = 1 ; 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						atemp [(o-n/2*k)*n/2 + (p-n/2*s)]= a[o*n+p];
					}
			}
		//b3
		k = 1 ; 
		s = 0 ; 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						btemp [(o-n/2*k)*n/2 + (p-n/2*s)]= b[o*n+p];
					}
			}
			
		HANDLE_ERROR(cudaMemcpy(bd, btemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(ad, atemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, m, n/2); 
		HANDLE_ERROR(cudaMemcpy(ctemp, cd, n*n* sizeof(float)/4, cudaMemcpyDeviceToHost));
		//c 3
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						c[o*n+p] += ctemp[(o-n/2*k)*n/2 + (p-n/2*s)];
					}
				}
	    // b4 
		s = 1 ; 
		k = 1 ; 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
				
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						btemp [(o-n/2*k)*n/2 + (p-n/2*s)]= b[o*n+p];
					}
			}
			
		HANDLE_ERROR(cudaMemcpy(bd, btemp, n*n * sizeof(float)/4, cudaMemcpyHostToDevice));
		kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, m, n/2); 
		HANDLE_ERROR(cudaMemcpy(ctemp, cd, n*n* sizeof(float)/4, cudaMemcpyDeviceToHost));
	    //c4 
		for(int o = n/2 * k ; o< (k+1)*(n/2) ; ++o ){
					for(int p = n/2*s ; p<(s+1)*(n/2) ; ++p){
						c[o*n+p] += ctemp[(o-n/2*k)*n/2 + (p-n/2*s)];
					}
				}
		HANDLE_ERROR(cudaFree(ad));
		HANDLE_ERROR(cudaFree(bd));
		HANDLE_ERROR(cudaFree(cd));
		
	
    }

}
