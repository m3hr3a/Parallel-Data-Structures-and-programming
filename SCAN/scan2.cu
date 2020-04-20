#include "scan2.h"
#include "gpuerrors.h"

#define tx threadIdx.x
#define ty threadIdx.y

#define bx blockIdx.x
#define by blockIdx.y

__global__ void kernel(int* ad , int* cd ,int h,int n){
	
	int i = by*16384 + 512*bx+tx ; 

	if(i<h)
		cd[i] = ad[i];
	if(i < n-h)
	 cd[ i + h ] = ad[ i ] + ad[ i + h ];
	
	
}


__global__ void kernel1(int* ad , int* cd ,int h,int s){
	
	int i = by*16384 + 512*bx+tx ; 

	
		cd[i] = ad[i] + s;
	 cd[ i + h ] = ad[ i ] + ad[ i + h ] + s;
	
	
	
}
void gpuKernel(int* a, int* c,int n) {
	
	/*if(n<=(1<<26)){
	int * ad; 
	int * cd; 

	HANDLE_ERROR(cudaMalloc((void**)&ad, n* sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&cd, n* sizeof(int)));
	
	HANDLE_ERROR(cudaMemcpy(ad,a, n * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(cd,c, n * sizeof(int), cudaMemcpyHostToDevice));
	
	dim3 grid(16,n/16/1024);
	//print<<<1,1>>>(ad,n);
	int k = 1 ; 
	for(int h=1 ; h < n ; h=h*2){
		if(k&1){
			kernel<<<grid,1024>>>(ad,cd,h,n);
			//print<<<1,1>>>(cd,n);
			k++;
		}
		else{
			kernel<<<grid,1024>>>(cd,ad,h,n);
			//print<<<1,1>>>(ad,n);
			k++;
		}
	}
	
	if(k&1)
		cpy<<<grid,1024>>>(ad,cd,n);
	else{
		cpy<<<grid,1024>>>(cd,ad,n);
		//print<<<1,1>>>(ad,n);
		cpy2<<<grid,1024>>>(cd,ad,n);
	}
	
	HANDLE_ERROR(cudaMemcpy(c,cd, n * sizeof(int), cudaMemcpyDeviceToHost));
	
	//print<<<1,1>>>(ad,n);
	
	
	//for(int i = 0 ; i < n ; i++)
		//printf("%d ",c[i]);
	
	//printf("\n");
	
	HANDLE_ERROR(cudaFree(ad));
    HANDLE_ERROR(cudaFree(cd));
	}*/
	
	if(1) {
		
		int * ad; 
		int * cd; 
		int s = 0 ; 
		
		
		HANDLE_ERROR(cudaMalloc((void**)&ad, n/8* sizeof(int)));
		HANDLE_ERROR(cudaMalloc((void**)&cd, n/8* sizeof(int)));
		
		for(int q = 0 ; q < 8 ; q++){

		HANDLE_ERROR(cudaMemcpy(ad,a+n/8*q, n/8 * sizeof(int), cudaMemcpyHostToDevice));
		
		dim3 grid(32,n/65536/2); 
		dim3 grid2(32,n/512/16/16/2); 
		int k = 1 ; 
		for(int h=1 ; h < n/16 ; h=h*2){
			if(k&1){
				kernel<<<grid,512>>>(ad,cd,h,n/8);
				k++;
			}
			else{
				kernel<<<grid,512>>>(cd,ad,h,n/8);
				k++;
			}
		}
		if(k&1){
				kernel1<<<grid2,512>>>(ad,cd,n/16,s);
				k++;
			}
			else{
				kernel1<<<grid2,512>>>(cd,ad,n/16,s);
				k++;
			}
		if(!(k&1)){

			if(q<7)
				HANDLE_ERROR(cudaMemcpy(c+n/8*q+1,cd, n/8 * sizeof(int), cudaMemcpyDeviceToHost));
			else
				HANDLE_ERROR(cudaMemcpy(c+n/8*q+1,cd, (n/8-1) * sizeof(int), cudaMemcpyDeviceToHost));
		}
		else{

			if(q<7)
				HANDLE_ERROR(cudaMemcpy(c+n/8*q+1,ad, n/8 * sizeof(int), cudaMemcpyDeviceToHost));
			else
				HANDLE_ERROR(cudaMemcpy(c+n/8*q+1,ad, (n/8-1) * sizeof(int), cudaMemcpyDeviceToHost));
		}
		
		
		s = c[n/8*(q+1)-1] + a[n/8*(q+1) -1];
		
		}
		c[0] = 0 ; 
		HANDLE_ERROR(cudaFree(ad));
		HANDLE_ERROR(cudaFree(cd));

		
	}

}


