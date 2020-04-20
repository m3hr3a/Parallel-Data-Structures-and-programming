#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include "gpuerrors.h"
#include "math.h"

#define D 128
#define D_L 100
#define N_ref 1000000

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z
// ===========================> Functions Prototype <===============================
int fvecs_read (const char *fname, int d, int n, float *a);
int ivecs_write (const char *fname, int d, int n, const int *v);
void get_inputs(int argc, char *argv[], unsigned int& N, unsigned int& K);
void gpuKernels(float* ref, float* query, int* hist, unsigned int N, unsigned int K, double* gpu_kernel_time);
// =================================================================================

int main(int argc, char *argv[]) {

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);

    // get parameters from command line
    unsigned int N, K;
    get_inputs(argc, argv, N, K);

    // allocate memory in CPU for calculation
    float* reference; // reference vectors
    float* query; // query points
    int* hist;

    // Memory Allocation
	reference = (float*)malloc(N_ref * 128 * sizeof(float)); 
	query = (float*)malloc(N * 128 * sizeof(float)); 
	hist = (int*)malloc(K * N *sizeof(int)); 

    // fill references, query and labels with the values read from files
    fvecs_read("/home/data/ref.fvecs", D, N_ref, reference);
    fvecs_read("/home/data/query.fvecs", D, N, query);
    
    // time measurement for GPU calculation
    double gpu_kernel_time = 0.0;
    clock_t t0 = clock();
	  gpuKernels(reference, query, hist, N, K, &gpu_kernel_time);
    clock_t t1 = clock();

    printf("k=%d n=%d GPU=%g ms GPU-Kernels=%g ms\n",
    K, N, (t1-t0)/1000.0, gpu_kernel_time);

    // write the output to a file
    ivecs_write("outputs.ivecs", K, N, hist);
    
    // free allocated memory for later use
    free(reference);
    free(hist);
    free(query);

    return 0;
}
//-----------------------------------------------------------------------------
__global__ void caldist(float* query_c, float* ref_c, float* dist_c){

	__shared__ float q[128][32];
	__shared__ float r[128][64];
	
	int qnum = ty; 
	int rnum1 = 2 * tx; 
	int rnum2 = 2 * tx + 1;
	int off1 = 2 * tx * 128 + 4 * ty + (by * 125 + bx) * 128 * 64; 

	float d1, d2, temp1 = 0, temp2 = 0; 
	
	for(int k = 0; k < 4; k++)
		q[4 * ty + k][tx] = query_c[tx * 128 + 4 * ty + k];
	
	for(int k = 0; k<4; k++){
		r[4 * ty + k][2 * tx]     = ref_c[off1 + k];
		r[4 * ty + k][2 * tx + 1] = ref_c[off1 + 128 + k];
	}

	__syncthreads();		

	for(int index = 0; index < 128; index++){
		d1 = q[index][qnum] - r[index][rnum1]; 
		d2 = q[index][qnum] - r[index][rnum2]; 
		temp1 = temp1 + d1 * d1;
		temp2 = temp2 + d2 * d2;
	}
	
	dist_c[qnum * 1000000 + 64*((by * 125 )+ bx) + rnum1] = sqrt(temp1);
	dist_c[qnum * 1000000 + 64*((by * 125 )+ bx) + rnum2] = sqrt(temp2);
		
}

__global__ void caldist2(float* query_c, float* ref_c, float* dist_c){

	__shared__ float q[128][16];
	__shared__ float r[128][64];
	
	int qnum = ty; 
	int rnum1 = 2 * tx; 
	int rnum2 = 2 * tx + 1;
	int off1 = 2 * tx * 128 + 8 * ty + (by * 125 + bx) * 128 * 64;
	float d1 , d2; 
	float temp1 = 0;
	float temp2 = 0;
	
	for(int k = 0; k < 4; k++)
		q[4 * tx + k][ty] = query_c[ty * 128 + 4 * tx + k];
	
	for(int k = 0; k<8; k++){
		r[8 * ty + k][2 * tx]     = ref_c[off1 + k];
		r[8 * ty + k][2 * tx + 1] = ref_c[off1 + 128 +k];
	}

	__syncthreads();		
	
	for(int index = 0; index < 128; index++){
		d1 = q[index][qnum] - r[index][rnum1]; 
		d2 = q[index][qnum] - r[index][rnum2]; 
		temp1 = temp1 + d1 * d1;
		temp2 = temp2 + d2 * d2;	
	}
	
	dist_c[qnum*1000000 + rnum1 + 64*(by * 125 + bx)] = sqrt(temp1);
	dist_c[qnum*1000000 + rnum2 + 64*(by * 125 + bx)] = sqrt(temp2);
		
}


//-----------------------------------------------------------------------------
__global__ void minmax(float* border_c, float* dist_c){ // 32(16) block 64 thread
	
	__shared__ float min[64];
	__shared__ float max[64];
	
	int i = tx; 
	int offset = bx * 1000000;
	
	min[i] = dist_c[offset + i * 15625];
	max[i] = dist_c[offset + i * 15625];
	
	for(int j=1; j<15625; j++){
		if(min[i] > dist_c[offset + i * 15625 + j])
		   min[i] = dist_c[offset + i * 15625 + j];
		if(max[i] < dist_c[offset + i * 15625 + j])
		   max[i] = dist_c[offset + i * 15625 + j];
	}
	__syncthreads();
	
	
	if(i==0){
		float min1 = min[0];
		float max1 = max[0];
	
		for(int j=1; j < 64; j++){
			if(min[j] < min1)
				min1 = min[j];
			if(max[j] > max1)
				max1 = max[j];
		}
		
		border_c[2 * bx] = min1;
		border_c[2 * bx + 1] = max1;
	}

}
//-------------------------------------------------------------------------------------
__global__ void histcal(float* border_c, float* dist_c,int* hist_c2, unsigned int K){ // 32(16) block 64 thread

	int length = 15625;
	float min = border_c[2 * bx];
	float max = border_c[2 * bx+1];
	float bin_ilength = K / (max - min);
	int index;
	int index2;
	int mul = bx * 1000000 + tx*length;
	int mul2 = 64 * K * bx + K * tx;
	
	for(int j=0; j < K; j++){
		hist_c2[mul2 + j] = 0; 
	}
	
	__syncthreads();
	
	for(int i=0; i<length; i++){
		index = mul + i;
		index2 = (dist_c[index] - min) * bin_ilength;
		if(index2 == K) index2 = K - 1;
			atomicAdd(&hist_c2[mul2 + index2],1);				
	}
}
//---------------------------------------------------------------------------------------
__global__ void adder(int* hist_c, int* hist_c2, unsigned int K){ // K block 32 thread
	
	hist_c[bx + tx*K] = 0; 
	
	for(int i = 0; i < 64; i++){
		hist_c[bx + tx * K] = hist_c2[64 * K * tx + i * K + bx] + hist_c[tx * K + bx]; 
	}
	
}
//---------------------------------------------------------------------------------------
void gpuKernels(float* reference, float* query, int* hist, unsigned int N, unsigned int K, double* gpu_kernel_time) {
	
	float* query_c;
	float* ref_c; 
	int* hist_c;
	float* border_c; 
	float* dist_c;
	int* hist_c2;


	HANDLE_ERROR(cudaMalloc((void**)&hist_c2, 64*K*32  * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&query_c, 32 * 128 * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&ref_c, N_ref * 128 * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&hist_c, 32 * K * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dist_c, 32 * N_ref * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&border_c, 64  * sizeof(float)));
    
	dim3 dimblock(32, 32);
	dim3 dimgrid(125, 125);
	dim3 dimblock3(32,16);  
	
	HANDLE_ERROR(cudaMemcpy(ref_c, reference, N_ref * 128 * sizeof(float), cudaMemcpyHostToDevice));
	
	GpuTimer timer;
    timer.Start();
	
	for (int i = 0; i < 312; i++) {
		HANDLE_ERROR(cudaMemcpy(query_c, query + 32 * 128 * i , 32 * 128 * sizeof(float), cudaMemcpyHostToDevice));
		caldist<<<dimgrid, dimblock>>>(query_c, ref_c, dist_c);
		minmax<<<32, 64>>>(border_c, dist_c);
		histcal<<<32, 64>>>(border_c, dist_c,hist_c2, K);
		adder<<<K, 32>>>(hist_c, hist_c2, K);
		HANDLE_ERROR(cudaMemcpy(hist + 32 * K * i, hist_c,  32 * K * sizeof(int), cudaMemcpyDeviceToHost));
	}
	
	HANDLE_ERROR(cudaMemcpy(query_c, query + 32 * 128 * 312, 16 * 128 * sizeof(float), cudaMemcpyHostToDevice));
	caldist2<<<dimgrid, dimblock3>>>(query_c, ref_c, dist_c);
	minmax<<<16, 64>>>(border_c, dist_c);
	histcal<<<16, 64>>>(border_c, dist_c,hist_c2, K);
	adder<<<K, 16>>>(hist_c, hist_c2, K);
  	
	
	timer.Stop();
	
	  *gpu_kernel_time = timer.Elapsed();
	
	
    HANDLE_ERROR(cudaMemcpy(hist + 32 * K * 312, hist_c,  16 * K * sizeof(int), cudaMemcpyDeviceToHost));
	
	HANDLE_ERROR(cudaFree(query_c));
	HANDLE_ERROR(cudaFree(dist_c));
	HANDLE_ERROR(cudaFree(ref_c));
	HANDLE_ERROR(cudaFree(hist_c));
	HANDLE_ERROR(cudaFree(hist_c2));
	HANDLE_ERROR(cudaFree(border_c));
	
}
//-----------------------------------------------------------------------------
void get_inputs(int argc, char *argv[], unsigned int& N, unsigned int& K)
{
    if (
	argc != 3 ||
	atoi(argv[1]) < 0 || atoi(argv[1]) > 10000 ||
	atoi(argv[2]) < 0 || atoi(argv[2]) > 5000
	) {
        printf("<< Error >>\n");
        printf("Enter the following command:\n");
        printf("\t./nn  N  K\n");
        printf("\t\tN must be between 0 and 10000\n");
        printf("\t\tK must be between 0 and 5000\n");
		exit(-1);
    }
	N = atoi(argv[1]);
	K = atoi(argv[2]);
}
//-----------------------------------------------------------------------------
int fvecs_read (const char *fname, int d, int n, float *a)
{
  FILE *f = fopen (fname, "r");
  if (!f) {
    fprintf (stderr, "fvecs_read: could not open %s\n", fname);
    perror ("");
    return -1;
  }

  long i;
  for (i = 0; i < n; i++) {
    int new_d;

    if (fread (&new_d, sizeof (int), 1, f) != 1) {
      if (feof (f))
        break;
      else {
        perror ("fvecs_read error 1");
        fclose(f);
        return -1;
      }
    }

    if (new_d != d) {
      fprintf (stderr, "fvecs_read error 2: unexpected vector dimension\n");
      fclose(f);
      return -1;
    }

    if (fread (a + d * (long) i, sizeof (float), d, f) != d) {
      fprintf (stderr, "fvecs_read error 3\n");
      fclose(f);
      return -1;
    }
  }
  fclose (f);

  return i;
}


int ivecs_write (const char *fname, int d, int n, const int *v)
{
  FILE *f = fopen (fname, "w");
  if (!f) {
    perror ("ivecs_write");
    return -1;
  }

  int i;
  for (i = 0 ; i < n ; i++) {
    fwrite (&d, sizeof (d), 1, f);
    fwrite (v, sizeof (*v), d, f);
    v+=d;
  }
  fclose (f);
  return n;
}