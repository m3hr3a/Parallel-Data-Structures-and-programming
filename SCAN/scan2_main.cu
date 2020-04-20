//Do NOT MODIFY THIS FILE
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gpuerrors.h"
#include "scan2.h"

//-----------------------------------------------------------------------------
void fill(int* data, int size) {			// to fill array with -2 -1 0 1 2
    for (int i=0; i<size; ++i)
        data[i] = (int) (rand() % 3);
}

double calc_mse (int* data1, int* data2, int size) {	// calculate error by (  )^2
	double mse = 0.0;
	int i; 
	int e = 0;
	for (i=0; i<size; i++) {
		e = data1[i] - data2[i];
		e = e * e;
		mse += (double) e;
	}
	mse = mse / ((double)size);
	return mse;
}
//-----------------------------------------------------------------------------
void cpuKernel (int* a, int* c, int n) {		// calculate scan algorithm  
	int i = 0;
	int sum = 0;
	for (i = 0; i < n; i += 1){
		c[i] = sum;			// to have exclusive scan
		sum += (a[i]);
		//c[i] = sum;			// to have inclusive scan
	}
	return;
}
//-----------------------------------------------------------------------------
int main ( int argc, char** argv) {
	   
	int* a;
	int* c_serial;
	int* c;	
	
	int m = 5; 
	int n = 32;

	if (argc > 1){
		m = atoi(argv[1]);
		n = (1 << m);
	}

	a        = (int*)malloc(n * sizeof(int));
	c_serial = (int*)malloc(n * sizeof(int));
	c        = (int*)malloc(n * sizeof(int));
				
	srand(0);
	fill(a, n);

	clock_t t0 = clock(); 
	cpuKernel (a, c_serial, n);
	clock_t t1 = clock(); 
		
	clock_t t2 = clock(); 
	gpuKernel (a, c, n);
	clock_t t3 = clock();
		
	float mse;
	mse = calc_mse( c_serial, c, n );
	
	printf("n=%d\t CPU=%06ld ms GPU=%06ld ms mse=%f\n",n, (t1-t0)/1000, (t3-t2)/1000, mse);	
	//printf("%d\t%d\t%d\t%d\t%d\t", c[0],c[1],c[2],c[3],c[4]);

	free(a);
	free(c_serial);
	free(c);
	return 0;
}
//Do NOT MODIFY THIS FILE

