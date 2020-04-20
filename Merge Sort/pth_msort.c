// Include your C header files here

#include "pth_msort.h"
#include <stdlib.h>
#include <stdio.h>
int* gsorted;
int gN; 

void merge(int l, int h){
	int length = h - l + 1; 
	int* list = malloc(length * sizeof(int)); ; 
	int i, j, k; 
	for (i = 0; i < length; i++) 
        list[i] = gsorted[i + l]; 
	
	i = 0;
	j = length / 2;
	k = 0;
	
	while (i < length / 2 && j < length) { 
        if (list[i] <= list[j]) { 
            gsorted[k + l] = list[i];
			++i;
			++k;
		}
        else {
            gsorted[k + l] = list[j]; 
			++j;
			++k;
		}
    } 
	
	while (i < length / 2) { 
        gsorted[k + l] = list[i];
		++k;
		++i;
    } 
  
    while (j < length) { 
        gsorted[k + l] = list[j];
		++k;
		++j;
    } 
}

void msort(int l, int h){
		int length = h - l + 1;
		
		if(l < h){
        msort(l, l + length / 2 - 1); 
        msort(l + length / 2, h); 
        merge(l, h); 
		}
}
void* msortt(void* rank) { 
    long tnum = (long) rank; 
    int l = tnum * (gN / 4); 
    int h = (tnum + 1) * (gN / 4) - 1; 
    msort(l, h);   
} 
void mergeSortParallel (const int* values, unsigned int N, int* sorted) {
	int i; 
	long j;
	pthread_t threads[4];
	gN = N; 
	gsorted = malloc(N * sizeof(int));
	
	for (i = 0; i < N; i++){
		gsorted[i] = values[i];
	}
	
	for (j = 0; j < 4; j++) 
        pthread_create(&threads[j], NULL, msortt, (void*)j); 
  
    for (j = 0; j < 4; j++) 
        pthread_join(threads[j], NULL); 
	
	merge(0, N/2 - 1);
	merge(N/2, N - 1);
	merge(0, N - 1);
	
	for (i = 0; i < N; i++){
		sorted[i] = gsorted[i];
	}
	
}