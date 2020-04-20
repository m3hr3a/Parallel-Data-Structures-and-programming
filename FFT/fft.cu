// GPU fft , written by Mehrsa Pourya , May 2019

#include "fft.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y

#define rate 3
#define rate2 8

__global__ void kernelorderr2(float* x_r_d, float* x_i_d, const unsigned int n) {
	
	float temp ;
	int t, t1, rev_t, p ,i ;
	i = (by * 8192 ) + ( bx * 128 ) + tx ;
	 

	for(int k = 0 ; k < rate2 ; k = k + 2 ){
			
		t =( i << rate ) + k ; 
		t1 = t ; 
		rev_t = 0 ;
		p = n ; 
			
		for(int j = 1 ; t>=1 ; j++) {
				
			rev_t += ( p >> 1 ) * ( t  & 1);
			p = p >> 1 ; 
			t = t >> 1 ;
				
		}
		
		if( t1 < rev_t ){
				
			temp = x_r_d[ t1 ]; 
			x_r_d[ t1 ] = x_r_d[ rev_t ]  ; 
			x_r_d[ rev_t] = temp ; 
			temp= x_i_d[ t1 ]; 
			x_i_d[ t1 ] = x_i_d[ rev_t ]  ; 
			x_i_d[ rev_t ] = temp ;   
				
		}
		
		t1 = t1+1 ; 
		rev_t=(rev_t+(n>>1));
		if( (t1 < rev_t) & (rev_t<n)){
				
			temp = x_r_d[ t1 ]; 
			x_r_d[ t1 ] = x_r_d[ rev_t ]  ; 
			x_r_d[ rev_t] = temp ; 
			temp= x_i_d[ t1 ]; 
			x_i_d[ t1 ] = x_i_d[ rev_t ]  ; 
			x_i_d[ rev_t ] = temp ; 
				
		}
		
	}
	
}

//-----------------------------------------------------------------------------

__global__ void fftr2(float* x_r_d, float* x_i_d,const unsigned int n,const unsigned int h) {
	
	float w_r, w_i, z_r1, z_i1, z_r0, z_i0 ;
	int i, start, end ; 
		
	i = (by * 8192 ) + ( bx * 128 ) + ty * 4 + tx ;
		 
	start = ( i % h ) +  ( i / h ) * ( h << 1 ) ; 
	end = start + h ; 
		
	z_r0 = (float) ( i % h ) / (h << 1)  ; 
	z_r0 = (2 * PI) * z_r0 ;
	
	w_r =  cos(z_r0);
	w_i = -sin(z_r0);
       
	z_r0 = x_r_d[start] ; 
	z_i0 = x_i_d[start] ; 
	
	z_r1 = w_r * x_r_d[end] - w_i * x_i_d[end];
	z_i1 = w_r * x_i_d[end] + w_i * x_r_d[end];	
		
	x_r_d[start] = z_r0 + z_r1;
	x_r_d[end] = z_r0 - z_r1;
		
	x_i_d[start] = z_i0 + z_i1;
	x_i_d[end] = z_i0 - z_i1; 
				
}
//-----------------------------------------------------------------------------
__global__ void kernelorderr4(float* x_r_d, float* x_i_d, const unsigned int n) {

	float temp ;
	int t, t1, rev_t, p ,i;
	i = (by * 8192 ) + ( bx * 128 ) + tx ;
	 

	for(int k = 0 ; k < rate2 ; k = k + 4 ){
			
		t =( i << rate ) + k ; 
		t1 = t ; 
		rev_t = 0 ;
		p = n ; 
			
		for(int j = 1 ; t>=1 ; j++) {
				
			rev_t += ( p >> 2 ) * ( t  & 3 );
			p = p >> 2 ; 
			t = t >> 2 ;
				
		}
		
		if( t1 < rev_t ){
				
			temp = x_r_d[ t1 ]; 
			x_r_d[ t1 ] = x_r_d[ rev_t ]  ; 
			x_r_d[ rev_t] = temp ; 
			temp= x_i_d[ t1 ]; 
			x_i_d[ t1 ] = x_i_d[ rev_t ]  ; 
			x_i_d[ rev_t ] = temp ;   
				
		}
		for(int u = 1 ; u < 4 ; u++){
		t1 = t1+ 1 ; 
		rev_t= rev_t+(n>>2);
		if( (t1 < rev_t) & (rev_t<n)){
				
			temp = x_r_d[ t1 ]; 
			x_r_d[ t1 ] = x_r_d[ rev_t ]  ; 
			x_r_d[ rev_t] = temp ; 
			temp= x_i_d[ t1 ]; 
			x_i_d[ t1 ] = x_i_d[ rev_t ]  ; 
			x_i_d[ rev_t ] = temp ; 
				
		}
		}
	}
		
}
//-----------------------------------------------------------------------------
__global__ void fftr4(float* x_r_d, float* x_i_d,const unsigned int n,const unsigned int m) {
	
	float z_i0 , z_i1 , z_i2 , z_i3 , /*z_r0,*/  z_r1 , z_r2 , z_r3 ,w1_i, w1_r , w2_i, w2_r  , w3_i , w3_r , a1  , a2  , a3  ; 
	int i, q, s0 ;
	
	i = (by * 8192 ) + ( bx * 128 ) + (ty * 4) + tx ;
	q = i % m ;
	s0 = ( q )  + ( i / m ) * ( m << 2 ); 
	
	
		a1 = (float)(6.2831853) * (float)q  / (m << 2)  ; 
		a2 = a1 + a1 ; 
		a3 = a2 + a1 ; 

		w1_r =  cos(a1);
		w1_i = -sin(a1);
		w2_r =  cos(a2);
		w2_i = -sin(a2);
		w3_r =  cos(a3);
		w3_i = -sin(a3);
		
	
		//z_r0 = x_r_d[s0] ;
		z_i0 = x_i_d[s0] ;
		
		z_r1 = x_r_d[s0 + m] * w1_r - x_i_d[s0 + m] * w1_i ; 
		z_i1 = x_i_d[s0 + m] * w1_r + x_r_d[s0 + m] * w1_i ; 
		
		z_r2 = x_r_d[s0 + 2*m] * w2_r - x_i_d[s0 + 2*m] * w2_i ; 
		z_i2 = x_i_d[s0 + 2*m] * w2_r + x_r_d[s0 + 2*m] * w2_i ;
		
		z_r3 = x_r_d[s0 + 3*m] * w3_r - x_i_d[s0 + 3*m] * w3_i ; 
		z_i3 = x_i_d[s0 + 3*m] * w3_r + x_r_d[s0 + 3*m] * w3_i ; 
		
		a1 = z_i1 - z_i3 ;
		a2 = x_r_d[s0] - z_r2 ;
		x_r_d[s0 + m] =  a2 + a1; 
		x_r_d[s0 + 3 *m] =  a2 - a1 ; 
		a1 = x_r_d[s0] + z_r2 ;
		a2 = z_r1 +   z_r3;
		x_r_d[s0] =  a1 + a2; 
		x_r_d[s0 + 2*m] =  a1 - a2 ; 
		
		a1 = z_i0 + z_i2 ;
		a2 = z_i1 +   z_i3;
		x_i_d[s0] =  a1 + a2 ; 
		x_i_d[s0 + 2*m] =  a1 -a2;
		
		a1 = z_r1 -z_r3;
		a2 = z_i0 - z_i2 ;
		x_i_d[s0 + m] =  a2 - a1 ;
		x_i_d[s0 + 3*m] =  a2+ a1 ;
	    
}
/////
__global__ void kernelorderr2m(float* x_r_d, float* x_i_d, const unsigned int n) {
	
	float temp ;
	int t, t1, rev_t, p ,i ;
	i = 1024 * bx + tx; ;
	 

	for(int k = 0 ; k < rate2 ; k = k + 2 ){
			
		t =( i << rate ) + k ; 
		t1 = t ; 
		rev_t = 0 ;
		p = n ; 
			
		for(int j = 1 ; t>=1 ; j++) {
				
			rev_t += ( p >> 1 ) * ( t  & 1);
			p = p >> 1 ; 
			t = t >> 1 ;
				
		}
		
		if( t1 < rev_t ){
				
			temp = x_r_d[ t1 ]; 
			x_r_d[ t1 ] = x_r_d[ rev_t ]  ; 
			x_r_d[ rev_t] = temp ; 
			temp= x_i_d[ t1 ]; 
			x_i_d[ t1 ] = x_i_d[ rev_t ]  ; 
			x_i_d[ rev_t ] = temp ;   
				
		}
		
		t1 = t1+1 ; 
		rev_t=(rev_t+(n>>1));
		if( (t1 < rev_t) & (rev_t<n)){
				
			temp = x_r_d[ t1 ]; 
			x_r_d[ t1 ] = x_r_d[ rev_t ]  ; 
			x_r_d[ rev_t] = temp ; 
			temp= x_i_d[ t1 ]; 
			x_i_d[ t1 ] = x_i_d[ rev_t ]  ; 
			x_i_d[ rev_t ] = temp ; 
				
		}
		
	}
	
}

//-----------------------------------------------------------------------------

__global__ void fftr2m(float* x_r_d, float* x_i_d,const unsigned int n,const unsigned int h) {
	
	float w_r, w_i, z_r1, z_i1, z_r0, z_i0 ;
	int i, start, end ; 
		
	i = 1024 * bx + tx;
		 
	start = ( i % h ) +  ( i / h ) * ( h << 1 ) ; 
	end = start + h ; 
		
	z_r0 = (float) ( i % h ) / (h << 1)  ; 
	z_r0 = (2 * PI) * z_r0 ;
	
	w_r =  cos(z_r0);
	w_i = -sin(z_r0);
       
	z_r0 = x_r_d[start] ; 
	z_i0 = x_i_d[start] ; 
	
	z_r1 = w_r * x_r_d[end] - w_i * x_i_d[end];
	z_i1 = w_r * x_i_d[end] + w_i * x_r_d[end];	
		
	x_r_d[start] = z_r0 + z_r1;
	x_r_d[end] = z_r0 - z_r1;
		
	x_i_d[start] = z_i0 + z_i1;
	x_i_d[end] = z_i0 - z_i1; 
				
}
//-----------------------------------------------------------------------------
__global__ void kernelorderr4m(float* x_r_d, float* x_i_d, const unsigned int n) {

	float temp ;
	int t, t1, rev_t, p ,i;
	i = 1024 * bx + tx;
	 

	for(int k = 0 ; k < rate2 ; k = k + 4 ){
			
		t =( i << rate ) + k ; 
		t1 = t ; 
		rev_t = 0 ;
		p = n ; 
			
		for(int j = 1 ; t>=1 ; j++) {
				
			rev_t += ( p >> 2 ) * ( t  & 3 );
			p = p >> 2 ; 
			t = t >> 2 ;
				
		}
		
		if( t1 < rev_t ){
				
			temp = x_r_d[ t1 ]; 
			x_r_d[ t1 ] = x_r_d[ rev_t ]  ; 
			x_r_d[ rev_t] = temp ; 
			temp= x_i_d[ t1 ]; 
			x_i_d[ t1 ] = x_i_d[ rev_t ]  ; 
			x_i_d[ rev_t ] = temp ;   
				
		}
		for(int u = 1 ; u < 4 ; u++){
		t1 = t1+ 1 ; 
		rev_t= rev_t+(n>>2);
		if( (t1 < rev_t) & (rev_t<n)){
				
			temp = x_r_d[ t1 ]; 
			x_r_d[ t1 ] = x_r_d[ rev_t ]  ; 
			x_r_d[ rev_t] = temp ; 
			temp= x_i_d[ t1 ]; 
			x_i_d[ t1 ] = x_i_d[ rev_t ]  ; 
			x_i_d[ rev_t ] = temp ; 
				
		}
		}
	}
		
}
//-----------------------------------------------------------------------------
__global__ void fftr4m(float* x_r_d, float* x_i_d,const unsigned int n,const unsigned int m) {
	
	float z_i0 , z_i1 , z_i2 , z_i3 , /*z_r0,*/  z_r1 , z_r2 , z_r3 ,w1_i, w1_r , w2_i, w2_r  , w3_i , w3_r , a1  , a2  , a3  ; 
	int i, q, s0 ;
	
	i = 1024 * bx + tx;
	q = i % m ;
	s0 = ( q )  + ( i / m ) * ( m << 2 ); 
	
	
		a1 = (float)(6.2831853) * (float)q  / (m << 2)  ; 
		a2 = a1 + a1 ; 
		a3 = a2 + a1 ; 

		w1_r =  cos(a1);
		w1_i = -sin(a1);
		w2_r =  cos(a2);
		w2_i = -sin(a2);
		w3_r =  cos(a3);
		w3_i = -sin(a3);
		
	
		//z_r0 = x_r_d[s0] ;
		z_i0 = x_i_d[s0] ;
		
		z_r1 = x_r_d[s0 + m] * w1_r - x_i_d[s0 + m] * w1_i ; 
		z_i1 = x_i_d[s0 + m] * w1_r + x_r_d[s0 + m] * w1_i ; 
		
		z_r2 = x_r_d[s0 + 2*m] * w2_r - x_i_d[s0 + 2*m] * w2_i ; 
		z_i2 = x_i_d[s0 + 2*m] * w2_r + x_r_d[s0 + 2*m] * w2_i ;
		
		z_r3 = x_r_d[s0 + 3*m] * w3_r - x_i_d[s0 + 3*m] * w3_i ; 
		z_i3 = x_i_d[s0 + 3*m] * w3_r + x_r_d[s0 + 3*m] * w3_i ; 
		
		a1 = z_i1 - z_i3 ;
		a2 = x_r_d[s0] - z_r2 ;
		x_r_d[s0 + m] =  a2 + a1; 
		x_r_d[s0 + 3 *m] =  a2 - a1 ; 
		a1 = x_r_d[s0] + z_r2 ;
		a2 = z_r1 +   z_r3;
		x_r_d[s0] =  a1 + a2; 
		x_r_d[s0 + 2*m] =  a1 - a2 ; 
		
		a1 = z_i0 + z_i2 ;
		a2 = z_i1 +   z_i3;
		x_i_d[s0] =  a1 + a2 ; 
		x_i_d[s0 + 2*m] =  a1 -a2;
		
		a1 = z_r1 -z_r3;
		a2 = z_i0 - z_i2 ;
		x_i_d[s0 + m] =  a2 - a1 ;
		x_i_d[s0 + 3*m] =  a2+ a1 ;
	    
}
//------------------------------------------------------------------------------
__global__ void zojfard(float* x_r_d, float* temp, const unsigned int n) {

	int i = bx * 1024 + tx ; 
	temp[i] = x_r_d[2*i];
	temp[n/2+i]= x_r_d[2*i+1];
		
}
__global__ void cpy(float* x_r_d, float* temp, const unsigned int n) {

	int i = bx * 1024 + tx ; 
	temp[2*i] = x_r_d[2*i];
	temp[2*i+1]= x_r_d[2*i+1];
		
}
void gpuKernel(float* x_r_d, float* x_i_d, const unsigned int n, const unsigned int M)
{ 
	dim3 gridsizer2(64,n/16384);
	dim3 gridsizer4(64,n/32768);
	dim3 gridsize1(64,n/(64*128*rate2));
	dim3 blk(4,32);
	
	if( M & 1 ){
		
		float* temp ; 
		HANDLE_ERROR(cudaMalloc((void**)&temp, n * sizeof(float)));
		zojfard<<<n/2048,1024>>>(x_r_d,temp,n);
		zojfard<<<n/2048,1024>>>(x_i_d,x_r_d,n);
		kernelorderr4m<<<n/8192,1024>>>(temp, x_r_d,n/2);
		kernelorderr4m<<<n/8192,1024>>>(temp+(n/2), x_r_d+(n/2),n/2);
		for(int h = 1 ; h < n/2 ; h = h << 2)
			fftr4m<<<n/8192,1024>>>(temp, x_r_d,n/2,h); 
		for(int h = 1 ; h < n/2 ; h = h << 2)
			fftr4m<<<n/8192,1024>>>(temp+(n/2), x_r_d+(n/2),n/2,h); 
		fftr2m<<<n/2048,1024>>>(temp, x_r_d,n,n/2); 
		cpy<<<n/2048,1024>>>(x_r_d,x_i_d,n);
		cpy<<<n/2048,1024>>>(temp,x_r_d,n); 
		
	} 
	else{
		
		kernelorderr4<<<gridsize1,128>>>(x_r_d, x_i_d,n);

         for(int h = 1 ; h < n ; h = h << 2)
			fftr4<<<gridsizer4,blk>>>(x_r_d, x_i_d,n,h); 
		
	}
	
}