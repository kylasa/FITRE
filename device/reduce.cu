
#include <device/warpsum.h>
#include <device/reduce.h>
#include <device/device_defines.h>

GLOBAL void reduce(const real *input, real *results, const size_t count) {
	extern __shared__ real warp_results[];
	unsigned int lane = threadIdx.x >> 5;
	unsigned int idx =  blockDim.x * blockIdx.x + threadIdx.x;

	real sdata;
	real x = 0;
	warp_results[ lane ] = 0;
	sdata = 0;

	for (; idx < gridDim.x * blockDim.x; idx += gridDim.x * blockDim.x ) {
		x = 0; 
		if(idx < count) x = input [idx];
		sdata = x;
		sdata = warpSum ( sdata );
		if (threadIdx.x % WARP_SIZE == 0) warp_results[lane] += sdata;
		__syncthreads ();
	}

	if (lane == 0) { 
		sdata = warp_results[ threadIdx.x ];
		sdata = warpSum( sdata );
		if(threadIdx.x == 0) results [ blockIdx.x  ] =  sdata;
	}
}

GLOBAL void reduce_grid( const real *input, real *results, const size_t count )
{
	//shared memory per block here. 
	extern __shared__ real warp_results[]; 

	//unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x; 
	//unsigned int idx = blockId * blockDim.x + threadIdx.x; 
	unsigned int lane = threadIdx.x >> 5; 

	//int max_loops = (count + offset - 1) / offset;
	//unsigned int stride = gridDim.x * blockDim.x; 
	unsigned int stridedIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	

	real sdata;
	real x = 0;
	warp_results[ lane ] = 0;
	sdata = 0;

	//threads for the entire COUNT no. of inputs. 
	//processed per block.
	//for (; stridedIdx < gridDim.y * gridDim.x * blockDim.x; stridedIdx += stride){
	//for (; stridedIdx < gridDim.y * gridDim.x * blockDim.x; stridedIdx += offset){
	for (; stridedIdx < count; stridedIdx += gridDim.x * blockDim.x){
		x  = 0 ; 
		if (stridedIdx < count) x = input[ stridedIdx ]; 
		sdata = x; 
		sdata = warpSum( sdata ); 
		if (threadIdx.x % WARP_SIZE == 0) warp_results[ lane ] += sdata; 
		__syncthreads ();
	}
	
	if (lane == 0) { 
		sdata = warp_results[ threadIdx.x ];
		sdata = warpSum( sdata );
		if(threadIdx.x == 0) results [ blockIdx.x  ] =  sdata;
	}
}


//Templated version
//template <unsigned int blockSize>
GLOBAL void reduce6( real *in, real *out, unsigned int count)
{
	extern __shared__ real sdata[]; 
	unsigned int tid = threadIdx.x; 
	unsigned int i = blockIdx.x * blockDim.x * 2 + tid; 
	unsigned int gridSize = blockDim.x * gridDim.x * 2; 

	real x = 0;
	sdata[ tid ] = 0; 

	while( i  < count ) {
		sdata[ tid ] += in[ i ] + in[ i + blockDim.x] ;
		i += gridSize; 
	}
	__syncthreads (); 

		if (tid < 512) sdata[ tid ] += sdata[ tid + 512 ]; 
		__syncthreads (); 

		if (tid < 256) sdata[ tid ] += sdata[ tid + 256 ]; 
		__syncthreads (); 

		if (tid < 128) sdata[ tid ] += sdata[ tid + 128 ]; 
		__syncthreads (); 

		if (tid < 64) sdata[ tid ] += sdata[ tid + 64 ]; 
		__syncthreads (); 

	if (tid < 32){
		x = sdata[ tid ] + sdata[ tid + 32 ]; 
		x = warpSum( x ); 	
	}

	if (tid == 0) out[ blockIdx.x ] = x; 
}

GLOBAL void kerNormInf( real *in, real *out, unsigned int count)
{
	extern __shared__ real sdata[]; 
	unsigned int tid = threadIdx.x; 
	unsigned int i = blockIdx.x * blockDim.x + tid; 
	unsigned int gridSize = blockDim.x * gridDim.x; 


	real x = 0;
	sdata[ tid ] = 0; 

	if (i < count) sdata[ tid ] = fabs( in[ i ] ); 
	i += gridSize; 
	__syncthreads (); 

	while( i  < count ) {
		if (sdata[ tid ] < fabs( in[ i ] ) )
			sdata[ tid ] = fabs( in [i] );
		i += gridSize; 
		__syncthreads ();
	}
		if ((tid < 512) && (sdata[ tid ] < sdata[ tid + 512 ])) sdata[ tid ] = sdata[ tid + 512 ]; 
		__syncthreads (); 

		if ((tid < 256) && (sdata[ tid ] < sdata[ tid + 256 ]))  sdata[ tid ] = sdata[ tid + 256 ];
		__syncthreads (); 

		if ((tid < 128) && (sdata[ tid ] < sdata[ tid + 128 ]))  sdata[ tid ] = sdata[ tid + 128 ]; 
		__syncthreads (); 

		if ((tid < 64) && (sdata[ tid ] < sdata[ tid + 64 ]))  sdata[ tid ] = sdata[ tid + 64 ]; 
		__syncthreads (); 

	if (tid < 32){
			if (tid < 32 && sdata[tid] < sdata[ tid + 32 ]) sdata[ tid ] = sdata[ tid + 32 ]; 
			if (tid < 16 && sdata[tid] < sdata[ tid + 16 ]) sdata[ tid ] = sdata[ tid + 16 ]; 
			if (tid < 8 && sdata[tid] < sdata[ tid + 8 ]) sdata[ tid ] = sdata[ tid + 8 ]; 
			if (tid < 4 && sdata[tid] < sdata[ tid + 4 ]) sdata[ tid ] = sdata[ tid + 4 ]; 
			if (tid < 2 && sdata[tid] < sdata[ tid + 2 ]) sdata[ tid ] = sdata[ tid + 2 ]; 
			if (tid < 1 && sdata[tid] < sdata[ tid + 1 ]) sdata[ tid ] = sdata[ tid + 1 ]; 
	}

	if (tid == 0) out[ blockIdx.x ] = sdata[0] ;
}
