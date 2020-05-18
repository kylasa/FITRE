
#include <utilities/reduce.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>

/*
__device__ __inline__ double my_shfl(double x, int lane)
{
        // Split the double number into 2 32b registers.
        int lo, hi; 
        asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x));

        // Shuffle the two 32b registers.
        lo = __shfl_xor(lo, lane);
        hi = __shfl_xor(hi, lane);

        // Recreate the 64b number.
        //asm volatile( "mov.b64 %0, {%1,%2};" : "=d(x)" : "r"(lo), "r"(hi));
        //return x;
        return __hiloint2double( hi, lo);
}

__device__ __inline__ double warpSum( double x )
{
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
                x += my_shfl( x, offset);
        return x;
}
*/

GLOBAL void reduce_shared(const real *input, real *results, const size_t count) {
        extern __shared__ real my_results[];
        unsigned int lane = threadIdx.x >> 5;
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

        real sdata;
        real x = 0;

        sdata = 0;
        my_results[ lane ] = 0;
        if(idx < count) x = input [idx];
        sdata = x;

        sdata = warpSum ( sdata );
        if (threadIdx.x % WARP_SIZE == 0) my_results[lane] = sdata;
        __syncthreads ();

			//sdata = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? my_results[ threadIdx.x ] : 0; 
         //sdata = (threadIdx.x < 1) ? my_results[threadIdx.x] : 0;
         sdata = (lane < 1) ? my_results[threadIdx.x] : 0;
        __syncthreads ();

        if (lane == 0) sdata = warpSum( sdata );
        if(threadIdx.x == 0) results [ blockIdx.x  ] =  sdata;
}

//results is pageLockPtr... 
void myreduce( real *input, int count, real *devPtr, real *pagePtr )
{
	real *curInput = input; 
	real *curDevPtr = devPtr; 

	int blocks = count;
	int elements = count; 

	while ( blocks > 1 ){ 
		
		blocks = ( blocks + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 
		reduce_shared <<< blocks, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 
			( curInput, curDevPtr, elements ); 
		cudaThreadSynchronize (); 	
		cudaCheckError (); 

		elements = blocks;
		curInput = curDevPtr; 
		curDevPtr = curDevPtr + blocks; 
	}

	reduce_shared <<< 1, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 
		( curInput, pagePtr, elements ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}
