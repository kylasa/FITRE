
#include <functions/dev_dist_errors.h>

#include <device/gen_random.h>
#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

#include <core/errors.h>


GLOBAL void ker_compute_dist_classes ( real *probs, 
	int samples, int numClasses, real *distProbs, real *distClasses )
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	
	real c = 0; 

	if (idx < samples) {

		distClasses[ idx ] = -1; 
		for (int i = 0; i < numClasses; i ++) {
			c += probs[ idx * numClasses + i ]; 
	
			if (c >= distProbs[ idx ]){
				distClasses[ idx ] = i + 1; 
				i += (numClasses + 1); 
			}
		}

		if (distClasses[ idx ] == -1) 
			distClasses[ idx ] = numClasses + 1; 
	}
}


// numClasses * samples --> columnMajor Order
void getMultinomialDistSample( real *probs, int samples, int numClasses, 
   real *distClasses, real *devPtr ) {

	int blocks;
	getRandomVector( samples, NULL, devPtr, RAND_UNIFORM ); 
	
	blocks = ( samples + BLOCK_SIZE - 1) / BLOCK_SIZE; 

	ker_compute_dist_classes <<< blocks, BLOCK_SIZE >>> 
		( probs, samples, numClasses, devPtr, distClasses ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 
}
