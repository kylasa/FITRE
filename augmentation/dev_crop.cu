
#include <augmentation/dev_flip.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>
#include <device/gen_random.h>

#include <core/errors.h>

GLOBAL void ker_random_crop (real *input, 
	int samples, int height, int width, int channels, 
	int *randomX, int *randomY, int padding, real defaultVal, 
	real *output, real *probs ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int imgIdx = blockIdx.y; 

	int count = height * width * channels; 

	int chId, chIdx, row, col, cRow, cCol; 
	real val = -1; 

	if (idx < count){
			chId = idx / (height * width ); 
			chIdx = idx % (height * width );

			row = chIdx % height; 
			col = chIdx / height; 

			//cRow = (probs[ imgIdx ] > 0.5) ? randomX[ imgIdx ] : 0; 
			//cCol = (probs[ imgIdx ] > 0.5) ? randomY[ imgIdx ] : 0; 

			cRow = randomX[ imgIdx ] ; 
			cCol = randomY[ imgIdx ] ; 

			if ( 	( (cRow + row) < 0 )  || ( (cCol + col) < 0 ) || 
					( (cRow + row) >= height) || ((cCol + col) >= width) )
				val = defaultVal;
			else
				val = input[ chId * samples * height * width + 
									imgIdx * height * width + 
									(col + cCol) * height + (row + cRow) ]; 

			output[ chId * samples * height * width + 
						imgIdx * height * width + 
						col * height + row ] = val; 
	}
}

GLOBAL void ker_convert ( real *input, int *output, int samples, int padding )
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 

	if (idx < (2 * samples) )
		output[ idx ] = __double2int_rz( input[ idx ] * 100. ) % ( 2 * padding + 1 ) - padding; 
}

void randomCrop( real *input, int samples ,int height, int width, int channels, 
		int padding, real *output, real *devPtr, real *hostPtr, real defaultValue, real *probs )
{
	//generate random numbers... 
	getRandomVector( 2 * samples, NULL, devPtr, RAND_UNIFORM ); 

	//conversion of random numbers. to range ( -padding, 0, padding )
	int *randomIndices = (int *)(devPtr + 2 * samples ); 
	int blocks = ( 2 * samples + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 
	ker_convert <<< blocks, BLOCK_SIZE >>> 
		( devPtr, randomIndices, samples, padding ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 	

/*
	fprintf( stderr, "Random Indices are as follows\n"); 
	copy_host_device( hostPtr, randomIndices, sizeof(int) * 2 * samples, cudaMemcpyDeviceToHost, 
		ERROR_MEMCPY_DEVICE_HOST); 
	int *src = (int *) hostPtr; 
	for (int i = 0; i < 2 * samples; i ++) 
		fprintf( stderr, " %d ", src[ i ] ); 
	fprintf( stderr, "\n\n" ); 

	cuda_memset( output, 0, sizeof(real) * samples * channels * height * width, ERROR_MEMSET ); 
*/


	blocks = (height * width * channels + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	dim3 blockDims( blocks, samples, 1); 

	ker_random_crop <<< blockDims, BLOCK_SIZE >>> 
		( input, samples, height, width, channels, randomIndices, randomIndices + samples, 
			padding, defaultValue, output, probs); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 	
}
