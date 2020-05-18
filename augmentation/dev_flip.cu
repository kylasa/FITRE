
#include <augmentation/dev_flip.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

GLOBAL void ker_vertical_flip (real *input, 
	int numImages, int height, int width, int channels, 
	real *output ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int imgIdx = blockIdx.y; 

	int count = height * width * channels; 

	int chId, chIdx, row, col, srcLoc, targetLoc; 

	if (idx < count){
			chId = idx / (height * width); 
			chIdx = idx % (height * width);

			row = chIdx % height; 
			col = chIdx / height; 

			srcLoc = imgIdx * channels * height * width + 
							chId * height * width + 
							col * height + row ;
			targetLoc = imgIdx * channels * height * width + 
							chId * height * width + 
							col * height + (height - 1 - row); 

			output[ targetLoc ] = input[ srcLoc ]; 
	}
}

GLOBAL void ker_horizontal_flip (real *input, 
	int numImages, int height, int width, int channels, 
	real *output ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int imgIdx = blockIdx.y; 

	int count = height * width * channels; 

	int chId, chIdx, row, col, srcLoc, targetLoc; 

	if (idx < count){
			chId = idx / (height * width); 
			chIdx = idx % (height * width);

			row = chIdx % width; 
			col = chIdx / width; 

			srcLoc = imgIdx * channels * height * width + 
							chId * height * width + 
							col * height + row ;
			targetLoc = imgIdx * channels * height * width + 
							chId * height * width + 
							(width - 1 - col) * height + row; 

			output[ targetLoc ] = input[ srcLoc ]; 
	}
}

GLOBAL void ker_random_flip (real *input, 
	int numImages, int height, int width, int channels, 
	real *output, real *probs ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int imgIdx = blockIdx.y; 

	int count = height * width * channels; 

	int chId, chIdx, row, col, srcLoc, targetLoc; 

	if (idx < count){
			chId = idx / (height * width); 
			chIdx = idx % (height * width);

			row = chIdx % width; 
			col = chIdx / width; 

			srcLoc = chId * numImages * height * width + 
							imgIdx * height * width + 
							col * height + row ;

			targetLoc = srcLoc; 

			if (probs[ imgIdx ] < 0.5) {
				//horizontal flip
				targetLoc = chId * numImages * height * width + 
							imgIdx * height * width + 
							(width - 1 - col) * height + row; 
			} /*else { 
				//vertical flip
				targetLoc = chId * numImages  * height * width + 
							imgIdx * height * width + 
							col * height + (height - 1 - row); 
			}*/

			output[ targetLoc ] = input[ srcLoc ]; 
	}
}

void flipData( real *input, real *output, int samples, int channels, 
	int height, int width, real *probs, real *devPtr, real *hostPtr )
{
	int blocks = (channels * height * width + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	dim3 blockXYZ( blocks, samples, 1); 

	ker_random_flip <<< blockXYZ, BLOCK_SIZE >>> 
		( input, samples, height, width, channels, output, probs ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 
}
	
