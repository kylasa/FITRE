
#include <augmentation/dev_rotate.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void ker_rotate_right ( real *input, 
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
                     col * height + row;
         targetLoc = imgIdx * channels * height * width + 
                     chId * height * width + 
							(height - 1 - row )	* height + col;

         output[ targetLoc ] = input[ srcLoc ];  
   } 
}

GLOBAL void ker_rotate_left ( real *input, 
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
						col * height + row;

			targetLoc = imgIdx * channels * height * width + 
							chId * height * width + 
							row * height + (width - 1 - col);

			output[ targetLoc ] = input[ srcLoc ]; 

	}
}

GLOBAL void ker_random_rotate ( real *input, 
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
						col * height + row;

			if (probs[ imgIdx ] < 0.33) { 
				//Rotate Left
				targetLoc = chId * numImages * height * width + 
							imgIdx * height * width + 
							row * height + (width - 1 - col);
			} else if (probs[ imgIdx ] > 0.66 ){ 
				// Rotate Right
         	targetLoc = chId * numImages * height * width + 
                     imgIdx * height * width + 
							(height - 1 - row )	* height + col;
			}  else { 
				//Nothing... 
				targetLoc = srcLoc; 
			} 

			output[ targetLoc ] = input[ srcLoc ]; 
	}
}


void rotate( real *input, real *output, int samples, int channels, 
	int height, int width, real *probs, real *devPtr, real *hostPtr )
{
	int blocks = ( channels * height * width + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	dim3 blockXYZ( blocks, samples, 1); 

	ker_random_rotate <<< blockXYZ, BLOCK_SIZE >>> 
		( input, samples, height, width, channels, output, probs ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 
}
