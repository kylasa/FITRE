
#include <drivers/augmentation_driver.h>

#include <core/errors.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>
#include <device/gen_random.h>

#include <augmentation/dev_flip.h>
#include <augmentation/dev_rotate.h>
#include <augmentation/dev_crop.h>
#include <augmentation/dev_normalize.h>


#include <utilities/print_utils.h>
#include <utilities/utils.h>
#include <utilities/dataset.h>


#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define DATASET_SIZE 500


void convertAugDataset( real *src, real *tgt, int ch, int h, int w)
{
   for (int i = 0; i < ch; i ++){
   	for (int p = 0; p < DATASET_SIZE; p ++) {
      	for (int j = 0; j < h * w; j ++) {
				tgt[ i * h * w * DATASET_SIZE + 
						p * h * w + 
						j ] = src[ p * h * w * ch + 
										i * h * w + 
										j ]; 	
      	}
   	}
   }
}

void initAugCNNDataset( DEVICE_DATASET *data, SCRATCH_AREA *scratch,
   int h, int w, int ch, int k, int out_ch, int numClasses )
{  
   int points = DATASET_SIZE; ;
   real counter=1;
	real val = 1; 
   
   real *host = scratch->hostWorkspace;
   real *dev = scratch->devWorkspace;
	real *batchMajorDataset = host + h * w * ch * DATASET_SIZE; 
   
   for (int p = 0; p < points; p ++) {
		counter = 0; 
   	for (int i = 0; i < ch; i ++){
			counter ++; 
      	for (int c = 0; c < w; c ++) { 
      		for (int r = 0; r < h; r ++) {
            	host[ i * h * w * points + h * c + r + p * h * w ] = c + 1; 
         	}
      	}
   	}
   }


   
   cuda_malloc( (void **)&data->trainSetX, sizeof(real) * ch * h * w * points, 0,
            ERROR_MEM_ALLOC );

	//convertAugDataset( host, batchMajorDataset, ch, h, w ); 

   copy_host_device( host, data->trainSetX, sizeof(real) * ch * h * w * points,
         cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );

	for (int p = 0; p < points; p ++) host[ p ] = 1; 
   cuda_malloc( (void **)&data->trainSetY, sizeof(real) * points, 0,
            ERROR_MEM_ALLOC ); 
   copy_host_device( host, data->trainSetY, sizeof(real) * points,
         cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );

   data->trainSizeX = points;
   data->trainSizeY = points;
   data->numClasses = numClasses;
}


void testAugmentation( CNN_MODEL *model, DEVICE_DATASET *data, 
					SCRATCH_AREA *scratch )
{
	int HEIGHT = 32; 
	int WIDTH = 32; 
	int CHANNELS = 3; 
	int OUT_CHANNELS = 1; 
	int NUM_CLASSES = 10; 
	int KERNEL = 1; 
	int PADDING = 4; 
	
	//fprintf( stderr, "Dataset Augmentation Test: \n"); 
	//initAugCNNDataset( data, scratch, HEIGHT, WIDTH, CHANNELS, KERNEL, OUT_CHANNELS, NUM_CLASSES); 	
	//fprintf( stderr, "Dataset initialized \n"); 

	real *nextDevPtr = scratch->nextDevPtr; 
	real *hostPtr = scratch->nextHostPtr; 

	fprintf( stderr, "Initial Dataset ... \n\n"); 
	//copy_host_device( scratch->hostWorkspace, data->trainSetX, sizeof(real) * DATASET_SIZE * HEIGHT * WIDTH * CHANNELS, 
	//		cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	//writeIntMatrix( scratch->hostWorkspace, 10, HEIGHT * WIDTH * CHANNELS ); 
	//print2DMatrix( scratch->hostWorkspace, DATASET_SIZE * WIDTH * HEIGHT, CHANNELS); 
	//print4DMatrix( scratch->hostWorkspace, DATASET_SIZE , CHANNELS , HEIGHT, WIDTH); 

	/*
	// Normalization Here. 
	normalizeCIFAR10( data->trainSetX, DATASET_SIZE, CHANNELS, HEIGHT, WIDTH, 
		nextDevPtr, hostPtr); 


	fprintf( stderr, "Normalized Dataset ... \n\n"); 
	copy_host_device( scratch->hostWorkspace, data->trainSetX, sizeof(real) * DATASET_SIZE * HEIGHT * WIDTH * CHANNELS, 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	print2DMatrix( scratch->hostWorkspace, DATASET_SIZE * WIDTH * HEIGHT, CHANNELS); 
	//print4DMatrix( scratch->hostWorkspace, DATASET_SIZE , CHANNELS, HEIGHT, WIDTH); 
	*/


	//Random Cropping here. 
	//randomCrop( data->trainSetX, DATASET_SIZE, HEIGHT, WIDTH, CHANNELS, PADDING, nextDevPtr,
	//				nextDevPtr + DATASET_SIZE * CHANNELS * HEIGHT * WIDTH, hostPtr, 0, NULL );
	//flipData( data->trainSetX, nextDevPtr, DATASET_SIZE, CHANNELS, HEIGHT, WIDTH, NULL, NULL, NULL ); 
	rotate( data->trainSetX, nextDevPtr, DATASET_SIZE, CHANNELS, HEIGHT, WIDTH, NULL, NULL, NULL ); 

	fprintf( stderr, "Results after cropping... \n" ); 
	copy_host_device( scratch->hostWorkspace, nextDevPtr, sizeof(real) * DATASET_SIZE * HEIGHT * WIDTH * CHANNELS, 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	writeIntMatrix( scratch->hostWorkspace, 10, HEIGHT * WIDTH * CHANNELS ); 
	fprintf( stderr, "Done creating the augmented file... \n"); 
	//print4DMatrix( scratch->hostWorkspace, DATASET_SIZE, CHANNELS, HEIGHT, WIDTH ); 
	

	/*
	int x_blocks = ( CHANNELS * HEIGHT * WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE; 
   dim3 blocks(1, DATASET_SIZE, x_blocks);  


	// Flip - vertical
	ker_vertical_flip <<< blocks, BLOCK_SIZE >>> 
		( data->trainSetX, DATASET_SIZE, HEIGHT, WIDTH, CHANNELS, nextDevPtr ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 

	fprintf( stderr, "vertical flip dataset... \n"); 
	copy_host_device( scratch->hostWorkspace, nextDevPtr, sizeof(real) * DATASET_SIZE * HEIGHT * WIDTH * CHANNELS, 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	print4DMatrix( scratch->hostWorkspace, DATASET_SIZE, CHANNELS, HEIGHT, WIDTH ); 

	// Flip - Horizaontal 
	ker_horizontal_flip <<< blocks, BLOCK_SIZE >>> 
		( data->trainSetX, DATASET_SIZE, HEIGHT, WIDTH, CHANNELS, nextDevPtr ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 

	fprintf( stderr, "Horizontal flip dataset... \n"); 
	copy_host_device( scratch->hostWorkspace, nextDevPtr, sizeof(real) * DATASET_SIZE * HEIGHT * WIDTH * CHANNELS, 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	print4DMatrix( scratch->hostWorkspace, DATASET_SIZE, CHANNELS, HEIGHT, WIDTH ); 

	// Rotate - Horizontal
	ker_rotate_right <<< blocks, BLOCK_SIZE >>> 
		( data->trainSetX, DATASET_SIZE, HEIGHT, WIDTH, CHANNELS, nextDevPtr ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 

	fprintf( stderr, "Rotate Right dataset... \n"); 
	copy_host_device( scratch->hostWorkspace, nextDevPtr, sizeof(real) * DATASET_SIZE * HEIGHT * WIDTH * CHANNELS, 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	print4DMatrix( scratch->hostWorkspace, DATASET_SIZE, CHANNELS, HEIGHT, WIDTH ); 
	

	// Rotate - Left
	ker_rotate_left<<< blocks, BLOCK_SIZE >>> 
		( data->trainSetX, DATASET_SIZE, HEIGHT, WIDTH, CHANNELS, nextDevPtr ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 

	fprintf( stderr, "Rotate Left dataset... \n"); 
	copy_host_device( scratch->hostWorkspace, nextDevPtr, sizeof(real) * DATASET_SIZE * HEIGHT * WIDTH * CHANNELS, 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	print4DMatrix( scratch->hostWorkspace, DATASET_SIZE, CHANNELS, HEIGHT, WIDTH ); 
	*/

}
