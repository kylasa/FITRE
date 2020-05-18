
#include <solvers/kfac_da.h>

#include <core/datadefs.h>
#include <core/errors.h>

#include <nn/nn_decl.h>

#include <functions/dev_backprop_convolution.h>

#include <augmentation/dev_crop.h>
#include <augmentation/dev_flip.h>
#include <augmentation/dev_rotate.h>
#include <augmentation/dev_normalize.h>

#include <device/gen_random.h>
#include <device/cuda_utils.h>

#include <utilities/print_utils.h>


void augmentData( CNN_MODEL *model, DEVICE_DATASET *data, int offset, int currentBatchSize, real *devPtr, real *hostPtr, int enableDA, DATASET_TYPE datasetType )
{

	CONV_LAYER *c = &( model->convLayer[ 0 ] ); 

	int channels = c->inChannels; 
	int height = c->height; 
	int width = c->width; 
	int samples = currentBatchSize; 

	real *first = data->sampledTrainX; 
	real *second = first + samples * channels * height * width; 

	//real *input = data->trainSetX + offset * data->features; 

	reshapeMatrix( first, samples, channels, height * width, second); 


/*
copy_host_device( hostPtr, second, sizeof(real) * channels * height * width * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
	writeIntMatrix( hostPtr, 10, 1024 ); 

	exit( -1 ); 
*/

	if (enableDA >= 1) {
		//Random Crop
   	getRandomVector( samples, NULL, devPtr, RAND_UNIFORM );
		randomCrop( second, samples, height, width, channels, 4, first, devPtr + samples, hostPtr, 0, devPtr); 
	
		//Random Flip
   	getRandomVector( samples, NULL, devPtr, RAND_UNIFORM );
		flipData( first, second, samples, channels, height, width, devPtr, NULL, NULL ); 
	
/*
		//Random rotate
   	getRandomVector( samples, NULL, devPtr, RAND_UNIFORM );
		rotate( second, first, samples, channels, height, width, devPtr, NULL, NULL ); 
*/

		//Normalization Here. 
		normalizeCIFAR10( second, samples, channels, height, width, devPtr, hostPtr, datasetType ); 
		data->currentBatch = second; 
	} else {

/*
		copy_device( first, input, sizeof(real) * channels * height * width * samples, 
			ERROR_MEMCPY_DEVICE_DEVICE ); 

		normalizeCIFAR10( first, samples, channels, height, width, devPtr, hostPtr ); 
		data->currentBatch = first; 
*/
	
		normalizeCIFAR10( second, samples, channels, height, width, devPtr, hostPtr, datasetType ); 
		data->currentBatch = second; 
/*
copy_host_device( hostPtr, second + 500 * 32 * 32 + 499 * 32 * 32, sizeof( real ) * 32 * 32, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
for (int r = 0; r < 32; r ++) {
	for (int c = 0; c < 32; c ++) {
		fprintf( stderr, "%.3f ", hostPtr[ c * 32 + r ]  ); 
	}
	fprintf( stderr, "\n"); 
}
exit( -1 ); 
*/
	}
}
