
#include <drivers/mem_driver.h>

#include <functions/cnn_forward.h>
#include <functions/cnn_backward.h>
#include <functions/cnn_hv_forward.h>
#include <functions/cnn_hv_backward.h>

#include <nn/read_nn.h>

#include <device/cuda_utils.h>
#include <core/errors.h>

void testCudaMemcpy2D( SCRATCH_AREA *scratch )
{
	real *devPtr = scratch->nextDevPtr; 
	real *hostPtr = scratch->nextHostPtr; 

	real *src = devPtr; 
	real *tgt = src + 9; 

	for (int i = 0; i < 9; i ++) 
		hostPtr[ i ] = i + 1; 

	fprintf( stderr, "Source Matrix... \n"); 
	for( int i = 0; i < 3; i ++) {
		for (int j = 0; j < 3; j ++ )
			fprintf( stderr, "%f ", hostPtr[ j * 3 + i ] ); 
		fprintf( stderr, "\n"); 
	}

	fprintf( stderr, "\n\n\n... CUDAMEMCPY2D Testing... \n\n\n"); 

	copy_host_device( hostPtr, src, sizeof(real) * 9, 
		cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	for (int i = 0; i < 12; i ++) 
		hostPtr[ i ] = -1; 

	copy_host_device( hostPtr, tgt, sizeof(real) * 12, cudaMemcpyHostToDevice, 
								ERROR_MEMCPY_HOST_DEVICE ); 

	//cuda_memset( tgt, 0, sizeof(real) * 4 * 3, ERROR_MEMSET ); 

	cudaMemcpy2D( tgt, sizeof(real) * 4, src, sizeof(real) * 3, 
							sizeof(real) * 3, sizeof(real) * 3, cudaMemcpyDeviceToDevice ); 

	copy_host_device( hostPtr, tgt, sizeof(real) * 4 * 3, cudaMemcpyDeviceToHost, 
								ERROR_MEMCPY_DEVICE_HOST ); 

	for( int i = 0; i < 4; i ++) {
		for (int j = 0; j < 3; j ++ )
			fprintf( stderr, "%f ", hostPtr[ j * 4 + i ] ); 
		fprintf( stderr, "\n"); 
	}

	fprintf( stderr, "... Done with the testing... \n\n\n"); 
}

void getMemRequired( CNN_MODEL *model )
{
	fprintf( stderr, "\n\n");
	fprintf( stderr, "*** Memory Requirement Report *** \n\n"); 

	for (int i = 1; i < pow(2, 15); i *= 2 ){ 

   	readLenetCNN( model, 3, 32, 32, i, 1, 0);  

		long forward = cnnForwardMemRequired( model ); 
		long backward = cnnBackwardMemRequired( model ); 
		long hvForward = cnnROpForwardMemRequired( model ); 
		long hvBackward = cnnROpBackwardMemRequired( model ); 

		long gradient = forward + backward; 
		long hv = hvForward + hvBackward + gradient; 
		long total = hv + 3 * model->zSize +  5 * model->maxDeltaSize; 

		fprintf( stderr, "Batch Size: %d\n", i );
		fprintf( stderr, " Gradient Buffer Requirement: (%ld GB, %ld MB, %ld KB, %ld) \n", 	
			(long)(((double)gradient) / (1024 * 1024 * 1024)), 
			(long)(((double)gradient) / (1024 * 1024)), 
			(long)(((double)gradient) / (1024)), 
			gradient ); 
	
		fprintf( stderr, " Hv Buffer Requirement: (%ld GB, %ld MB, %ld KB, %ld) \n\n\n", 
			(long)(((double)hv) / (1024 * 1024 * 1024)), 
			(long)(((double)hv) / (1024 * 1024)), 
			(long)(((double)hv) / (1024)), 
			hv ); 

		fprintf( stderr, " Total Buffer Requirement: (%ld GB, %ld MB, %ld KB, %ld) \n\n\n", 
			(long)(((double)total) / (1024 * 1024 * 1024)), 
			(long)(((double)total) / (1024 * 1024)), 
			(long)(((double)total) / (1024)), 
			total ); 
	}

	fprintf( stderr, "*** End Report *** \n\n"); 
}

