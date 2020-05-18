
#include <drivers/reshape_driver.h>

#include <core/errors.h>

#include <functions/dev_backprop_convolution.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>

#include <nn/read_nn.h>

#include <utilities/print_utils.h>
#include <utilities/utils.h>


#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define HEIGHT 6
#define WIDTH 6
#define CHANNELS 3

#define OUT_CHANNELS 6
#define KERNEL 3
#define POOL_KERNEL 2
#define POINTS 1

#define OUT_HEIGHT 4
#define OUT_WIDTH 4

#define POOL_HEIGHT 2
#define POOL_WIDTH 2

#define PADDING 0
#define STRIDE 1


void initReshapeWeights( real *weights, int ch, int k, int out_ch, 
	real *host )
{
	for (int c = 0; c < out_ch; c ++ ){
		for (int i = 0; i < ch * k * k; i ++) {
			host[ c * ch * k * k + i ] = c+1; 
		}
	}	

	copy_host_device( host, weights, sizeof(real) * ch * k * k * out_ch, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	for (int i = 0; i < out_ch; i ++) 
		host[ i ] = (i+1) * 100 ; 

	copy_host_device( host, weights + ch * k * k * out_ch, sizeof(real) * out_ch, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
}

void printReshapeWeights( real *weights, int size, real *host )
{
	copy_host_device( host, weights, sizeof(real) * (size + OUT_CHANNELS), 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 

	for (int i = 0; i < size; i ++ )
		fprintf( stderr, "%6f ", host[ i ] ); 
	fprintf( stderr, "\n\n" ); 
}

void testReshape ( CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch )
{

	real start, total; 

	fprintf( stderr, "Test case begin... \n"); 

	cuda_malloc( (void **)&data->weights, sizeof(real) * OUT_CHANNELS * HEIGHT * WIDTH * CHANNELS, 0, 
				ERROR_MEM_ALLOC ); 

	//init weights here. 
	initReshapeWeights( data->weights, CHANNELS, KERNEL, OUT_CHANNELS, scratch->hostWorkspace ); 
	printReshapeWeights( data->weights, KERNEL * KERNEL * CHANNELS * OUT_CHANNELS, scratch->hostWorkspace ); 

	reshapeMatrix( data->weights, OUT_CHANNELS, CHANNELS, KERNEL * KERNEL, scratch->devWorkspace ); 
	fprintf( stderr, "After reshaping... the result is.... \n"); 
	printReshapeWeights( scratch->devWorkspace, KERNEL * KERNEL * CHANNELS * OUT_CHANNELS, scratch->hostWorkspace ); 


	fprintf( stderr, "Test case end... \n"); 
}
