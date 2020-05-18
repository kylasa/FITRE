
#include <drivers/pool_driver.h>

#include <core/errors.h>

#include <functions/dev_pool.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>
#include <device/gen_random.h>

#include <nn/read_nn.h>
#include <nn/nn_decl.h>

#include <utilities/print_utils.h>
#include <utilities/utils.h>



#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define HEIGHT 2
#define WIDTH 2

#define POINTS 1

#define OUT_HEIGHT 1
#define OUT_WIDTH 1

#define OUT_CHANNELS 8

#define KERNEL 3
#define PADDING 1
#define STRIDE 2


void initPoolWeights( real *weights, real *host, int h, int w )
{
	for (int c = 0; c < OUT_CHANNELS; c ++ ){
		for (int i = 0; i < h * w * POINTS; i ++) {
			host[ c * h * w * POINTS + i ] = 1;
		}
	}	

	copy_host_device( host, weights, sizeof(real) * POINTS * w * h * OUT_CHANNELS, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
}

void printPoolWeights( real *weights, real *host, int before )
{

	int rowsLimit = HEIGHT * WIDTH * POINTS; 
	//if (rowsLimit > 10) rowsLimit = 10; 

	int colsLimit = OUT_CHANNELS; 
	//if (colsLimit > 4) colsLimit = 4; 

	int limit = OUT_HEIGHT * OUT_WIDTH * POINTS; 
	//if (limit > 10) limit = 10; 

	copy_host_device( host, weights, sizeof(real) * POINTS * OUT_CHANNELS * HEIGHT * WIDTH, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
	print2DMatrix( host, POINTS * HEIGHT * WIDTH, OUT_CHANNELS ); 
	
	/*
	if (before) {
		copy_host_device( host, weights, sizeof(real) * POINTS * WIDTH * HEIGHT * OUT_CHANNELS, 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 

		for (int i = 0; i < rowsLimit ; i ++) {
			for (int c = 0; c < colsLimit; c ++ ){
				fprintf( stderr, "%4.2f ", host[ i + c * POINTS * HEIGHT * WIDTH ] ); 
			}
			fprintf( stderr, "\n");
		}	
	} else {
		copy_host_device( host, weights, sizeof(real) * POINTS * OUT_WIDTH * OUT_HEIGHT * OUT_CHANNELS, 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 

		for (int i = 0; i < limit; i ++) {
			for (int c = 0; c < colsLimit; c ++ ){
				fprintf( stderr, "%4.2f ", host[ i + c * POINTS * OUT_HEIGHT * OUT_WIDTH ] ); 
			}
			fprintf( stderr, "\n");
		}	
	}
	*/
}

void testPoolDerivative ( CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch )
{

	real start, total; 

	fprintf( stderr, "Test case begin... \n"); 

	/*
	cuda_malloc( (void **)&data->weights, sizeof(real) * POINTS * OUT_HEIGHT * OUT_WIDTH * OUT_CHANNELS , 0, ERROR_MEM_ALLOC ); 

	//init weights here. 
	initPoolWeights( data->weights, scratch->hostWorkspace, OUT_HEIGHT, OUT_WIDTH ); 
	printPoolWeights( data->weights, scratch->hostWorkspace, 0); 
	cuda_memset( scratch->devWorkspace, 0, sizeof(real) * HEIGHT * WIDTH * OUT_CHANNELS * POINTS, 
		ERROR_MEMSET ); 
	*/

	/*
	computePoolDerivative_in( data->weights, OUT_HEIGHT, HEIGHT, OUT_CHANNELS, 
		scratch->devWorkspace, KERNEL, STRIDE, PADDING, POINTS ); 
	*/

	real *devPtr = scratch->nextDevPtr; 
	real *delta, *z_in, *output; 
	delta = devPtr; 
	z_in = delta + POINTS * OUT_HEIGHT * OUT_WIDTH * OUT_CHANNELS; 
	output = z_in + POINTS * HEIGHT * WIDTH * OUT_CHANNELS; 

	cuda_memset( output, 0, sizeof(real) * HEIGHT * WIDTH * POINTS * OUT_CHANNELS, ERROR_MEMSET ); 

	//initPoolWeights( delta, scratch->hostWorkspace, OUT_HEIGHT, OUT_WIDTH ); 
   getRandomVector( POINTS * OUT_HEIGHT * OUT_WIDTH * OUT_CHANNELS, NULL, delta, RAND_NORMAL );

	//initPoolWeights( z_in, scratch->hostWorkspace, HEIGHT, WIDTH ); 
   getRandomVector( POINTS * HEIGHT * WIDTH * OUT_CHANNELS, NULL, z_in, RAND_NORMAL );

	computeMaxPoolDerivative( delta, z_in, OUT_CHANNELS, HEIGHT, WIDTH, KERNEL, STRIDE, PADDING, 
										OUT_HEIGHT, OUT_WIDTH, POINTS, output ); 

	fprintf( stderr, "Z_IN... \n"); 
	printPoolWeights( z_in, scratch->hostWorkspace, 1 ); 

	fprintf( stderr, "Delta... \n"); 
	printPoolWeights( delta, scratch->hostWorkspace, 1); 

	fprintf( stderr, "After reshaping... the result is.... \n"); 
	printPoolWeights( output, scratch->hostWorkspace, 1 ); 


	fprintf( stderr, "Test case end... MAX_POOL_DERIVATIVE.MAX_POOL_DERIVATIVE...\n"); 
}

void testPoolForwardPass( CNN_MODEL *model, DEVICE_DATASET *data, SCRATCH_AREA *scratch )
{
	fprintf( stderr, "Test case Begin (ForwardPass).... \n"); 
	cuda_malloc( (void **)&data->weights, sizeof(real) * POINTS * HEIGHT * WIDTH * OUT_CHANNELS, 0, ERROR_MEM_ALLOC ); 

	//init weights here. 
	initPoolWeights( data->weights, scratch->hostWorkspace, HEIGHT, WIDTH ); 
	printPoolWeights( data->weights, scratch->hostWorkspace, 1 ); 
	cuda_memset( scratch->devWorkspace, 0, sizeof(real) * HEIGHT * WIDTH * OUT_CHANNELS * POINTS, 
		ERROR_MEMSET ); 

	//Forward Pass here. 
	applyPoolLayer( data->weights, POINTS, OUT_CHANNELS, HEIGHT, WIDTH, 
		KERNEL, STRIDE, PADDING, AVG_POOL, scratch->devWorkspace, 1. ); 

	fprintf( stderr, "After the forwardPass.... \n" ); 
	printPoolWeights( scratch->devWorkspace, scratch->hostWorkspace, 0 ); 

	fprintf( stderr, "Done with forwradPass.... \n\n\n"); 
}
