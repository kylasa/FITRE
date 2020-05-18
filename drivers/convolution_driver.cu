
#include <drivers/convolution_driver.h>

#include <core/errors.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>

#include <nn/read_nn.h>

#include <functions/dev_initializations.h>
#include <functions/eval_convolution.h>
#include <functions/dev_backprop_convolution.h>
#include <functions/cnn_forward.h>
#include <functions/cnn_backward.h>

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
#define POINTS 5

#define OUT_HEIGHT 4
#define OUT_WIDTH 4

#define POOL_HEIGHT 2
#define POOL_WIDTH 2

#define PADDING 0
#define STRIDE 1

void printImg(real *dev, real *host, int size, 
	int points, int channels, int height, int width )
{
	copy_host_device( host, dev, size * sizeof(real), 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	
	fprintf( stderr, "img.... \n\n"); 
	for (int p = 0; p < points; p ++) {
	for (int c = 0; c < channels; c ++){
	for (int i = 0; i < height; i ++){
		for (int j = 0; j < width; j ++){
			fprintf( stderr, "%4f  ", 
						//host[ j * height + i + c * height* width + p * height * width * channels] ); 
						host[ c * points * height * width + p * height * width + j * height + i ] ); 
		}
		fprintf( stderr, "\n"); 
	}
	fprintf( stderr, "====\n"); 
	}
	}
	fprintf( stderr, "\n\n"); 
}

void printImgCol (real *dev, real *host, int size, 
	int points, int out_height, int out_width, int channels, int kernel)
{
	copy_host_device( host, dev, size * sizeof(real), 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 

	fprintf( stderr, "Printing ImgCols output... \n\n"); 

	for (int x = 0; x < points; x ++ ){ 
	for (int j = 0; j < out_height * out_width; j ++ ) {
		for (int i = 0; i < kernel * kernel * channels; i ++) {
			if (((i % kernel * kernel) == 0)) fprintf( stderr, "| " ); 
			fprintf( stderr, "%4.2f  ", 
				host[ x * out_height * out_width + j + i * out_height * out_width ] ); 
		}
		fprintf( stderr, "\n"); 
	}
	fprintf( stderr, "*********\n"); 
	}

	/*
	for (int i = 0; i < OUT_HEIGHT * OUT_WIDTH * KERNEL * KERNEL * CHANNELS; i ++ )
		fprintf( stderr, " %3d", (int) host[ i ] ); 
	fprintf( stderr, "\n"); 
	fprintf( stderr, "\n"); 
	*/
}

void printImgColWeights( real *dev, real *host, int size)
{
	copy_host_device( host, dev, size * sizeof(real), 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 

	fprintf( stderr, "img.... \n\n"); 
	for (int p = 0; p < POINTS; p ++) {
	for (int c = 0; c < OUT_CHANNELS; c ++) {
		for (int j = 0; j < OUT_HEIGHT; j ++){
			for (int i = 0; i < OUT_WIDTH; i ++){
				//fprintf( stderr, "%2.3f ", host[ j * OUT_HEIGHT + i + c * OUT_HEIGHT * OUT_WIDTH ] ); 
				//fprintf( stderr, "%2.3f ", host[ j + i * OUT_WIDTH + c * OUT_HEIGHT * OUT_WIDTH + 
						//p * OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH ] ); 
				//fprintf( stderr, "%2.3f ", host[ (j + p * OUT_HEIGHT * OUT_WIDTH) + (i * POINTS * OUT_HEIGHT * OUT_WIDTH) ] );
				fprintf( stderr, "%2.3f ", host[ j + i * OUT_HEIGHT  + c * (POINTS * OUT_HEIGHT * OUT_WIDTH) + p * (OUT_HEIGHT * OUT_WIDTH)] ); 
			}
			fprintf( stderr, "\n"); 
		}
		fprintf( stderr, "--------------\n"); 
	}
	fprintf( stderr, "\n\n"); 
	} 
	fprintf( stderr, "\n\n"); 

	
	/*
	for (int i = 0; i < OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH; i ++) 
		fprintf( stderr, "%6.2f",  host[ i ] ); 
	fprintf( stderr, "\n\n"); 
	*/	
}

void printPool( real *dev, real *host, int c, int h, int w, int points)
{
	copy_host_device( host, dev, POINTS * (c * h * w)* sizeof(real), 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 

	fprintf( stderr, "img.... \n\n"); 
	for( int p = 0; p < points; p ++) {
	for (int ch = 0; ch < c; ch ++){
		for (int r = 0; r < h; r ++) { 
			for (int col = 0; col < w; col ++ ) { 
				//fprintf( stderr, "%4.2f ", host[ points * ch * h * w + p * h * w + col * h + r ] ); 
				//fprintf( stderr, "%4.2f ", host[ p * c * h * w + ch * h * w + col * h + r ] ); 
				//fprintf( stderr, "%4.2f ", host[ POINTS * ch * h * w + col * h + r ] ); 
				fprintf( stderr, "%4.2f ", host[ ch * points * h * w + p * h * w + col * h + r ] ); 
			}
			fprintf( stderr, "\n" ); 
		}
		fprintf( stderr, "=========\n"); 
	}
	fprintf( stderr, "***********\n") ; 
	}
	fprintf( stderr, "\n\n"); 

	for (int ch = 0; ch < c ; ch ++) { 
		for (int i = 0; i < points * h * w; i ++ )
			fprintf( stderr, "%4.2f ", host[ i + ch * points * h * w ] ); 
		fprintf( stderr, "\n"); 
	}
}

void initDataset( DEVICE_DATASET *data, SCRATCH_AREA *scratch, 
	int h, int w, int ch, int k, int out_ch )
{
	int height_col = ( h - k ) + 1; 
	int width_col = ( w - k ) + 1; 
	int points = POINTS; 
	int counter=0;

	real *host = scratch->hostWorkspace; 
	real *dev = scratch->devWorkspace; 

	counter = 0; 
	for (int p = 0; p < points; p ++) {
	counter = p + 1; 
	for (int i = 0; i < ch; i ++){
		for (int r = 0; r < h; r ++) {
			for (int c = 0; c < w; c ++) { 
				host[ i * h * w * points + h * c + r + p * h * w ] = (counter) ;
			} 
		}
		counter ++; 
	}
	//counter += 10; 
	}

	cuda_malloc( (void **)&data->trainSetX, sizeof(real) * ch * h * w * points, 0, 
				ERROR_MEM_ALLOC ); 

	copy_host_device( host, data->trainSetX, sizeof(real) * ch * h * w * points, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	host[ 0 ] = 1; 
	cuda_malloc( (void **)&data->trainSetY, sizeof(real) * points, 0, 
				ERROR_MEM_ALLOC ); 
	copy_host_device( host, data->trainSetY, sizeof(real) * points, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	data->trainSizeX = points; 
	data->trainSizeY = points;
	data->numClasses = 10; 
}

void initDeltas( real *devData, real *host, 
	int h, int w, int ch ){
	int counter; 

	for (int p = 0; p < POINTS; p ++) {
		for (int i = 0; i < ch; i ++){
			counter = (i + 1) * 10; 
			for (int r = 0; r < h; r ++) {
				for (int c = 0; c < w; c ++) { 
					host[ i * h * w + h * c + r + p * ch * h * w ] = (counter ++) * (p+1); 
				} 
			}
		}
	}
	copy_host_device( host, devData, sizeof(real) * ch * h * w * POINTS, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
}

void initWeights( real *weights, int ch, int k, int out_ch, 
	real *host )
{
	int counter = 1; 
	for (int c = 0; c < out_ch; c ++ ){
		for (int i = 0; i < ch * k * k; i ++) {
			host[ c * ch * k * k + i ] = counter; //(counter ++) * 0.01; 
		}
		counter ++; 	
	}	

	copy_host_device( host, weights, sizeof(real) * ch * k * k * out_ch, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	for (int i = 0; i < out_ch; i ++) 
		host[ i ] = (i+1) * 0.1; 

	copy_host_device( host, weights + ch * k * k * out_ch, sizeof(real) * out_ch, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
}

void printWeights( real *weights, int size, real *host )
{
	copy_host_device( host, weights, sizeof(real) * (size + OUT_CHANNELS), 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 

	for (int i = 0; i < size; i ++ )
		fprintf( stderr, "%6f ", host[ i ] ); 
	fprintf( stderr, "\n\n" ); 

/*
	fprintf( stderr, " BIAS---> \n" ); 
	for (int i = 0; i < OUT_CHANNELS; i ++)
		fprintf( stderr, "%6f ", *(host+ size + i));
*/

	fprintf( stderr, "\n\n" ); 
}

void testConvolution( CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch )
{

	real start, total; 
	int ACTIVATION_OFFSET = OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH * POINTS; 
	int POOL_OFFSET = 2 * ACTIVATION_OFFSET; 
	int BATCH_NORM_OFFSET = POOL_OFFSET + POOL_HEIGHT * POOL_WIDTH * OUT_CHANNELS * POINTS;

	fprintf( stderr, "Test case begin... \n"); 

	cuda_malloc( (void **)&data->weights, sizeof(real) * OUT_CHANNELS * HEIGHT * WIDTH * CHANNELS, 0, 
				ERROR_MEM_ALLOC ); 
	cuda_memset( scratch->devWorkspace, 0, sizeof(real) * OUT_WIDTH * OUT_HEIGHT * KERNEL * KERNEL * CHANNELS, ERROR_MEMSET ); 

	//init weights here. 
	initWeights( data->weights, CHANNELS, KERNEL, OUT_CHANNELS, scratch->hostWorkspace ); 

	//init Dataset here. 
	initDataset( data, scratch, HEIGHT, WIDTH, CHANNELS, KERNEL, OUT_CHANNELS ); 	
	printImg( data->trainSetX, scratch->hostWorkspace, POINTS * CHANNELS * HEIGHT * WIDTH, 
			POINTS, CHANNELS, HEIGHT, WIDTH ); 

	//start timer
	start = Get_Time ();

	applyConvolutionLayer( data->trainSetX, POINTS, 
		CHANNELS, HEIGHT, WIDTH, KERNEL, PADDING, STRIDE, 
		OUT_HEIGHT, OUT_WIDTH,
		data->weights, data->weights + OUT_CHANNELS * CHANNELS * KERNEL * KERNEL * POINTS, 
		scratch->devWorkspace, OUT_CHANNELS, CNN_ACT_SOFTPLUS, 2, 2, 0, AVG_POOL, 
		POOL_HEIGHT, POOL_WIDTH, PERFORM_NO_BATCH_NORM, 0, 
		ACTIVATION_OFFSET, POOL_OFFSET, BATCH_NORM_OFFSET,
		0, 0,
		scratch->devWorkspace + POINTS * OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH, 
		scratch->hostWorkspace, MODEL_TRAIN, 0, 0 ); 

	//end timer
	total = Get_Timing_Info( start );

	fprintf( stderr, "Total time in msecs for convolution layer for batch: %d is: %2.6f\n", 
			POINTS, total * 1000.); 

	/*
	fprintf( stderr, "ImgCols.... \n"); 
	printImgCol( scratch->devWorkspace + POINTS * OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH, scratch->hostWorkspace, OUT_HEIGHT * OUT_WIDTH * KERNEL * KERNEL * CHANNELS * POINTS, POINTS, OUT_HEIGHT, OUT_WIDTH, CHANNELS, KERNEL); 

	fprintf( stderr, "weights.... \n"); 
	printWeights( data->weights, KERNEL * KERNEL * CHANNELS * OUT_CHANNELS, scratch->hostWorkspace ); 

	fprintf( stderr, "ImgColWeights.... \n"); 
	printImgColWeights( scratch->devWorkspace + OUT_HEIGHT * OUT_WIDTH * KERNEL * KERNEL * CHANNELS * POINTS, 
			scratch->hostWorkspace, OUT_HEIGHT * OUT_WIDTH * OUT_CHANNELS * POINTS ); 

	fprintf( stderr, "Pool.... \n"); 
	printPool( scratch->devWorkspace + OUT_HEIGHT * OUT_WIDTH * KERNEL * KERNEL * CHANNELS * POINTS  +  // imgcols
			OUT_HEIGHT * OUT_WIDTH * OUT_CHANNELS * POINTS, // imgcolsweights
			scratch->hostWorkspace, OUT_CHANNELS, POOL_HEIGHT, POOL_WIDTH, POINTS ); 
	*/

	fprintf( stderr, "Test case end... \n"); 
}

void testBackPropConvolution( CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch )
{

	real start, total; 
	real *deltas = scratch->nextDevPtr; 
	real *deltas_1 = deltas + POINTS * OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH; 
	real *nextDevPtr = deltas_1 + POINTS * CHANNELS * HEIGHT * WIDTH; 
	real *host = scratch->nextHostPtr; 

	fprintf( stderr, "Test case begin... \n"); 

	cuda_malloc( (void **)&data->weights, sizeof(real) * OUT_CHANNELS * HEIGHT * WIDTH * CHANNELS, 0, 
				ERROR_MEM_ALLOC ); 


	//init weights here. 
	initWeights( data->weights, CHANNELS, KERNEL, OUT_CHANNELS, scratch->hostWorkspace ); 

	//init Dataset here. 
	initDeltas( deltas, host, OUT_HEIGHT, OUT_WIDTH, OUT_CHANNELS ); 
	printImg( deltas, host, POINTS * OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH, 
			POINTS, OUT_CHANNELS, OUT_HEIGHT, OUT_WIDTH ); 

	cuda_memset( nextDevPtr, 0, sizeof(real) * WIDTH * HEIGHT * KERNEL * KERNEL * CHANNELS, ERROR_MEMSET ); 

	//start timer
	start = Get_Time ();

	backpropConvolution( deltas, OUT_HEIGHT, OUT_WIDTH, OUT_CHANNELS, 
		data->weights, KERNEL, KERNEL, 
		HEIGHT, WIDTH, 0, CHANNELS, POINTS, 
		deltas_1, 	nextDevPtr, host ); 

	//end timer
	total = Get_Timing_Info( start );

	fprintf( stderr, "Total time in msecs for convolution layer for batch: %d is: %2.6f\n", 
			POINTS, total * 1000.); 

	fprintf( stderr, "ImgCols.... \n" ); 
	printImgCol( nextDevPtr, host, POINTS * HEIGHT * WIDTH * KERNEL * KERNEL * OUT_CHANNELS, 
		POINTS, HEIGHT, WIDTH, OUT_CHANNELS, KERNEL ); 

	fprintf( stderr, "Converted Weights... \n") ; 
	printWeights( data->weights, KERNEL * KERNEL * CHANNELS * OUT_CHANNELS, host); 
	fprintf( stderr, "After conversion ... \n\n"); 
	printWeights( nextDevPtr + POINTS * HEIGHT * WIDTH * KERNEL * KERNEL * OUT_CHANNELS, 
							OUT_CHANNELS * CHANNELS * KERNEL * KERNEL, host ); 
	
	fprintf( stderr, "Converted Img.... \n"); 
	printPool ( deltas_1, host, CHANNELS, HEIGHT, WIDTH, POINTS ); 

	fprintf( stderr, "Test case end... \n"); 
}
