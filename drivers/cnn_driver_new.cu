
#include <drivers/cnn_driver_new.h>

#include <core/errors.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>
#include <device/gen_random.h>


#include <nn/read_nn.h>

#include <functions/dev_initializations.h>
#include <functions/eval_convolution.h>
#include <functions/dev_backprop_convolution.h>
#include <functions/cnn_forward.h>
#include <functions/cnn_backward.h>
#include <functions/cnn_hessian_vec.h>
#include <functions/cnn_gradient.h>

#include <utilities/print_utils.h>
#include <utilities/utils.h>


#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define DATASET_SIZE_NEW	1

void initCNNDatasetCIFAR( DEVICE_DATASET *data, SCRATCH_AREA *scratch, 
	int h, int w, int k, int ch )
{
   int height_col = ( h - k ) + 1;
   int width_col = ( w - k ) + 1;
   int points = DATASET_SIZE_NEW; ;
   
   real *host = scratch->hostWorkspace;
   real *dev = scratch->devWorkspace;
   
   for (int p = 0; p < points; p ++)
   	for (int i = 0; i < ch; i ++)
      	for (int c = 0; c < w; c ++)
      		for (int r = 0; r < h; r ++)
            	host[ i * h * w * points + h * c + r + p * h * w ] = 1.; 
   
   cuda_malloc( (void **)&data->trainSetX, sizeof(real) * ch * h * w * points, 0,
            ERROR_MEM_ALLOC );
   
   copy_host_device( host, data->trainSetX, sizeof(real) * ch * h * w * points,
         cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );
   
	for (int p = 0; p < points; p ++) host[ p ] = 9; 
   cuda_malloc( (void **)&data->trainSetY, sizeof(real) * points, 0,
            ERROR_MEM_ALLOC ); 
   copy_host_device( host, data->trainSetY, sizeof(real) * points,
         cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );
   
   data->trainSizeX = points;
   data->trainSizeY = points;
	data->numClasses = 10; 
}

void initCNNDatasetNew( DEVICE_DATASET *data, SCRATCH_AREA *scratch,
   int h, int w, int ch, int k, int out_ch )
{  
   int height_col = ( h - k ) + 1;
   int width_col = ( w - k ) + 1;
   int points = DATASET_SIZE_NEW; ;
   real counter=0;
	real val = 1; 
   
   real *host = scratch->hostWorkspace;
   real *dev = scratch->devWorkspace;
   
   for (int p = 0; p < points; p ++) {
   for (int i = 0; i < ch; i ++){
      for (int c = 0; c < w; c ++) { 
      	for (int r = 0; r < h; r ++) {
            host[ i * h * w * points + h * c + r + p * h * w ] = val; 
         }
      }
   }
   }
   
   cuda_malloc( (void **)&data->trainSetX, sizeof(real) * ch * h * w * points, 0,
            ERROR_MEM_ALLOC );
   
   copy_host_device( host, data->trainSetX, sizeof(real) * ch * h * w * points,
         cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );

   
	for (int p = 0; p < points; p ++) host[ p ] = 1; 
   cuda_malloc( (void **)&data->trainSetY, sizeof(real) * points, 0,
            ERROR_MEM_ALLOC ); 
   copy_host_device( host, data->trainSetY, sizeof(real) * points,
         cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );
   
   data->trainSizeX = points;
   data->trainSizeY = points;
   data->numClasses = 10;
}

void initVectorNew( real *host, real *devPtr, int pSize )
{
	/*
	for (int i = 0; i < pSize; i ++ ) host[ i ]  = 0.1; 
   copy_host_device( host, devPtr, sizeof(real) * pSize,
         cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );
	*/
	getRandomVector( pSize, NULL, devPtr, RAND_NORMAL); 

	real alpha = 0.05;
	cublasCheckError( cublasDscal( cublasHandle, pSize, &alpha, devPtr, 1 )); 
}

void initWeightsCIFAR( CNN_MODEL *model, DEVICE_DATASET *data, real *hostPtr )
{
	/*
	for (int i = 0; i < model->pSize; i ++) hostPtr[ i ] = 0.1; 
	copy_host_device( hostPtr, data->weights , sizeof(real) * model->pSize, 
		cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
	*/
	getRandomVector( model->pSize, NULL, data->weights, RAND_NORMAL ); 
	real alpha = 0.05;
	cublasCheckError( cublasDscal( cublasHandle, model->pSize, &alpha, data->weights, 1 )); 
}

void initWeightsNew( CNN_MODEL *model, DEVICE_DATASET *data, real *hostPtr )
{
	int *wOffsets = model->wOffsets; 
	int *bOffsets = model->bOffsets; 
	real *weights = data->weights; 

	int index = 0; 

	//Convolution Layers;;; 
   //for (int c = 0; c < 1 * 1 * 3 * 3; c ++) { 
   for (int c = 0; c < 6 * 3 * 3 * 3; c ++) { 
   	hostPtr[ c ] = ((real)c + 1) * 0.1; 
   	hostPtr[ c ] = 0.1;
		index ++; 
   }
	fprintf( stderr, "Done with CW... \n"); 

	//for (int i = 0; i < 1; i ++) {
	for (int i = 0; i < 6; i ++) {
		//hostPtr[ index ++ ] = ((real)i + 1) * 0.1;
		hostPtr[ index ++ ] = 0.1;
	}
	fprintf( stderr, "Done with Cb... \n"); 

	//linear - weights ( 10 * 24 )
	//for (int i = 0; i < 2 * 4; i ++) {
	for (int i = 0; i < 10 * 24; i ++) {
		//hostPtr[ index ++ ] = ((real)i + 1) * 0.1;
		hostPtr[ index ++ ] = 0.1;
	}
	fprintf( stderr, "Done with LW... \n"); 

	//for (int i = 0; i < 2; i ++) {
	for (int i = 0; i < 10; i ++) {
		//hostPtr[ index ++ ] = ((real)i + 1) * 0.1; 
		hostPtr[ index ++ ] = 0.1;
	}
	fprintf( stderr, "Done with Lb... \n"); 

	fprintf( stderr, " Model Parameters: %d, .... %d \n", model->pSize, index ); 

	copy_host_device( hostPtr, weights, sizeof(real) * index, 
		cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
}

void printGradientCIFAR( real *data, int *wOffsets, int *bOffsets, CNN_MODEL *model)
{
	for (int i = 0; i < model->cLayers; i ++) {
		CONV_LAYER c = model->convLayer[ i ]; 
		fprintf( stderr, "W[ %d ] --> \n", i ) ;
		print2DMatrix( data + wOffsets[ i ], c.kSize, c.outChannels ); 
		fprintf( stderr, "b[ %d ] --> \n", i ); 
		print2DMatrix( data + bOffsets[ i ], 1, c.outChannels ); 
	}

	for (int i = 0; i < model->lLayers; i ++) {
		FC_LAYER f = model->fcLayer[ i ] ;
		fprintf( stderr, "W[ %d ] --> \n", i ) ;
		print2DMatrix( data + wOffsets[ model->cLayers + i ], f.out, f.in ); 
		fprintf( stderr, "b[ %d ] --> \n", i ); 
		print2DMatrix( data + bOffsets[ model->cLayers + i ], 1, f.out ); 
	}
	
}

void printGradientNew( real *data, int *wOffsets, int *bOffsets )
{
	fprintf( stderr, "W[0] --> \n"); 
	//print4DMatrix( data, 6, 3, 3, 3 ); 
	//print2DMatrix( data, 1 * 3 * 3, 1 ); 
	print2DMatrix( data, 3 * 3 * 3, 6 ); 

	fprintf( stderr, "b[0] --> \n"); 
	//printHostVector( data + bOffsets[ 0 ], 1, NULL); 
	printHostVector( data + bOffsets[ 0 ], 6, NULL); 

	fprintf( stderr, "W[1] --> \n"); 
	//print2DMatrix( data + wOffsets[ 1 ], 2, 4 ); 
	print2DMatrix( data + wOffsets[ 1 ], 10, 24 ); 
	
	fprintf( stderr, "b[1] --> \n"); 
	//printHostVector( data + bOffsets[ 1 ], 2, NULL); 
	printHostVector( data + bOffsets[ 1 ], 10, NULL); 
}

void testCNN_new( CNN_MODEL *model, DEVICE_DATASET *data, 
					SCRATCH_AREA *scratch )
{
	real ll = 0; 

	//initCNNDatasetNew( data, scratch, 6, 6, 3, 3, 6); 	
	initCNNDatasetCIFAR( data, scratch, 32, 32, 5, 3 ) ; 
	fprintf( stderr, "Dataset initialized \n"); 

	//readTestCNN( model, 3, 6, 10, 6, 6, DATASET_SIZE_NEW); 
	readLenetCNN( model, 3, 32, 32, DATASET_SIZE_NEW, 1, 0 ); 

	fprintf( stderr, "Done with Network initialization... \n"); 
	cnnInitializations( model, data );
	fprintf( stderr, "Done with weights initialization\n"); 

	//compute the gradient here. 
	real *z = scratch->nextDevPtr; 
	real *dx = z + model->zSize; 
	real *gradient= dx + model->zSize; 
	real *errors = gradient + model->pSize; 
	real *errors_1 = errors + model->maxDeltaSize; 

	real *lossFuncErrors = errors_1 + model->maxDeltaSize; 

	real *rz = lossFuncErrors + model->maxDeltaSize; 
	real *rerror = rz + model->zSize; 
	real *probs = rerror + model->maxDeltaSize; 
	real *vector = probs + DATASET_SIZE_NEW * data->numClasses; 
	real *hv = vector + model->pSize; 

	real *nextDevPtr = hv + model->pSize; 
	real *hostPtr = scratch->nextHostPtr; 

	//upate the nextDevPtrs here. 
	scratch->nextDevPtr = nextDevPtr; 

	//initWeightsNew( model, data, hostPtr );
	initWeightsCIFAR( model, data, hostPtr ); 
	fprintf( stderr, "Initialized the weights ... \n"); 

	ll = computeCNNGradient( model, data, scratch, z, dx, probs, lossFuncErrors, 
				gradient, 0, model->batchSize, 0); 

	copy_host_device( scratch->hostWorkspace, gradient, sizeof(real) * (model->pSize), 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 

	//printGradientNew( scratch->hostWorkspace, model->wOffsets, model->bOffsets ); 
	printGradientCIFAR( scratch->hostWorkspace, model->wOffsets, model->bOffsets, model ); 

	fprintf( stderr, "Done with Gradient......... \n\n\n\n\n"); 
	fprintf( stderr, "Begin with HessianVector here... \n\n\n"); 

	initVectorNew( scratch->hostWorkspace, vector, model->pSize ); 
	fprintf( stderr, "Done with vector initialization... \n"); 

	cuda_memset( hv, 0, model->pSize * sizeof(real), ERROR_MEMSET ); 
	fprintf( stderr, "Done with hv initialization... \n"); 

	cnnHv ( model, data, z, probs, lossFuncErrors, dx, vector, hv, 0, DATASET_SIZE_NEW, nextDevPtr, scratch->nextHostPtr, 0 ); 
	copy_host_device( scratch->hostWorkspace, hv, sizeof(real) * (model->pSize), 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 

	printGradientCIFAR( scratch->hostWorkspace, model->wOffsets, model->bOffsets, model ); 

	//revert back here. 
	scratch->nextDevPtr = z; 
}
