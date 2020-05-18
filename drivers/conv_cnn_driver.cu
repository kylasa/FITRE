
#include <drivers/convolution_driver.h>
#include <drivers/conv_cnn_driver.h>

#include <core/errors.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>
#include <device/gen_random.h>


#include <nn/read_nn.h>
#include <nn/read_vgg.h>
#include <nn/nn_decl.h>

#include <functions/dev_initializations.h>
#include <functions/eval_convolution.h>
#include <functions/dev_backprop_convolution.h>
#include <functions/cnn_forward.h>
#include <functions/cnn_backward.h>
#include <functions/cnn_hessian_vec.h>

#include <utilities/print_utils.h>
#include <utilities/utils.h>
#include <utilities/dataset.h>


#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define DATASET_SIZE	4

inline int getMaxZSizes( int *elements, int count ) {
	int m = elements [ 0 ] - elements[ 1 ]; 
	for (int i = 1; i < count; i ++)
		if ( m < (elements[ i+1 ] - elements[ i ] )) m = elements[ i+1 ] - elements[ i ];

	return m; 
}

void convertDataset( real *src, real *tgt, int ch, int h, int w)
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

void initConvCNNDataset( DEVICE_DATASET *data, SCRATCH_AREA *scratch,
   int h, int w, int ch, int k, int out_ch, int numClasses )
{  
   int points = DATASET_SIZE; ;
   real counter=0;
	real val = 1; 
   
   real *host = scratch->hostWorkspace;
   real *dev = scratch->devWorkspace;
	real *batchMajorDataset = host + h * w * ch * DATASET_SIZE; 
   
   for (int p = 0; p < points; p ++) {
   	for (int i = 0; i < ch; i ++){
      	for (int c = 0; c < w; c ++) { 
      		for (int r = 0; r < h; r ++) {
            	host[ i * h * w * points + h * c + r + p * h * w ] = 1; 
         	}
      	}
   	}
   }


   
   cuda_malloc( (void **)&data->trainSetX, sizeof(real) * ch * h * w * points, 0,
            ERROR_MEM_ALLOC );

   getRandomVector( ch * h * w * points, NULL, data->trainSetX, RAND_UNIFORM );  
   writeVector( data->trainSetX, ch * h * w * points, "./cuda_dataset.txt", 0, host); 

	readVector( host, ch * h * w * points, "./cuda_dataset.txt", 0, NULL );

	convertDataset( host, batchMajorDataset, ch, h, w ); 

   copy_host_device( batchMajorDataset, data->trainSetX, sizeof(real) * ch * h * w * points,
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

void initConvCNNVector( real *host, real *devPtr, int pSize )
{
	for (int i = 0; i < pSize; i ++ ) host[ i ]  = 0.1; 

   getRandomVector( pSize, NULL, devPtr, RAND_UNIFORM );  
   writeVector( devPtr, pSize, "./cuda_weights2.txt", 0, host); 

   real alpha = 0.1; 
   cublasCheckError( cublasDscal( cublasHandle, pSize, &alpha, devPtr, 1 ) ); 

	readVector( host, pSize, "./cuda_weights2.txt", 0, NULL );

   copy_host_device( host, devPtr, sizeof(real) * pSize,
         cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );
}

void initConvCNNWeights( CNN_MODEL *model, DEVICE_DATASET *data, real *hostPtr )
{
	real *weights = data->weights; 
	int *wOffsets = model->wOffsets; 
	int *bOffsets = model->bOffsets; 

	int index = 0; 
	fprintf( stderr, " Model Parameters: %d, .... %d \n", model->pSize, index ); 

	for (int i = 0; i < model->pSize; i ++ ) hostPtr[ i ]  = 0.1; 
	copy_host_device( hostPtr, weights, sizeof(real) * model->pSize, 
		cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

   getRandomVector( model->pSize, NULL, data->weights, RAND_UNIFORM );  

   real alpha = .1; 
   cublasCheckError( cublasDscal( cublasHandle, model->pSize, &alpha, data->weights, 1 ) ); 

   writeVector( data->weights, model->pSize, "./cuda_weights.txt", 0, hostPtr); 

	readVector( hostPtr, model->pSize, "./cuda_weights.txt", 0, NULL );
	//readVector( hostPtr, model->pSize, "./lenet_kaiming.txt", 0, NULL );

	copy_host_device( hostPtr, weights, sizeof (real) * model->pSize, 
   		cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
}

void printConvCNNGradient( real *data, int *wOffsets, int *bOffsets, CNN_MODEL *model )
{
	CONV_LAYER *c;
	FC_LAYER *f;

	c = & (model->convLayer[ 0 ] ); 
	fprintf( stderr, "W0 -- \n"); 
	print4DMatrix( data, c->outChannels, c->inChannels, c->kSize, c->kSize ); 
	//print2DMatrix( data, c->kSize, c->kSize ); 
	//fprintf( stderr, "b0 -- \n"); 
	//print2DMatrix( data + bOffsets[ 0 ], 1, c->outChannels ); 

/*
	c = & (model->convLayer[ 1 ] ); 
	fprintf( stderr, "W1 -- \n"); 
	print4DMatrix( data + wOffsets[1], c->outChannels, c->inChannels, c->kSize, c->kSize ); 
	//print2DMatrix( data + wOffsets[ 1] , c->kSize, c->kSize ); 
	fprintf( stderr, "b1 -- \n"); 
	print2DMatrix( data + bOffsets[1], 1, c->outChannels ); 
*/

	f = & (model->fcLayer[ 0 ] ); 
	fprintf( stderr, "W1 -- \n"); 
	print2DMatrix( data + wOffsets[ 1 ], f->out, f->in ); 
	//fprintf( stderr, "b1 -- \n"); 
	//print2DMatrix( data + bOffsets[ 2 ], 1, f->out ); 

/*
	f = & (model->fcLayer[ 1 ] ); 
	fprintf( stderr, "W3 -- \n"); 
	print2DMatrix( data + wOffsets[ 3 ], f->out, f->in ); 
	fprintf( stderr, "b3 -- \n"); 
	print2DMatrix( data + bOffsets[ 3 ], 1, f->out ); 

	f = & (model->fcLayer[ 2 ] ); 
	fprintf( stderr, "W4 -- \n"); 
	print2DMatrix( data + wOffsets[ 4 ], f->out, f->in ); 
	fprintf( stderr, "b4 -- \n"); 
	print2DMatrix( data + bOffsets[ 4 ], 1, f->out ); 
*/
}

void testConvCNN( CNN_MODEL *model, DEVICE_DATASET *data, 
					SCRATCH_AREA *scratch )
{
	real ll = 0; 

/*
	int NUM_CLASSES = 4; 
	int HEIGHT = 8;
	int WIDTH = DATASET_SIZE;
	int CHANNELS  = 1; 
	int KERNEL = 0; 
	int OUT_CHANNELS = 1; 
	readFCNN( model, DATASET_SIZE); 
*/

	int HEIGHT = 16;
	int WIDTH = 16;
	int CHANNELS  = 2; 
	int KERNEL = 3; 
	int OUT_CHANNELS = 4; 
	int NUM_CLASSES = 10 ; 
	readTestCNN( model, CHANNELS, OUT_CHANNELS, NUM_CLASSES, WIDTH, HEIGHT, DATASET_SIZE); 

/*
	int NUM_CLASSES = 10; 
	int HEIGHT = 32;
	int WIDTH = 32;
	int CHANNELS  = 3; 
	int OUT_CHANNELS = 0; 
	int KERNEL = 3; 
	readLenetCNN( model, CHANNELS, HEIGHT, WIDTH, DATASET_SIZE ); 
*/

	/*
	int NUM_CLASSES = 6 * 8 * 8; 
	int HEIGHT = 32;
	int WIDTH = 32;
	int CHANNELS  = 3; 
	int KERNEL = 5; 
	int OUT_CHANNELS = 6; 
	readConv2CNN( model, CHANNELS, OUT_CHANNELS, NUM_CLASSES, HEIGHT, WIDTH, DATASET_SIZE ); 
	*/

	/*
	int NUM_CLASSES = 4 * 4; 
	int HEIGHT = 6;
	int WIDTH = 6;
	int CHANNELS  = 2; 
	int KERNEL = 3; 
	int OUT_CHANNELS = 4; 
	readConvCNN( model, CHANNELS, OUT_CHANNELS, NUM_CLASSES, HEIGHT, WIDTH, DATASET_SIZE); 
	*/

	/*
	int NUM_CLASSES = 4; 
	int HEIGHT = 6;
	int WIDTH = 6;
	int CHANNELS  = 1; 
	int KERNEL = 3; 
	int OUT_CHANNELS = 1; 
	readConvCNN( model, CHANNELS, OUT_CHANNELS, NUM_CLASSES, HEIGHT, WIDTH, DATASET_SIZE); 
	*/

	/*
	int NUM_CLASSES = 2 * 3 * 3; 
	int HEIGHT = 6;
	int WIDTH = 6;
	int CHANNELS  = 1; 
	int KERNEL = 5; 
	int OUT_CHANNELS = 2; 
	readConvCNN( model, CHANNELS, OUT_CHANNELS, NUM_CLASSES, HEIGHT, WIDTH, DATASET_SIZE); 
	*/

	/*
	int NUM_CLASSES = 10; 
	int HEIGHT = 4;
	int WIDTH = 4;
	int CHANNELS  = 2; 
	int KERNEL = 3; 
	int OUT_CHANNELS = 3; 
	readTestVGG( model, DATASET_SIZE, HEIGHT, WIDTH, NUM_CLASSES, CHANNELS ); 
	*/

	fprintf( stderr, "Done with Network initialization... \n"); 

	initConvCNNDataset( data, scratch, HEIGHT, WIDTH, CHANNELS, KERNEL, OUT_CHANNELS, NUM_CLASSES); 	
	fprintf( stderr, "Dataset initialized \n"); 

	cnnInitializations( model, data );
	fprintf( stderr, "Done with weights initialization\n"); 

	//compute the gradient here. 
	int maxDeltaSize = model->maxDeltaSize; 
	fprintf( stderr, "Max Z size is : %d \n", maxDeltaSize ); 
	real *z = scratch->nextDevPtr; 
	real *dx = z + model->zSize; 
	real *gradient= dx + model->zSize; 
	real *errors = gradient + model->pSize; 
	real *errors_1 = errors + maxDeltaSize; 

	real *lossFuncErrors = errors_1 + maxDeltaSize; 

	real *rz = lossFuncErrors + maxDeltaSize; 
	real *rerror = rz + model->zSize; 
	real *probs = rerror + maxDeltaSize; 
	real *vector = probs + DATASET_SIZE * data->numClasses; 
	real *hv = vector + model->pSize; 

	real *nextDevPtr = hv + model->pSize; 
	real *hostPtr = scratch->nextHostPtr; 
	scratch->nextDevPtr = nextDevPtr; 

	real start, total; 

	initConvCNNWeights( model, data, hostPtr );
	fprintf( stderr, "Done with initWeights ... \n"); 
	copy_host_device( scratch->hostWorkspace, data->weights, sizeof(real) * (model->pSize), 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	printConvCNNGradient( scratch->hostWorkspace, model->wOffsets, model->bOffsets, model ); 
	fprintf( stderr, "Done with printing initial values of Gradient... \n "); 

	__THREADS_PER_SAMPLE__ = 1; 
	data->currentBatch = data->trainSetX; 
	data->sampledTrainY = data->trainSetY; 


	start = Get_Time( ); 

	ll = cnnForward( model, data, scratch, z, probs, lossFuncErrors, 0, DATASET_SIZE, MODEL_TRAIN ); 
	fprintf( stderr, "Model Error is %f \n", ll ); 

	//fprintf( stderr, "Error Vector is ---> \n"); 
	//printVector( errors, 10, NULL, scratch->hostWorkspace ); 

	fprintf( stderr, "Beginning BACKWARD PASS... \n"); 
	copy_device( errors, lossFuncErrors, sizeof(real) * maxDeltaSize, ERROR_MEMCPY_DEVICE_DEVICE ); 
	cnnBackward( model, data, nextDevPtr, z, gradient, dx, errors, errors_1, 0, model->batchSize, scratch->nextHostPtr ); 

	fprintf( stderr, "Printing Gradient... \n "); 
	copy_host_device( scratch->hostWorkspace, gradient, sizeof(real) * (model->pSize), 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	printConvCNNGradient( scratch->hostWorkspace, model->wOffsets, model->bOffsets, model ); 

	fprintf( stderr, "Done with Gradient......... \n\n\n\n\n\n\n\n\n\n\n\n\n\n"); 


	fprintf( stderr, "Begin with HessianVector here... \n\n\n"); 

	initConvCNNVector( scratch->hostWorkspace, vector, model->pSize ); 
	fprintf( stderr, "Done with vector initialization... \n"); 

	cuda_memset( hv, 0, model->pSize * sizeof(real), ERROR_MEMSET ); 
	fprintf( stderr, "Done with hv initialization... \n"); 

	fprintf( stderr, "Printing vector to be used for HV computation... \n"); 
	copy_host_device( scratch->hostWorkspace, vector, sizeof(real) * model->pSize, 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	printConvCNNGradient( scratch->hostWorkspace, model->wOffsets, model->bOffsets, model ); 

	fprintf( stderr, "\n\n\n\n\n\n\n"); 

	cnnHv ( model, data, z, probs, lossFuncErrors, dx, vector, hv, 0, DATASET_SIZE, nextDevPtr, scratch->nextHostPtr, 0 ); 

	total = Get_Timing_Info( start ); 
	fprintf( stderr, " Time to compute one hessian vec is: %g\n\n\n", 
					total ); 
	fprintf( stderr, "Printing ... the HessianVec result... \n"); 
	copy_host_device( scratch->hostWorkspace, hv, sizeof(real) * (model->pSize), 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	printConvCNNGradient( scratch->hostWorkspace, model->wOffsets, model->bOffsets, model ); 


	//revert back here. 
	scratch->nextDevPtr = z; 

}
