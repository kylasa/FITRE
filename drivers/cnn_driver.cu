
#include <drivers/convolution_driver.h>

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

#include <utilities/print_utils.h>
#include <utilities/utils.h>


#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define DATASET_SIZE	256

inline int getMaxZSizes( int *elements, int count ) {
	int m = elements [ 0 ] - elements[ 1 ]; 
	for (int i = 1; i < count; i ++)
		if ( m < (elements[ i+1 ] - elements[ i ] )) m = elements[ i+1 ] - elements[ i ];

	return m; 
}

void initCNNDataset( DEVICE_DATASET *data, SCRATCH_AREA *scratch,
   int h, int w, int ch, int k, int out_ch, int numClasses )
{  
   int height_col = ( h - k ) + 1;
   int width_col = ( w - k ) + 1;
   int points = DATASET_SIZE; ;
   real counter=0;
	real val = 1; 
   
   real *host = scratch->hostWorkspace;
   real *dev = scratch->devWorkspace;
   
   for (int p = 0; p < points; p ++) {
   for (int i = 0; i < ch; i ++){
      //counter = (i + 1) * 10; //  + (p+1)*100;
      //counter = ((real)i + 1) * 0.1; //  + (p+1)*100;
      for (int c = 0; c < w; c ++) { 
      	for (int r = 0; r < h; r ++) {
            //host[ i * h * w + h * c + r + p * ch * h * w ] = (counter ++) ;
            //host[ i * h * w * points + h * c + r + p * h * w ] = 1; //(counter ++) ;
            //host[ i * h * w * points + h * c + r + p * h * w ] = counter; //(counter ++) ;
            host[ i * h * w * points + h * c + r + p * h * w ] = 1; 
				//val += 0.1;
         }
      }
   }
   }
   
   cuda_malloc( (void **)&data->trainSetX, sizeof(real) * ch * h * w * points, 0,
            ERROR_MEM_ALLOC );
   copy_host_device( host, data->trainSetX, sizeof(real) * ch * h * w * points,
         cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );

/*
   getRandomVector( ch * h * w * points, NULL, data->trainSetX, RAND_UNIFORM );  
	real alpha = 0.1; 
	cublasCheckError( cublasDscal( cublasHandle, ch * h * w * points, &alpha, 
												data->trainSetX, 1 ) ); 
	writeVector( data->trainSetX, ch * h * w * points, "./cuda_dataset.txt", 0, host); 
*/

	readVector( host, ch * h * w * points, "./cuda_dataset.txt", 0, NULL );
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

void initVector( real *host, real *devPtr, int pSize )
{
	for (int i = 0; i < pSize; i ++ ) host[ i ]  = 0.1; 

/*
   getRandomVector( pSize, NULL, devPtr, RAND_UNIFORM );  
	real alpha = 0.1; 
	cublasCheckError( cublasDscal( cublasHandle, pSize, &alpha, 
												devPtr, 1 ) ); 
	writeVector( devPtr, pSize, "./cuda_weights2.txt", 0, host); 
*/
	readVector( host, pSize, "./cuda_weights2.txt", 0, NULL );

   copy_host_device( host, devPtr, sizeof(real) * pSize,
         cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );
}

void initWeights( CNN_MODEL *model, DEVICE_DATASET *data, real *hostPtr )
{
	real *weights = data->weights; 
	int *wOffsets = model->wOffsets; 
	int *bOffsets = model->bOffsets; 

	int index = 0; 
	fprintf( stderr, " Model Parameters: %d, .... %d \n", model->pSize, index ); 

	for (int i = 0; i < model->pSize; i ++ ) hostPtr[ i ]  = 0.1; 
	copy_host_device( hostPtr, weights, sizeof(real) * model->pSize, 
		cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

/*
   getRandomVector( model->pSize, NULL, weights, RAND_UNIFORM );  
	real alpha = 0.1; 
	cublasCheckError( cublasDscal( cublasHandle, model->pSize, &alpha, 
												weights, 1 ) ); 
	writeVector( weights, model->pSize, "./cuda_weights.txt", 0, hostPtr); 
*/
	readVector( hostPtr, model->pSize, "./cuda_weights.txt", 0, NULL );
	copy_host_device( hostPtr, weights, sizeof (real) * model->pSize, 
   		cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
}

void printGradient( real *data, int *wOffsets, int *bOffsets, CNN_MODEL *model )
{

	CONV_LAYER *c = & (model->convLayer[ 0 ] ); 
	fprintf( stderr, "W0 -- \n"); 
	print4DMatrix( data, c->outChannels, c->inChannels, c->kSize, c->kSize ); 
	fprintf( stderr, "b0 -- \n"); 
	print2DMatrix( data + c->outChannels * c->inChannels * c->kSize * c->kSize, 1, c->outChannels ); 

	c = & (model->convLayer[ 1 ] ); 
	fprintf( stderr, "W1 -- \n"); 
	print4DMatrix( data + wOffsets[1], c->outChannels, c->inChannels, c->kSize, c->kSize ); 
	fprintf( stderr, "b1 -- \n"); 
	print2DMatrix( data + bOffsets[1], 1, c->outChannels ); 

	FC_LAYER *f = & (model->fcLayer[ 0 ] ); 
	fprintf( stderr, "W2 -- \n"); 
	print2DMatrix( data + wOffsets[ 2 ], f->out, f->in ); 
	fprintf( stderr, "b2 -- \n"); 
	print2DMatrix( data + bOffsets[ 2 ], 1, f->out ); 

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
}

void testCNN( CNN_MODEL *model, DEVICE_DATASET *data, 
					SCRATCH_AREA *scratch )
{
	real ll = 0; 
	int NUM_CLASSES = 10; 

	initCNNDataset( data, scratch, 32, 32, 3, 3, 6, NUM_CLASSES); 	
	fprintf( stderr, "Dataset initialized \n"); 
	//readTestCNN( model, 1, 1, 2, 6, 6, DATASET_SIZE); 
	//readConvCNN( model, 2, 4, NUM_CLASSES, 6, 6, DATASET_SIZE); 
	//readConv2CNN( model, 1, 1, NUM_CLASSES, 14, 14, DATASET_SIZE ); 
	//readConv2CNN( model, 2, 4, NUM_CLASSES, 14, 14, DATASET_SIZE ); 
	readLenetCNN( model, 3, 32, 32, DATASET_SIZE, 1, 0 ); 
	fprintf( stderr, "Done with Network initialization... \n"); 
	cnnInitializations( model, data );
	fprintf( stderr, "Done with weights initialization\n"); 

	//compute the gradient here. 
	int maxDeltaSize = getMaxZSizes (model->zOffsets, model->cLayers + model->lLayers + 1 ); 
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

	//initWeights( model, data, hostPtr );
/*
	copy_host_device( scratch->hostWorkspace, data->weights, sizeof(real) * (model->pSize), 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	printGradient( scratch->hostWorkspace, model->wOffsets, model->bOffsets, model ); 
*/
/*
fprintf( stderr, "Debug dataset... \n"); 

copy_host_device( hostPtr, data->trainSetX, DATASET_SIZE * 6 * 6 * 3 * sizeof(real), 
   cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST );  
print2DMatrix( hostPtr, DATASET_SIZE * 6 * 6, 3); 
exit( -1 );
*/

	start = Get_Time( ); 

	ll = cnnForward( model, data, scratch, z, probs, lossFuncErrors, 0, DATASET_SIZE, MODEL_TRAIN ); 
	fprintf( stderr, "Model Error is %f \n", ll ); 

	//fprintf( stderr, "Error Vector is ---> \n"); 
	//printVector( errors, 10, NULL, scratch->hostWorkspace ); 

	fprintf( stderr, "Beginning BACKWARD PASS... \n"); 
	copy_device( errors, lossFuncErrors, sizeof(real) * maxDeltaSize, ERROR_MEMCPY_DEVICE_DEVICE ); 
	cnnBackward( model, data, nextDevPtr, z, gradient, dx, errors, errors_1, 0, model->batchSize, scratch->nextHostPtr ); 

	copy_host_device( scratch->hostWorkspace, gradient, sizeof(real) * (model->pSize), 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	//printGradient( scratch->hostWorkspace, model->wOffsets, model->bOffsets, model ); 
	printHostVector( scratch->hostWorkspace, model->pSize, NULL ); 


	fprintf( stderr, "Done with Gradient......... \n\n\n\n\n"); 
	/*
	fprintf( stderr, "Begin with HessianVector here... \n\n\n"); 

	initVector( scratch->hostWorkspace, vector, model->pSize ); 
	fprintf( stderr, "Done with vector initialization... \n"); 

	cuda_memset( hv, 0, model->pSize * sizeof(real), ERROR_MEMSET ); 
	fprintf( stderr, "Done with hv initialization... \n"); 
	*/

	//copy_host_device( scratch->hostWorkspace, vector, sizeof(real) * model->pSize, 
	//	cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	//printGradient( scratch->hostWorkspace, model->wOffsets, model->bOffsets, model ); 

	/*
	cnnHv ( model, data, z, probs, lossFuncErrors, dx, vector, hv, 0, DATASET_SIZE, nextDevPtr, scratch->nextHostPtr ); 

	total = Get_Timing_Info( start ); 
	fprintf( stderr, " Time to compute one hessian vec is: %g\n\n\n", 
					total ); 
	*/
/*
	copy_host_device( scratch->hostWorkspace, hv, sizeof(real) * (model->pSize), 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_HOST_DEVICE ); 
	printGradient( scratch->hostWorkspace, model->wOffsets, model->bOffsets, model ); 
*/

	//revert back here. 
	scratch->nextDevPtr = z; 
}
