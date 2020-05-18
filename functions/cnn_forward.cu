
#include <functions/cnn_forward.h>
#include <functions/eval_convolution.h>
#include <functions/eval_gradient.h>
#include <functions/softmax_loss.h>
#include <functions/dev_batch_norm.h>
#include <functions/dev_transpose.h>

#include <functions/dev_backprop_convolution.h>

#include <device/cuda_utils.h>

#include <core/errors.h>

#include <utilities/print_utils.h>

long cnnForwardMemRequired(CNN_MODEL *model){

	long memRequired = 0; 	
	long imgColSize; 

	//convolution Layers... 
	//ImgCol Size: 
	for (int i = 0; i < model->cLayers; i ++) {
      CONV_LAYER *c = &(model->convLayer[ i ]);  
      POOL_LAYER *p = &(model->poolLayer[ i ]);

		if (imgColSize < ((p->height * p->width * model->batchSize) * (c->inChannels * c->kSize * c->kSize)) )
			imgColSize = (p->height * p->width * model->batchSize) * (c->inChannels * c->kSize * c->kSize) ;
	}
	memRequired += imgColSize; 

	/*
	//reshaping here. 
   POOL_LAYER *p = &( model->poolLayer[ model->cLayers - 1 ] );  
   CONV_LAYER *c = &( model->convLayer[ model->cLayers - 1 ] );    
   int p_height = ( p->height - p->pSize ) / p->pSize+ 1;  
   int p_width = ( p->width - p->pSize) / p->pSize + 1;  

	memRequired += c->outChannels * curBatchSize * p_height * p_width; 

	//Loss Functions... 
	memRequired += 8 * curBatchSize; 
	*/

	return memRequired;
}

real cnnForward(CNN_MODEL *model, DEVICE_DATASET *data, 
		SCRATCH_AREA *scratch, real *z, real *probs, real *errors, 
		int s, int curBatchSize, EVAL_TYPE forTesting){

	//Z stores the output of each layer... 
	//weights stores the parameters vector for network

	real *nextDevPtr = scratch->nextDevPtr; 
	real *nextPagePtr = scratch->nextPageLckPtr; 
	real *nextHostPtr = scratch->nextHostPtr; 

	real *weights = data->weights; 
	int *wOffsets = model->wOffsets; 
	int *bOffsets = model->bOffsets; 
	int *zOffsets = model->zOffsets; 

	if (model->bias == 0)
		bOffsets = NULL; 

	//outputs here. 
	real modelError = 0; 
	real *dataset = NULL; 
	real *target = NULL; 

	//locals
	//int curBatchSize = model->batchSize; 
	//int s = 0;

#ifdef DEBUG_CNN
fprintf( stderr, "CNNForward: Batch Size: %d \n", curBatchSize ); 
#endif

	//very first layer here. 
	CONV_LAYER *convLayer = &( model->convLayer[0] ); 
	POOL_LAYER *poolLayer = &( model->poolLayer[0] ); 

#ifdef DEBUG_DETAILED
fprintf( stderr, "Dataset at the beginning is as follows: \n"); 
copy_host_device( nextHostPtr, data->trainSetX, curBatchSize * convLayer->height * convLayer->width * convLayer->inChannels * sizeof(real), 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
//print2DMatrix( nextHostPtr, curBatchSize * convLayer->height * convLayer->width, convLayer->inChannels ); 
print4DMatrix( nextHostPtr, 1, convLayer->inChannels, convLayer->height , convLayer->width); 
#endif

/*
	if (forTesting > 0) {
		dataset = data->testSetX + s * data->features;
	} else {
		//dataset = data->trainSetX + s * data->features; 
		dataset = data->currentBatch; 
	}
*/

   switch( forTesting ) { 
      case MODEL_TRAIN: 
      case MODEL_TRAIN_ACCURACY: 
         dataset = data->currentBatch; 
         break;

      case MODEL_TEST_ACCURACY: 
         dataset = data->testSetX + s * data->features;
         dataset = data->currentBatch; 
         break;

      default: 
         fprintf( stderr, "Unknown EVAL_TYPE... \n\n"); 
         exit ( -1 );  
   } 


	applyConvolutionLayer( dataset, curBatchSize, 
				convLayer->inChannels, convLayer->height, convLayer->width, 
				convLayer->kSize, convLayer->padding, convLayer->stride, 
				convLayer->outHeight, convLayer->outWidth,
				weights, ((model->bias == 0) ? NULL : (weights + bOffsets[ 0 ])), 
				z + zOffsets[ 1 ], convLayer->outChannels, 
				model->actFuns[ 0 ], poolLayer->pSize, poolLayer->stride, 
				poolLayer->padding, poolLayer->type, 
				poolLayer->outHeight, poolLayer->outWidth, 
				convLayer->batchNorm, BATCH_NORM_EPSILON, 	
				convLayer->activationOffset, convLayer->poolOffset, convLayer->batchNormOffset, 
				convLayer->meansOffset, convLayer->variancesOffset, 
				nextDevPtr, nextHostPtr, forTesting, convLayer->runningMeansOffset, convLayer->runningVariancesOffset ); 

/*
copy_host_device( nextHostPtr, z + zOffsets[ 1 ], sizeof(real) * 10, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 

fprintf( stderr, "\n\n" ); 
print2DMatrix( nextHostPtr, 1, 10 ); 
fprintf( stderr, "\n\n" ); 

copy_host_device( nextHostPtr, weights + bOffsets[ 0 ], sizeof(real) * 10, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 

fprintf( stderr, "\n\n" ); 
print2DMatrix( nextHostPtr, 1, 10 ); 
fprintf( stderr, "\n\n" ); 

fprintf( stderr, "Weights-1... \n\n"); 
copy_host_device( nextHostPtr, weights, sizeof(real) * 5 * 5, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, 5, 5 ); 

fprintf( stderr, "data-1 ... \n\n"); 
copy_host_device( nextHostPtr, dataset, sizeof(real) * 1 * 3, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, 1, 3 ); 

fprintf( stderr, "Weights-2... \n\n"); 
copy_host_device( nextHostPtr, weights + 5 * 5, sizeof(real) * 5 * 5, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, 5, 5 ); 

fprintf( stderr, "data-2 ... \n\n"); 
copy_host_device( nextHostPtr, dataset + curBatchSize * 32 * 32, sizeof(real) * 1 * 3, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, 1, 3 ); 

fprintf( stderr, "Weights-3... \n\n"); 
copy_host_device( nextHostPtr, weights + 2 * 5 * 5, sizeof(real) * 5 * 5, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, 5, 5 ); 

fprintf( stderr, "data-3 ... \n\n"); 
copy_host_device( nextHostPtr, dataset + 2 * curBatchSize * 32 * 32, sizeof(real) * 1 * 3, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, 1, 3 ); 


exit ( -1 ); 
*/

#ifdef DEBUG_CNN
//printVector( z + zOffsets[ 1 ], 20, NULL, nextHostPtr ); 
fprintf( stderr, "CNNForward: Done with Convolution(0)... \n"); 
#endif

	for (int l = 1; l < model->cLayers; l ++) {

		CONV_LAYER *prevLayer = &( model->convLayer[l-1] ); 
		CONV_LAYER *convLayer = &( model->convLayer[l] ); 
		POOL_LAYER *poolLayer = &( model->poolLayer[l] ); 
		applyConvolutionLayer( z + zOffsets[ l ] + prevLayer->outputOffset, curBatchSize, 
					convLayer->inChannels, convLayer->height, convLayer->width, 
					convLayer->kSize, convLayer->padding, convLayer->stride, 
					convLayer->outHeight, convLayer->outWidth,
					weights + wOffsets[ l ] , ((model->bias == 0) ? NULL : (weights + bOffsets[ l ])) , 
					z + zOffsets[ l+1 ], convLayer->outChannels, 
					model->actFuns[ l ], poolLayer->pSize, poolLayer->stride, 
					poolLayer->padding, poolLayer->type, 
					poolLayer->outHeight, poolLayer->outWidth, 
					convLayer->batchNorm, BATCH_NORM_EPSILON, 
					convLayer->activationOffset, convLayer->poolOffset, convLayer->batchNormOffset,
					convLayer->meansOffset, convLayer->variancesOffset, 
					nextDevPtr, nextHostPtr, forTesting, convLayer->runningMeansOffset, convLayer->runningVariancesOffset ); 

#ifdef DEBUG_CNN
//printVector( z + zOffsets[ l+1 ], 20, NULL, nextHostPtr ); 
fprintf( stderr, "CNNForward: Done with Convolution(%d)... \n", l); 
#endif
	}

	//
	// h * w * n X Channels --> channels * h * w X n
	//
   POOL_LAYER *p = &( model->poolLayer[ model->cLayers - 1 ] );  
   CONV_LAYER *c = &( model->convLayer[ model->cLayers - 1 ] );    

	/*
   int p_height = ( p->height - p->pSize ) / p->pSize+ 1;  
   int p_width = ( p->width - p->pSize) / p->pSize + 1;  

   int col_height = (c->height + 2 * c->padding - c->kSize ) / c->stride + 1;  
   int col_width = (c->width + 2 * c->padding - c->kSize ) / c->stride + 1;  

   int poolOffset = 2 * col_height * col_width * c->outChannels * curBatchSize; 
	*/

	// SUDHIR TESTING ERRORS SK-1
	//reshapeMatrix( z + zOffsets[  model->cLayers ] + c->outputOffset, 
	//				curBatchSize, c->outChannels, p->outHeight * p->outWidth, nextDevPtr ); 
	// SUDHIR TESTING ERRORS SK-1

/*
	int outputOffset = c->outputOffset;
	reshapeMatrix( z + zOffsets[  model->cLayers ] + c->outputOffset, 
					c->outChannels, curBatchSize, p->outHeight * p->outWidth, nextDevPtr ); 

	int numElements = p->outHeight * p->outWidth * curBatchSize * c->outChannels; 
	int blocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE; 

	ker_transpose <<< blocks, BLOCK_SIZE >>> 
		( nextDevPtr, numElements, c->outChannels, p->outHeight, p->outWidth, curBatchSize, z + zOffsets[ model->cLayers ] + c->outputOffset ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 
*/
	int outputOffset = c->outputOffset;
	if ( p->type!= NO_POOL ) {
		reshapeMatrix( z + zOffsets[  model->cLayers ] + c->outputOffset, 
					c->outChannels, curBatchSize, p->outHeight * p->outWidth, nextDevPtr ); 

		int numElements = p->outHeight * p->outWidth * curBatchSize * c->outChannels; 
		int blocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE; 

		ker_transpose <<< blocks, BLOCK_SIZE >>> 
		( nextDevPtr, numElements, c->outChannels, p->outHeight, p->outWidth, curBatchSize, z + zOffsets[ model->cLayers ] + c->outputOffset ); 
		cudaDeviceSynchronize (); 
		cudaCheckError (); 
	} else {
		reshapeMatrix( z + zOffsets[  model->cLayers ] + c->outputOffset, 
					c->outChannels, curBatchSize, c->outHeight * c->outWidth, nextDevPtr ); 

		int numElements = c->outHeight * c->outWidth * curBatchSize * c->outChannels; 
		int blocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE; 

		ker_transpose <<< blocks, BLOCK_SIZE >>> 
		( nextDevPtr, numElements, c->outChannels, c->outHeight, c->outWidth, curBatchSize, z + zOffsets[ model->cLayers ] + c->outputOffset ); 
		cudaDeviceSynchronize (); 
		cudaCheckError (); 
	}

	
	//SK-2 COMMENTED OUT becahse of Tranpose Above 
	/*
	copy_device( z + zOffsets[ model->cLayers ] + c->outputOffset, nextDevPtr, 
							sizeof(real) * c->outChannels * curBatchSize * p->outHeight * p->outWidth, 
							ERROR_MEMCPY_DEVICE_DEVICE ); 
	*/

		//linear layers here. 
		//output of the last convolution layer is linearized
		//column major ordering is assumed. 
		//fcSizes = model->lSizes; 
		for (int l = 0; l < model->lLayers; l ++){

			if (l != 0) outputOffset = 0; 

			//z = f( Wx + b )
			FC_LAYER f = model->fcLayer[ l ]; 
/*
fprintf( stderr, "W in the linear layer... \n\n"); 
copy_host_device( nextHostPtr, weights + wOffsets[ model->cLayers + l ], sizeof(real) * f.out * f.in, 
						cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, f.out, f.in ); 
*/

			applyLayerActivation ( f.actFun, 
				weights + wOffsets[ l + model->cLayers ], f.out, f.in, 
				((model->bias == 0) ? (NULL) : (weights + bOffsets[ l + model->cLayers ])), f.out, 
				z + zOffsets[ l + model->cLayers ] + outputOffset, f.in, curBatchSize, 
				//NULL, NULL, NULL, 0, 
				z + zOffsets[ l + 1 + model->cLayers ], nextDevPtr, nextHostPtr ); 

#ifdef DEBUG_CNN
fprintf( stderr, "CNNForward: Linear Layer: %d, Parameters: Weights(%ld), Bias( %ld ), Z( %ld ), Z_o( %ld ) ",l,  wOffsets[ l + model->cLayers ], ((bOffsets != NULL) ? (bOffsets[ l + model->cLayers ]) : 0), zOffsets[ l + model->cLayers ], zOffsets[ l + 1 + model->cLayers ] ); 
//fprintf( stderr, "CNNForward: Linear Layer : (%d, %d) x (%d, %d) \n", f.out, f.in, f.in, curBatchSize ); 
//printVector( z + zOffsets[ l+1+model->cLayers ], 20, NULL, nextHostPtr ); 
#endif

#ifdef DEBUG_DETAILED
fprintf( stderr, "Linear Layers Output... \n"); 
copy_host_device ( nextHostPtr, z + zOffsets[ l + 1 + model->cLayers ], 
							sizeof(real) * f.out * curBatchSize, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, f.out, curBatchSize ); 
#endif

		}

#ifdef DEBUG_CNN
fprintf( stderr, "CNNForward: Classes: %d, BatchSize: %d, Rows: %d, Cols: %d \n", data->numClasses, curBatchSize, curBatchSize, data->numClasses ); 
#endif

	if (model->lLayers == 0)
		dataset = z + zOffsets[ model->cLayers ] + c->outputOffset;
	else
		dataset = z + zOffsets[ model->lLayers + model->cLayers ]; 

/*
	if (forTesting > 0 )
		target = data->testSetY + s; 
	else { 
		//target = data->trainSetY + s;
		target = data->sampledTrainY;
		//fprintf( stderr, "*** PLEASE REMOVE ME.... \n\n"); 
	}
*/

   switch( forTesting ) {
      case MODEL_TRAIN:
      case MODEL_TRAIN_ACCURACY:
         target = data->sampledTrainY;
         break;

      case MODEL_TEST_ACCURACY:
         target = data->testSetY + s;
         target = data->sampledTrainY;
         break;

      default:
         fprintf( stderr, "Unknown EVAL_TYPE.... \n\n");
         exit( -1 );
   }
/*
fprintf( stderr, " Result of the reshaping is as follows: ....(%d, %d) \n", poolOffset, 2 * c->convOffset); 
copy_host_device( nextHostPtr, dataset, sizeof(real) * p_height * p_width * c->outChannels * curBatchSize, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, 1, curBatchSize * c->outChannels * p_height * p_width); 
exit( -1 ); 
*/
	//compute the softmax here.
	modelError += computeProbsAndCrossEntropyLoss( dataset, 
									target, 
									curBatchSize, data->numClasses,  //rows, cols
									probs, nextDevPtr, nextPagePtr, nextHostPtr ); 			


	//probabilities - ground truth -- crossEntropy Loss here. 
	computeCrossEntropyError ( probs, // probabilities
						curBatchSize, data->numClasses, target, errors ); 

#ifdef DEBUG_DETAILED
fprintf( stderr, "CROSS ENTROPY LOSS VECTOR IS ..... \n"); 
copy_host_device( nextHostPtr, errors, sizeof(real) * data->numClasses * curBatchSize, 
						cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, data->numClasses, curBatchSize ); 
#endif


#ifdef DEBUG_CNN
fprintf( stderr, "CNNForward: Model Error is : %f \n", modelError ); 
fprintf( stderr, "CNNForward: New batch : (%d, %d) \n", s, curBatchSize ); 
#endif

/*
#ifdef DEBUG_DETAILED
fprintf( stderr, "Error vector is .... \n"); 
printHostVector( nextHostPtr, data->numClasses * curBatchSize, NULL ); 
#endif
*/

	return modelError ;
}
