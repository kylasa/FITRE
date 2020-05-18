
#include <functions/cnn_hessian_vec.h>
#include <functions/cnn_hv_forward.h>
#include <functions/cnn_hv_backward.h>
#include <functions/softmax_loss.h>

#include <functions/dev_initializations.h>


#include <core/errors.h>

#include <utilities/print_utils.h>

#include <device/device_defines.h>
#include <device/handles.h>
#include <device/cuda_utils.h>

void cnnHv ( CNN_MODEL *model, DEVICE_DATASET *data, 
	real *z, real *probs, real *lossFuncErrors, real *dx, 
	real *vector, real *hv, 
	int s, int curBatchSize,
	real *devPtr, real *hostPtr, real weightDecay ){

	//Variables
	real *rz = devPtr; 
	real *rError = rz + model->zSize; 
	real *nextDevPtr = rError + model->maxDeltaSize; 
	real *nextHostPtr = hostPtr; 

	real *dataset; 
	real alpha; 

	int *zOffsets = model->zOffsets; 
	int blocks;

	/*
		R Operator (forward and backward)
	*/

	//CNN ROp Forward Pass
	//input: z, 
	//output: rz
	cnnROpForward( model, data, NULL, 
		z, vector, rz, s, curBatchSize, nextDevPtr, nextHostPtr ); 

#ifdef DEBUG_CNN
fprintf( stderr, "..... \n\n Done with ROp Forward Pass.... \n\n "); 
#endif



	//Compute R{ Error } here. 
	/*
		R{ delta } = p_i R{ z_i } - p_i Sigma[ p_j R{ z_j } ]
	computeROpCrossEntropyError 
		( rz + zOffsets[ model->cLayers + model->lLayers ], 
			probs, data->trainSetY + s, curBatchSize, data->numClasses, rError, nextDevPtr); 
	*/

	if (model->lLayers == 0) {
		CONV_LAYER *c = &( model->convLayer [model->cLayers - 1] );
		dataset = rz + zOffsets[ model->cLayers ] + c->outputOffset; 
	} else {
		dataset = rz + zOffsets[ model->cLayers + model->lLayers ];
	}

	computeROpCrossEntropyError_simple( 
		//rz + zOffsets[ model->cLayers + model->lLayers ], 
		dataset, 
		probs, curBatchSize, data->numClasses, rError ); 

#ifdef DEBUG_CNN
fprintf( stderr, "... Done with R{ CrossEntropyLoss } .... \n\n\n"); 
#endif

#ifdef DEBUG_DETAILED
fprintf( stderr, ".... Probs.... \n"); 
copy_host_device( hostPtr, probs, sizeof(real) * curBatchSize * data->numClasses, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, data->numClasses, curBatchSize ); 

fprintf( stderr, ".... RZ.... \n"); 
copy_host_device( hostPtr, dataset, //rz + zOffsets[ model->cLayers + model->lLayers ], 
	sizeof(real) * curBatchSize * data->numClasses, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, data->numClasses, curBatchSize ); 

fprintf( stderr, "... R{ dx } \n"); 
copy_host_device( hostPtr, rError, sizeof(real) * curBatchSize * data->numClasses, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, data->numClasses, curBatchSize ); 
#endif

	//CNN ROp Backward Pass
	//Input: dx, hv
	//Output: hv
	cnnROpBackward( model, data, NULL, 
		z, dx, lossFuncErrors, rError, rz, vector, hv, s, curBatchSize, 
		nextDevPtr, nextHostPtr ); 

	//one vector
	// This is already added in the cnn_hessian_vec.cu file
	/*
   blocks = ( model->pSize + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 
   kerInitOneVector <<<blocks, BLOCK_SIZE >>> 
         ( devPtr, model->pSize );  
   cudaThreadSynchronize (); 
   cudaCheckError ();  

	alpha = weightDecay; 
	cublasCheckError( cublasDaxpy( cublasHandle, model->pSize, 
												&alpha, vector, 1, 
															hv, 1 ) ); 
	*/

#ifdef DEBUG_CNN
	fprintf( stderr, "..... \n\n Done with ROp Backward Pass.... \n\n "); 
#endif

	//alpha = curBatchSize; 
	//cublasCheckError( cublasDscal( cublasHandle, model->pSize, &alpha, hv, 1 ) ); 
}

