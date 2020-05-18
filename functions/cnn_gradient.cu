
#include <functions/cnn_gradient.h>

#include <functions/cnn_eval_model.h>
#include <functions/cnn_backward.h>

#include <core/errors.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>

real computeCNNGradient(CNN_MODEL *model, DEVICE_DATASET *data, SCRATCH_AREA *scratch,
	real *z, real *dx, real *probs, real *lossFuncErrors, 
	real *gradient, 
	int offset, int curBatchSize, real weightDecay )
{

	//store old ptrs
	real *oldDevPtr = scratch->nextDevPtr; 
	real *oldHostPtr = scratch->nextHostPtr; 
	real *oldPagePtr = scratch->nextPageLckPtr; 

	//locals here. 
	real *errors = scratch->nextDevPtr; 
	real *errors_1 = errors + model->maxDeltaSize; 

	real *nextDevPtr = errors_1 + model->maxDeltaSize; 
	real *nextHostPtr = scratch->nextHostPtr; 

	//update the scratch
	scratch->nextDevPtr = nextDevPtr; 
	scratch->nextHostPtr = nextHostPtr; 

	real lossVal = 0; 
	real alpha = 1; 

	//Forward Pass using evalModel
	lossVal = evaluateCNNModel( model, data, scratch, z, probs, lossFuncErrors, 
						offset, curBatchSize ); 

//fprintf( stderr, "Computed function value is: %f \n", lossVal ); 

	//Backward Pass to get the gradient.... 
	copy_device( errors, lossFuncErrors, sizeof(real) * model->maxDeltaSize, 
		ERROR_MEMCPY_DEVICE_DEVICE ); 

	cnnBackward( model, data, nextDevPtr, z, gradient, dx, errors, errors_1, 
			offset, curBatchSize, nextHostPtr ); 

	//regularization
	// gradient + weightDecay * weights.
	alpha = weightDecay; 
	cublasCheckError( cublasDaxpy( cublasHandle, model->pSize, 
								&alpha, data->weights, 1, 
											gradient, 1) ); 

	//return back to original pointers. 
	scratch->nextDevPtr = oldDevPtr; 
	scratch->nextHostPtr = oldHostPtr; 
	scratch->nextPageLckPtr = oldPagePtr; 

	return lossVal; 
}
