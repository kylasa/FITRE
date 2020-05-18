#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <core/datadefs.h>
#include <core/errors.h>

#include <utilities/utils.h>
#include <utilities/print_utils.h>
#include <utilities/reduce_helper.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>
#include <device/reduce.h>

#include <functions/eval_gradient.h>
#include <functions/dev_activations.h>
#include <functions/dev_loglikelihood.h>
#include <functions/dev_layer_error.h>
#include <functions/dev_mse_error.h>
#include <functions/dev_mat_vec_addition.h>
#include <functions/dev_initializations.h>
#include <functions/swish.h>



void applyLayerActivation (int actFunction, 
			real *W, int wRows, int wCols,  
			real *b, int bRows, 
			real *z, int zRows, int zCols, 
			//real *VW, real * Vb, real *rz, int rOperator,
			real *output, real *scratch, real *hostPtr ){

	int matElements;
	int numBlocks;
	int wBlocks;

	real *vec1 = NULL;
	real *colSums = NULL;

	real alpha = 1.0, beta = 0; 

#ifdef DEBUG_DETAILED
fprintf( stderr, "z in the Linear Layer... \n"); 
copy_host_device( hostPtr, z, sizeof(real) * zRows * zCols, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, zRows, zCols ); 
#endif

/*
	fprintf( stderr, "W == %ld, Rows: %d, Cols: %d \n", W, wRows, wCols ); 
	fprintf( stderr, "b == %ld, Rows: %d\n", b, bRows ); 
	fprintf( stderr, "z == %ld, Rows: %d, Cols: %d \n", z, zRows, zCols ); 
*/


	//compute W * Z
	cuda_memset( output, 0, sizeof(real) * wRows * zCols, 
		ERROR_MEMSET ); 
	cublasCheckError (
		cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
						wRows, zCols, wCols,
						&alpha, W, wRows, 
						z, zRows, &beta, output, wRows ) ); 

	/*
	if ( rOperator != 0 ){
		//output += VW * Rz
		cublasCheckError( 
			cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
						wRows, zCols, wCols, 
						&alpha, VW, wRows, 
						rz, zRows, &beta, scratch, wRows ) ); 

		cublasCheckError( 
			cublasDaxpy( cublasHandle, wRows * zCols,
								&alpha, scratch, 1, output, 1 ) ); 
	}
	*/


	matElements = wRows * zCols; 
	numBlocks = matElements / BLOCK_SIZE + 
					((matElements % BLOCK_SIZE == 0) ? 0 : 1);

#ifdef DEBUG_GRADIENT
real temp = 0;
cublasCheckError( cublasDnrm2( cublasHandle, matElements, output, 1, &temp )); 
fprintf( stderr, "WZ[ - ] == %6.10f\n", temp ); 
#endif

#ifdef DEBUG_DETAILED
fprintf( stderr, "Output of Wz + b in the Linear Layer... \n"); 
copy_host_device( hostPtr, output, sizeof(real) * wRows * zCols, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, wRows, zCols ); 
#endif

	if (b != NULL) { 
		kerUtilsAddColumnToMatrix <<<numBlocks ,BLOCK_SIZE>>> 
				( output, wRows, zCols, b ); 
		cudaThreadSynchronize (); 	
		cudaCheckError (); 
#ifdef DEBUG_GRADIENT
cublasCheckError( cublasDnrm2( cublasHandle, bRows, b, 1, &temp )); 
fprintf( stderr, "b[ - ] == %6.10f\n", temp ); 

cublasCheckError( cublasDnrm2( cublasHandle, matElements, output, 1, &temp )); 
fprintf( stderr, "input[ - ] == %6.10f\n", temp ); 
#endif

	}


//real *temp = (real *)malloc( sizeof(real) * 20 ); 
//printVector( output, 20, NULL, temp ); 
//free( temp ); 

#ifdef DEBUG_DETAILED
fprintf( stderr, "Weights(W) in Wz + b in the Linear Layer... \n"); 
copy_host_device( hostPtr, W, sizeof(real) * wRows * wCols, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, wRows, wCols ); 

fprintf( stderr, "Output of Wz + b in the Linear Layer... \n"); 
copy_host_device( hostPtr, output, sizeof(real) * wRows * zCols, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, wRows, zCols ); 
#endif

//TODO store (Wz + b)
#ifdef DEBUG_CNN
fprintf( stderr, "copying Wz + b, to the extended Z location... \n"); 
#endif
copy_device( output + wRows * zCols, output, sizeof(real) * wRows * zCols, 
					ERROR_MEMCPY_DEVICE_DEVICE ); 
//TODO  STORE (Wz + b)

	switch( actFunction ){
		case ACT_LOGISTIC: 	

			kerNNApplyLogistic <<<numBlocks, BLOCK_SIZE>>> 
				(output, wRows * zCols ); 
			cudaThreadSynchronize (); 
			cudaCheckError ();
			break; 

		case ACT_TANH: 

			kerNNApplyTanH <<<numBlocks, BLOCK_SIZE>>> 
				(output, wRows * zCols ); 
			cudaThreadSynchronize (); 
			cudaCheckError ();
			break; 

		case ACT_LINEAR: 
			// Do nothing here. 
			break; 

		case ACT_SOFTMAX: 

			vec1  = scratch; 
			colSums = vec1 + wRows; 
			
			//exp
			kerNNApplyExp <<<numBlocks, BLOCK_SIZE >>> 
				(output, wRows * zCols ); 
			cudaThreadSynchronize (); 
			cudaCheckError ();

			//compute column sums.
			//1-vector
			wBlocks = wRows / BLOCK_SIZE + 
						(( wRows % BLOCK_SIZE) == 0 ? 0 : 1); 
			kerInitVector <<< wBlocks, BLOCK_SIZE >>> 
				(vec1, wRows, 1.0); 
			cudaThreadSynchronize (); 
			cudaCheckError ();
				
			//matvec to compute col sums. 
			// A^T * vec1 = colSum
			cuda_memset( colSums, 0, wRows, ERROR_MEMSET) ;
			cublasCheckError( 
				cublasDgemv( cublasHandle, CUBLAS_OP_T, 
								wRows, zCols, &alpha, output, wRows, 
								vec1, 1, &beta, colSums, 1) ); 		

			//Compute Softmax Here. 
			kerNNComputeSoftmax <<<numBlocks, BLOCK_SIZE>>> 
				( output, wRows, zCols, colSums ); 
			cudaThreadSynchronize (); 
			cudaCheckError ();
				
			break; 

		case CNN_ACT_SOFTPLUS: 
			kerNNApplySOFTPLUS <<< numBlocks, BLOCK_SIZE >>> 
				(output, wRows * zCols ); 
			cudaThreadSynchronize (); 
			cudaCheckError ();
			break;

		case CNN_ACT_RELU: 
			fprintf( stderr, "NOT IMPLEMNETED YET.... \n"); 
			break;
		
		case CNN_ACT_ELU: 
			fprintf( stderr, "NOT IMPLEMNETED YET.... \n"); 
			break;

		case CNN_ACT_SWISH: 
			kerNNSwish <<< numBlocks, BLOCK_SIZE >>> 
				( output, output, wRows * zCols ); 
			cudaThreadSynchronize (); 
			cudaCheckError ();
			break;

		case CNN_ACT_NONE: 
			break;


		default: 
			fprintf (stderr, "computeGradient: Error Unknown Activation Function \n"); 
			exit (-1); 
	}
}

//void nnForwardPass (real *logLikelihood, real *modelError){
void evaluateModel (NN_MODEL *model, DEVICE_DATASET *data, SCRATCH_AREA *scratch, real *weights, 
				real *logLikelihood, real *modelError, DATASET_SIZE isGradient, DATA_TYPE dataType )
{
	real *z = scratch->nextDevPtr; 
	//TODO errTerm should handle both train and test datasets of different sizes.... 
	real *errTerm = z + model->zSize;  
	//real *nextDevPtr = errTerm + (data->features * data->trainSizeX); 
	real *nextDevPtr; 
	if ( dataType == TRAIN_DATA) 
		nextDevPtr = errTerm + (model->layerSizes[ model->numLayers ] * data->trainSizeX); 
	else 
		nextDevPtr = errTerm + (model->layerSizes[ model->numLayers ] * data->testSizeX); 

#ifdef STATS
	real start, total;
#endif

//fprintf( stderr, "Z space ==== %ld, %ld \n", devPtr, nextDevPtr);

	*logLikelihood = 0; 
	*modelError = 0; 

#ifdef STATS
	start = Get_Time (); 
#endif

	nnForwardPass( model, data, weights, z, errTerm, 
			logLikelihood, modelError, scratch->nextHostPtr, nextDevPtr, isGradient, dataType ); 

#ifdef STATS
	total = Get_Timing_Info ( start ); 
	//fprintf (stderr, "Time for Model evaluation (%d) is %f \n", isGradient, total * 1000. ); 
#endif
}

/*
Makes one forward pass and returns 
modelError, and loss/misfit value. 
*/

void nnForwardPass(NN_MODEL *model, DEVICE_DATASET *data, 
				real *weights, real *z, real *errTerm, 
				real *logLikelihood, real *modelError, 
				real *hostPtr, real *devPtr, DATASET_SIZE isGradient, DATA_TYPE dataType)
{

	int numRows, numCols, numColsTarget; 
	int numBlocks;
	int numElements;
	real *dataset, *target;
	int numLayers = model->numLayers; 
	int numFeatures = data->features;
	int *layerSizes = model->layerSizes; 

	//Offsets here. 
	int *zOffsets = (isGradient == FULL_DATASET) ?  
					model->zOffsets : model->sZOffsets;
	int *wOffsets = model->wOffsets; 
	int *bOffsets = model->bOffsets; 
	
	real *nextDevPtr = devPtr; 

#ifdef DEBUG_GRADIENT
	real temp;
#endif

	if (dataType == TEST_DATA){
		dataset = data->testSetX; 
		target = data->testSetY; 
		numRows = data->features; 
		numCols = data->testSizeX; 	
		numColsTarget = data->testSizeY; 
	} else {
		if (isGradient == FULL_DATASET){
			dataset = data->trainSetX; 
			target = data->trainSetY; 
			numRows = data->features; 
			numCols = data->trainSizeX;  	
			numColsTarget = data->trainSizeY;
		} else {
			dataset = data->sampledTrainX; 
			target = data->sampledTrainX;	
			numRows = data->features; 
			numCols = data->sampleSize; 
			numColsTarget = data->sampleSize; 
		}
	}
		

//fprintf( stderr, "Beginning the forward pass... \n"); 

	//Forward Pass
	for (int i = 0; i < numLayers; i ++) {

		//fprintf( stderr, "Layer No: ---------- %d ----------- \n", i); 
		if (i == 0) {
			//SUDHIR-1
			cuda_memset( z, 0, sizeof(real) * numRows * numCols, ERROR_MEMSET ); 
			applyLayerActivation( model->actFuns[0], 
				weights, layerSizes[1], layerSizes[0],  //W
				weights + bOffsets[ 0 ], layerSizes[ 1 ], //b
				dataset, numRows, numCols, //z_i
				//NULL, NULL, NULL, 0, 
				z + zOffsets[1], nextDevPtr, NULL ); // z_i+1
#ifdef DEBUG_GRADIENT
temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, layerSizes[ 1 ], weights + bOffsets[ 1 ], 1, &temp ) );
fprintf( stderr, "norm of b[0] == %6.10f \n", temp ); 
cublasCheckError( cublasDnrm2( cublasHandle, layerSizes[ 1 ], weights + bOffsets[ 0 ], 1, &temp ) );
fprintf( stderr, "norm of b[0] == %6.10f \n", temp ); 
#endif

		}
		else {
			applyLayerActivation( model->actFuns[ i ], 
					weights + wOffsets[ i ], layerSizes[i+1], layerSizes[i], //W 
					weights + bOffsets[i], layerSizes[i+1], //b
					z + zOffsets[i], layerSizes[i], numCols, //z
				//NULL, NULL, NULL, 0, 
					z + zOffsets[ i+1 ], nextDevPtr, NULL ); //z_i+1
		}

#ifdef DEBUG_GRADIENT
temp = 0; 
		cublasCheckError( cublasDnrm2( cublasHandle, 
				model->layerSizes[i+1] * numCols, 
				z + model->zOffsets[ i + 1], 1, &temp ) ); 
fprintf( stderr, "Z[ %d ] ==  %6.10f \n", 	i+1, temp ); 
#endif
	}

//fprintf( stderr, "Done with all the activations...STARTING MODEL_ERROR \n");  

	//After the last activation function. 
	//compute the log likelihood of the loss function. 
	// and error terms for backward pass here. 

	*logLikelihood = 0;

	if (ACT_LINEAR == model->actFuns[ model->numLayers - 1] ){

		//numElements = data->features * data->trainSizeY; 
		numElements = data->features * numColsTarget; 
		numBlocks = numElements / BLOCK_SIZE + 
							((numElements % BLOCK_SIZE) == 0 ? 0 : 1);

		cuda_memset( nextDevPtr, 0, sizeof(real) * BLOCK_SIZE, ERROR_MEMSET ); 
		kerNNComputeLogLikelihoodLinear<<<numBlocks, BLOCK_SIZE>>> 
			//( data->trainSetY, z + model->zOffsets[ numLayers ], 
			( target, z + zOffsets[ numLayers ], 
				numElements, nextDevPtr ); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
	
		//reduce here. 
		reduce_cublas( nextDevPtr, logLikelihood, nextDevPtr + numElements,
				layerSizes[numLayers] , numColsTarget ); 
/*
		reduce <<< numBlocks, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 
			(nextDevPtr, nextDevPtr + BLOCK_SIZE, numElements); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
		
		reduce <<< 1, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 
			(nextDevPtr + BLOCK_SIZE, logLikelihood, numBlocks); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
*/
		


		//errorTerm here. 
		// this is a matrix of size --> (trainSetY);
		kerNNComputeLayerError <<< numBlocks, BLOCK_SIZE >>> 
			//( data->trainSetY, z + model->zOffsets[ numLayers ], numElements, 2.0, errTerm ); 
			( target, z + zOffsets[ numLayers ], numElements, 2.0, errTerm ); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 

	} else if (ACT_LOGISTIC == model->actFuns[ numLayers - 1] ) {

		//fprintf( stderr, "TrainY: %ld,    Points: %d, ==== %ld \n", data->trainSetY, data->trainSizeY, nextDevPtr ); 

		//numElements = data->features * data->trainSizeY; 
		numElements = data->features * numColsTarget; 
		numBlocks = numElements / BLOCK_SIZE + 
							((numElements % BLOCK_SIZE) == 0 ? 0 : 1);

		//This will give us xi
		applyLayerActivation( ACT_LINEAR,
				weights + wOffsets[numLayers-1], layerSizes[numLayers], layerSizes[numLayers-1],  //W
				weights + bOffsets[numLayers-1], layerSizes[ numLayers ], //b
				//z + model->zOffsets[numLayers-1], model->layerSizes[numLayers-1], data->trainSizeX, //z_i
				z + zOffsets[numLayers-1], layerSizes[numLayers-1], numCols, //z_i
				//z + model->zOffsets[numLayers], nextDevPtr ); // z_i+1
				//NULL, NULL, NULL, 0, 
				nextDevPtr, NULL, NULL ); // z_i+1


		kerNNComputeLogLikelihoodLogistic <<< numBlocks, BLOCK_SIZE >>> 
			//( z + model->zOffsets[numLayers], data->trainSetY, numElements, nextDevPtr );
			//( nextDevPtr, data->trainSetY, numElements, nextDevPtr );
			( nextDevPtr, target, numElements, nextDevPtr );
		cudaThreadSynchronize (); 
		cudaCheckError ();

#ifdef DEBUG_GRADIENT
cublasCheckError( cublasDnrm2( cublasHandle, 
				model->layerSizes[numLayers] * numCols, 
				nextDevPtr, 1, &temp ) ); 
fprintf( stderr, "Log likelihood matrix %e \n", 	
					temp ); 
#endif

		//reduce here. 
		reduce_cublas( nextDevPtr, logLikelihood, nextDevPtr + numElements, 
			layerSizes[ numLayers ], numColsTarget ); 
/*
		reduce <<< numBlocks, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 
			(nextDevPtr, nextDevPtr + BLOCK_SIZE, numElements); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
		
		reduce <<< 1, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 
			(nextDevPtr + BLOCK_SIZE, logLikelihood, numBlocks); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
*/

		//computer Error term here. 
		kerNNComputeLayerError <<< numBlocks, BLOCK_SIZE >>> 
			//( data->trainSetY, z + zOffsets[ numLayers ], numElements, 1.0, errTerm ); 
			( target, z + zOffsets[ numLayers ], numElements, 1.0, errTerm ); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 

#ifdef DEBUG_GRADIENT
cublasCheckError( cublasDnrm2( cublasHandle, numElements, 
							errTerm, 1, &temp ) ); 
fprintf(stderr, "Norm of the error matrix here. == %6.10f, elements: %d  \n", temp, numElements); 

cublasCheckError( cublasDnrm2( cublasHandle, numElements, 
							data->trainSetY, 1, &temp ) ); 
fprintf( stderr, "Norm of the train Labels: %6.10f\n", temp ); 

cublasCheckError( cublasDnrm2( cublasHandle, numElements, 
							data->trainSetX, 1, &temp ) ); 
fprintf( stderr, "Norm of the train features: %6.10f\n", temp ); 

cublasCheckError( cublasDnrm2( cublasHandle, numElements, 
							z + model->zOffsets[ numLayers ], 1, &temp )); 
fprintf( stderr, "Norm of the z[numLayers]: %6.10f\n", temp ); 
#endif

	} else if ( ACT_SOFTMAX == model->actFuns[ numLayers - 1] ) {

		//numElements = data->features * data->trainSizeY; 
		numElements = data->features * numColsTarget; 
		numBlocks = numElements / BLOCK_SIZE + 
							((numElements % BLOCK_SIZE) == 0 ? 0 : 1);

		kerNNComputeLogLikelihoodSoftmax <<< numBlocks, BLOCK_SIZE >>> 
			//( data->trainSetY, z + model->zOffsets[ numLayers ], numElements, nextDevPtr); 
			( target, z + zOffsets[ numLayers ], numElements, nextDevPtr); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 

		//reduce here. 
		reduce_cublas( nextDevPtr, logLikelihood, nextDevPtr + numElements, 
			layerSizes[ numLayers ], numColsTarget ); 
/*
		reduce <<< numBlocks, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 
			(nextDevPtr, nextDevPtr + BLOCK_SIZE, numElements); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
		
		reduce <<< 1, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 
			(nextDevPtr + BLOCK_SIZE, logLikelihood, numBlocks); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
*/

		kerNNComputeLayerError <<< numBlocks, BLOCK_SIZE >>> 
			//( data->trainSetY, z + model->zOffsets[ numLayers ], numElements, 1.0, errTerm ); 
			( target, z + zOffsets[ numLayers ], numElements, 1.0, errTerm ); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
	}

	//negative log likelihood here. 
	*logLikelihood *= -1.0 / real(numColsTarget);

	*modelError = 0; 

	//NN Output Error measure here 
	if (model->type == MODEL_TYPE_MSE) {

		//numElements = data->features * data->trainSizeY; 
		numElements = numFeatures * numColsTarget; 
		numBlocks = numElements / BLOCK_SIZE + 
							((numElements % BLOCK_SIZE) == 0 ? 0 : 1);

		kerNNComputeModelError <<< numBlocks, BLOCK_SIZE >>> 
			//( data->trainSetY, z + model->zOffsets[ numLayers ], numElements, nextDevPtr ); 
			( target, z + zOffsets[ numLayers ], numElements, nextDevPtr ); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 

		//reduce here. 
		reduce_cublas( nextDevPtr, modelError, nextDevPtr + numElements, 
			layerSizes[ numLayers ], numColsTarget ); 
/*
		reduce <<< numBlocks, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 
			(nextDevPtr, nextDevPtr + BLOCK_SIZE, numElements); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
		
		reduce <<< 1, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 
			(nextDevPtr + BLOCK_SIZE, modelError, numBlocks); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
*/

		//*modelError /= data->trainSizeX; 
		*modelError = (*modelError) / (real)numColsTarget; 

	} else if ( model->type == MODEL_TYPE_CLASSIFICATION ) {
		//TODO -- fill this one
		/*
		kerNNComputeClassificationError <<< >>> 
			( data->trainY, z + model->dW[ numLayers - 1], modelError ); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 
		*/
	}

}

/*
	Output: z, dx pointers
*/

void computeGradient( NN_MODEL *model, 
		DEVICE_DATASET *data, SCRATCH_AREA *scratch, real *weights,
		real *z, real *dx, real *gradient, 
		real *logLikelihood, real *modelError, DATASET_SIZE isGradient )
{
	//local declarations
	int n = (isGradient  == FULL_DATASET) ? data->trainSizeX : data->sampleSize; 
	real *dataset = (isGradient == FULL_DATASET) ? data->trainSetX : data->sampledTrainX;
	int numFeatures = data->features;
	int numLayers = model->numLayers; 
	int *layerSizes = model->layerSizes; 

	int numBlocks; 
	int b; 
	real alpha = 1, beta = 0; 

	//local pointers. 
	real *xi;

	//Statistics
#ifdef STATS
	real start, total; 
#endif

//#ifdef DEBUG_GRADIENT
real temp;
//#endif


	real *dW = gradient;
	real *errorTerm = scratch->nextDevPtr;
	//TODO chnage this to a reasonable value, so store the largest of the
	// matrices...
	real *oneVector = errorTerm + model->layerSizes[numLayers]* n;  
	real *nextDevPtr = oneVector + 2 * n;
	if (z == NULL) {
		z = oneVector + 2 * n;
		dx = z + model->zSize;
		if (isGradient == SAMPLED_DATASET) dx = z + model->sampledZSize;

		nextDevPtr = dx + model->rFullSize;
		if (isGradient == SAMPLED_DATASET) nextDevPtr = dx + model->sampledRSize;
	}

	//Offsets here. 
	int *zOffsets = model->zOffsets; 
	if (isGradient == SAMPLED_DATASET) zOffsets = model->sZOffsets; 
	
	int *rZOffsets = model->rZOffsets; 
	if (isGradient == SAMPLED_DATASET) rZOffsets = model->sRZOffsets; 

	int *bOffsets = model->bOffsets; 
	int *wOffsets = model->wOffsets; 


#ifdef DEBUG_GRADIENT
fprintf( stderr, "computeGradient: Dataset size: %d \n", n ); 
temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, weights, 1, &temp )); 
fprintf( stderr, "Norm of the weights: %6.10f \n", temp ); 
#endif
	//host space
	real *hostPtr = scratch->nextHostPtr;

	//begin coding here. 
	//Forward Pass
#ifdef STATS
	start = Get_Time (); 
#endif
	//nnForwardPass( model, data, data->weights, z, errorTerm, 
	nnForwardPass( model, data, weights, z, errorTerm, 
			logLikelihood, modelError, hostPtr, nextDevPtr, isGradient, TRAIN_DATA ); 
#ifdef STATS
	total = Get_Timing_Info ( start ); 
//fprintf( stderr, "computeGradient: ForwardPass Time: %f \n", total * 1000 ); 
#endif

#ifdef DEBUG_GRADIENT
temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, layerSizes[numLayers] * n, 
							errorTerm, 1, &temp ) ); 
fprintf(stderr, "Norm of the error matrix here. == %6.10f  \n", temp ); 
#endif

	//Begin the backward pass for compute the partial derivatives. 

//fprintf( stderr, "computeGradient: Beginning to compute the derivatives... \n"); 

#ifdef STATS
start = Get_Time (); 
#endif

	//init db
	//db[ numLayers - 1 ] = sum( errTerm, 2)
	b = n; 
	numBlocks = b / BLOCK_SIZE + 
					(( b % BLOCK_SIZE  == 0) ? 0 : 1 ); 
	kerInitOneVector <<< numBlocks, BLOCK_SIZE>>> 
		( oneVector, b ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	cublasCheckError ( 
		cublasDgemv ( cublasHandle, CUBLAS_OP_N, 
							layerSizes[ numLayers ], b,
							&alpha, errorTerm, layerSizes[ numLayers ], 
							oneVector, 1, 
							&beta, dW + bOffsets[numLayers - 1], 1 )
		); 

#ifdef DEBUG_GRADIENT
temp = 0;
cublasCheckError( cublasDnrm2( cublasHandle, layerSizes[ numLayers-1 ] * n, z + zOffsets[ numLayers - 1 ], 1, &temp ) );
fprintf( stderr, "Z [ %d ] == : %6.10f \n", numLayers - 1, temp );


/*
temp = 0;
cublasCheckError( cublasDnrm2( cublasHandle, layerSizes[ numLayers-1 ] * n, errorTerm, 1, &temp ) );
fprintf( stderr, "Updated Error Term norm: %6.10f \n", temp );


temp = 0;
cublasCheckError( cublasDnrm2( cublasHandle, b, 
											dW + model->bOffsets[ numLayers-1 ], 1, &temp ) ); 
fprintf( stderr, "Norm of db: %6.10f \n", temp );
*/
#endif

	//init dW
	// errTerm * z[ numLayers - 1]'	
	cublasCheckError( 
		cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
						layerSizes[numLayers], layerSizes[numLayers-1], n,
						&alpha, errorTerm, layerSizes[numLayers],  // errTerm
						z + zOffsets [numLayers-1], layerSizes[numLayers-1], //z term 
						&beta, dW + wOffsets[numLayers-1], layerSizes[numLayers] ) //dW term
	);
#ifdef DEBUG_GRADIENT
temp = 0;
cublasCheckError( cublasDnrm2( cublasHandle, layerSizes[ numLayers ] * layerSizes[ numLayers - 1], 
											dW + model->wOffsets[ numLayers-1 ], 1, &temp ) ); 
fprintf( stderr, "Norm of dW: %6.10f \n", temp );
#endif

	//init dx term here. 
	copy_device( dx + rZOffsets[ numLayers - 1], errorTerm, 
					n * layerSizes[numLayers] * sizeof(real), ERROR_MEMCPY_DEVICE_DEVICE ); 

#ifdef DEBUG_GRADIENT
cublasCheckError( cublasDnrm2( cublasHandle, layerSizes[ numLayers ] * n, dx + rZOffsets[ numLayers-1 ], 1, &temp ) ); 
fprintf( stderr, "Norm of dx: %6.10f \n", temp );
#endif

	//update errTerm here. 
	// errTerm = W[ numLayers - 1 ]' * errTerm;
	cublasCheckError( 
		cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
				layerSizes [numLayers-1], n, layerSizes[numLayers],  //m,n,k
				&alpha, weights + wOffsets [numLayers-1], layerSizes[ numLayers ], //W
				dx + rZOffsets[numLayers-1], layerSizes[numLayers], //errTerm
				&beta, errorTerm, layerSizes[numLayers - 1] )
	);

#ifdef DEBUG_GRADIENT
temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, dW, 1, &temp ) ); 
fprintf( stderr, "This SHOULD be the norm of gradient: %6.10f \n", temp ); 
#endif

//fprintf( stderr, "computeGradient: errTerm udpdate Done.. \n"); 
#ifdef STATS
total = Get_Timing_Info( start ); 
//fprintf( stderr, "computeGradient: Error Term time: %f \n", total * 1000 ); 
#endif

#ifdef STATS
start = Get_Time( ); 
#endif

	for (int i = numLayers - 2; i >= 0; i --){

//fprintf (stderr, "computeGradient: BackProp  ---  %d\n", i ); 
		//dimensions of  z[ i + 1]
		// model->layerSizes[i+1] * n
		xi = z + zOffsets[ i + 1 ]; 

		if (ACT_LOGISTIC == model->actFuns[ i ] ) {
			
			b = layerSizes[ i+1 ] * n; 
			numBlocks =  b / BLOCK_SIZE + 
							( (b % BLOCK_SIZE == 0) ? 0 : 1 ); 
			kerNNBackPropLogisticErrors <<<numBlocks, BLOCK_SIZE >>> 
				( errorTerm, xi, b ); 
			cudaThreadSynchronize (); 
			cudaCheckError (); 

		} else if (ACT_TANH == model->actFuns[ i ] ) {

			b = layerSizes[ i+1 ] * n; 
			numBlocks =  b / BLOCK_SIZE + 
							( (b % BLOCK_SIZE == 0) ? 0 : 1 ); 
			kerNNBackPropTanHErrors <<< numBlocks, BLOCK_SIZE >>> 
				( errorTerm, xi, b); 
			cudaThreadSynchronize (); 
			cudaCheckError (); 
		} else if (ACT_LINEAR == model->actFuns[ i ] ) {
			; // do nothing here. 
		} else {
			fprintf( stderr, "computeGradient: Unknown Layer at: %d (%d) \n", i, model->actFuns[i]);
			exit ( -1 );
		}

		//update the dW, dx, db terms here. 
		//dW[ i ] = sum( err, 2); 
		//dimensions of  z[ i + 1] = x [i+1]
		// model->layerSizes[i+1] * n
		//b = model->layerSizes[ i + 1]; 
		b = n; 
		numBlocks = b / BLOCK_SIZE + 
					(( b % BLOCK_SIZE  == 0) ? 0 : 1 ); 
		kerInitOneVector <<< numBlocks, BLOCK_SIZE>>> 
			( oneVector, b ); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 

		// sum(err, 2)
		cublasCheckError ( 
			cublasDgemv ( cublasHandle, CUBLAS_OP_N, 
							layerSizes [i+1], n, 
							&alpha, errorTerm, layerSizes[ i+1 ], 
							oneVector, 1,
							&beta, dW + bOffsets[ i ], 1 )
			); 

//db norm
#ifdef DEBUG_GRADIENT
temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, model->layerSizes[ i ] * n, z + zOffsets[ i ], 1, &temp ) ); 
fprintf( stderr, "Norm of Z[ %d ] = %6.10f \n", i, temp ); 

/*
temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, model->layerSizes[ i+1 ] * n, 
							errorTerm, 1, &temp ) ); 
fprintf( stderr, "Norm of errorTerm[ %d ] = %6.10f\n", i, temp ); 


temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, layerSizes[i+1], dW + model->bOffsets[ i ], 1, &temp ) ); 
fprintf( stderr, "Norm of db[ %d ] = %6.10f \n", i, temp ); 
*/
#endif

		// dW term here. 
		// dW[i] = err * z[i]'
		// err ( model->layerSizes[i+1] * n )  * z' (model->layerSizes[i] * n )
		cublasCheckError (
			cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
								layerSizes[ i+1 ], layerSizes[ i ], n,  //mxk, kxn
								&alpha, errorTerm, layerSizes[ i + 1], //errTerm
								//z + zOffsets[i], layerSizes[ i ],  // z'
								(i == 0) ? dataset : z + zOffsets[i], layerSizes[ i ],  // z'
								&beta, dW + wOffsets[i], layerSizes[i+1] ) // dW
			); 

//db norm
//printVector( dW + model->wOffsets[ i ], 10, NULL, hostPtr ); 
#ifdef DEBUG_GRADIENT
temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, layerSizes[ i+1 ] * layerSizes[ i ], dW + model->wOffsets[ i ], 1, &temp ) ); 
fprintf( stderr, "Norm of dW[ %d ] = %6.10f \n", i, temp ); 
#endif

		//dx term here. 
		// THIS IS NOT NEEDED AT THE MOMENT, check HESSIAN-VEC code. 
		// BUT dx = err, which is used in computing err = w[i]' * err; 
		//copy_device( dx + model->zOffsets[ i ], errorTerm, 
		copy_device( dx + rZOffsets[ i ], errorTerm, 
							sizeof(real) * layerSizes[i+1] * n, ERROR_MEMCPY_DEVICE_DEVICE ); 

#ifdef DEBUG_GRADIENT
cublasCheckError( cublasDnrm2( cublasHandle, model->layerSizes[ i + 1] * n, dx + rZOffsets[ i ], 1, &temp ));
fprintf (stderr, "Norm of dx[ %d ] = %6.10f\n", i, temp ); 
#endif

		//update err = w[i]' * err
		//w[i] = (layerSizes[i+1] * layerSizes[i])
		cublasCheckError (
			cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
							layerSizes[i], n, layerSizes[i+1],
							//&alpha, dW + model->wOffsets[i], model->layerSizes[i+1], 
							&alpha, weights + wOffsets[i], layerSizes[i+1], 
							//dx + model->zOffsets[i], model->layerSizes[ i+1 ], 
							dx + rZOffsets[i], layerSizes[ i+1 ], 
							&beta, errorTerm, layerSizes[ i ] )
			);

//db norm
#ifdef DEBUG_GRADIENT
temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, model->layerSizes[ i + 1] * model->layerSizes[ i ], 
							weights + model->wOffsets[ i ], 1, &temp ));
fprintf (stderr, "Norm of W[ %d ] = %6.10f\n", i, temp ); 

#endif

	} // end of for loop here. 

#ifdef STATS
total = Get_Timing_Info( start ); 
//fprintf( stderr, "computeGradient: BackPropagation Time: %f \n", total * 1000 ); 
#endif

	//computer the gradient here. 
	alpha = -(1.0/real(n));
	cublasCheckError (
		cublasDscal( cublasHandle, model->pSize, &alpha, dW, 1 )
	); 

	//compute LogLikelihood
	//*logLikelihood *= -1.0/((real)n);

//compute the norm of the gradient

#ifdef DEBUG_GRADIENT
temp = 0; 
cublasCheckError( cublasDnrm2 (cublasHandle, model->pSize, dW, 1, &temp ) ); 
fprintf( stderr, "Gradient Norm is : %6.10f\n", temp  ); 
temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, model->zSize, 
							z, 1, &temp ) ); 
fprintf( stderr, "Z norm is : %6.10f \n", temp ); 

temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, model->rFullSize, 
							dx, 1, &temp )); 
fprintf( stderr, "DX norm is : %6.10f \n", temp ); 
#endif


}
