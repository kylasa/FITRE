
#include <functions/cnn_accuracy.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

#include <core/errors.h>

#include <functions/cnn_forward.h>
#include <utilities/dataset_utils.h>

#include <solvers/kfac_da.h>



GLOBAL void ker_compute_accuracy ( real *probs, real *target, 
	int rows, int numClasses, real *hits ) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	real maxProb = 0; 
	int  myClass = 0; 

	if (idx < rows) {
		for (int c = 0; c < numClasses; c ++) {
			if ( maxProb < probs[ idx * numClasses + c ] ) {
				maxProb = probs[ idx * numClasses + c ]; 
				myClass = c; 
			}
		}

		hits[ idx ] = 0; 
		if ( (myClass + 1) == target[ idx ] ) {
			hits[ idx ]  = 1;
		}
	}
}

real computeAccuracy( real *probs, real *target, 
	int rows, int numClasses, real *devPtr, real *pageLckPtr ) {

	int hits; 
	int blocks; 

	cuda_memset( devPtr, 0, sizeof(real) * rows, ERROR_MEMSET ); 

	blocks = ( rows + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 

	ker_compute_accuracy <<< blocks, BLOCK_SIZE >>>
		( probs, target, rows, numClasses, devPtr ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	//Reduce here. 
	cublasCheckError( cublasDdot( cublasHandle, rows, 
			devPtr, 1, devPtr, 1, pageLckPtr ) ); 

	return (*pageLckPtr * 100.) / (real) rows; 
}

void computeTestGeneralizationErrors( CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch, real *z, real *probs, real *errors, 
	real *likelihood, real *accuracy  )
{

	real hits = 0; 
	real batchLikelihood = 0; 

	int curBatchSize = 0; 
	int curIdx = 0; 

	real *nextDevPtr = scratch->nextDevPtr; 
	real *nextPageLckPtr = scratch->nextPageLckPtr; 
	real *nextHostPtr = scratch->nextHostPtr; 
	
	*likelihood = 0; 
	*accuracy = 0; 
	curBatchSize = model->batchSize; 
	while( curIdx < data->testSizeX ) {

      copy_device( data->sampledTrainX, data->testSetX + data->features * curIdx, 
            sizeof(real) * data->features * curBatchSize, ERROR_MEMCPY_DEVICE_DEVICE );  
      copy_device( data->sampledTrainY, data->testSetY + curIdx, 
            sizeof(real) * curBatchSize, ERROR_MEMCPY_DEVICE_DEVICE); 

      augmentData( model, data, curIdx, curBatchSize, nextDevPtr, nextHostPtr , 0, data->datasetType );  

		batchLikelihood = 0; 
		batchLikelihood += cnnForward( model, data, scratch, z, probs, errors, curIdx, curBatchSize, MODEL_TEST_ACCURACY); 

		*likelihood += (batchLikelihood * curBatchSize); 

		computeAccuracy( probs, data->testSetY + curIdx, curBatchSize, data->numClasses, 
			nextDevPtr, nextPageLckPtr ); 

		hits += *nextPageLckPtr; 
		//fprintf( stderr, "idx: %d, size: %d, Likelihood: %e, hits: %e \n", curIdx, curBatchSize, *likelihood, hits ); 

		curIdx += curBatchSize; 
		if ( (curIdx + model->batchSize) < data->testSizeX ) {
				curBatchSize = model->batchSize; 
		} else {
				curBatchSize = data->testSizeX - curIdx; 
		}
	}

	*accuracy = hits * 100. / (real) data->testSizeX; 
	*likelihood /= (real) data->testSizeX;
}


void computeTrainGeneralizationErrors( CNN_MODEL *model, DEVICE_DATASET *data, HOST_DATASET *host,  
   SCRATCH_AREA *scratch, real *z, real *probs, real *errors, 
   real *likelihood, real *accuracy  )
{

   real hits = 0;  
   real batchLikelihood = 0;  

   int curBatchSize = 0;  
   int curIdx = 0;  

	int *hostIndices = (int *)scratch->nextHostPtr;

   real *nextDevPtr = scratch->nextDevPtr; 
   real *nextPageLckPtr = scratch->nextPageLckPtr; 
   real *nextHostPtr = scratch->nextHostPtr + data->trainSizeX; 

	for (int i = 0; i < data->trainSizeX; i ++) hostIndices[ i ] = i; 

   *likelihood = 0;  
   *accuracy = 0;  
   curBatchSize = model->batchSize; 
   while( curIdx < data->trainSizeX ) { 

		if (data->datasetType != IMAGENET) { 
      	copy_device( data->sampledTrainX, data->trainSetX + data->features * curIdx, 
         	sizeof(real) * data->features * curBatchSize, ERROR_MEMCPY_DEVICE_DEVICE );  
      	copy_device( data->sampledTrainY, data->trainSetY + curIdx, sizeof(real) * curBatchSize, ERROR_MEMCPY_DEVICE_DEVICE); 
		} else { 
			selectHostMatrix( host, data->sampledTrainX, data->sampledTrainY, 
						curBatchSize, hostIndices, curIdx, nextHostPtr ); 
		} 

		//BUG-FIX
      augmentData( model, data, curIdx, curBatchSize, nextDevPtr, nextHostPtr , 0, data->datasetType );  

      batchLikelihood = cnnForward( model, data, scratch, z, probs, errors, curIdx, curBatchSize, MODEL_TRAIN_ACCURACY);  
      // 1 will compute batch-means, will need to change this later. 

      *likelihood += (batchLikelihood * curBatchSize); 

		if (data->datasetType != IMAGENET) { 
      	computeAccuracy( probs, data->trainSetY + curIdx, curBatchSize, data->numClasses, 
         	nextDevPtr, nextPageLckPtr );  
		}  else { 
      	computeAccuracy( probs, data->sampledTrainY, curBatchSize, data->numClasses, 
         	nextDevPtr, nextPageLckPtr );  
		} 

      hits += *nextPageLckPtr; 

      curIdx += curBatchSize; 
      if ( (curIdx + model->batchSize) < data->trainSizeX ) { 
            curBatchSize = model->batchSize; 
      } else {
            curBatchSize = data->trainSizeX - curIdx; 
				break;
      }   
   }   

   *accuracy = hits * 100. / (real) data->trainSizeX; 
   *likelihood /= (real) data->trainSizeX;
}
