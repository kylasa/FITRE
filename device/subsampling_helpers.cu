
#include <device/subsampling_helpers.h> 

#include <device/cuda_utils.h>
#include <device/gen_random.h>
#include <device/device_defines.h>
#include <device/handles.h>

#include <core/datadefs.h>
#include <core/errors.h>



GLOBAL void kerInitSampleMatrix( int *row, int *col, real *val, real *labels, real *srcLabels, int count, int maxRows )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < count) {
		row[ idx ] = idx; 
		val[ idx ] = 1.; 

		//reshuffle the labels here. 	
		labels[ idx ] = srcLabels[ col[ idx ] ] ; 
	}
}

GLOBAL void kerInitSampleMatrixNoLabels( int *row, int *col, real *val, int count, int maxRows )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < count) {
		row[ idx ] = idx; 
		val[ idx ] = 1.; 
	}
}

void initSamplingMatrix( int rows, SparseDataset *sampledSet, real *sampledLabels, real *srcLabels, int sampledSize ){

	int blocks = (sampledSize / BLOCK_SIZE) + 
			(((sampledSize % BLOCK_SIZE) == 0) ? 0 : 1) ;

	if (sampledLabels == NULL && srcLabels == NULL){
		kerInitSampleMatrixNoLabels <<< blocks, BLOCK_SIZE >>> 
			(sampledSet->rowPtr, sampledSet->colPtr, sampledSet->valPtr, sampledSize, rows ); 
	} else {
		kerInitSampleMatrix <<< blocks, BLOCK_SIZE >>> 
			(sampledSet->rowPtr, sampledSet->colPtr, sampledSet->valPtr, sampledLabels, srcLabels, 
				sampledSize, rows ); 
	}
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}


void prepareForSampling (SparseDataset *spSamplingMatrix, real *sampledLabels, real *srcLabels, int rows, int sampleSize, int *hostPtr) {

	//generate random rows here for sampling. 
	//genRandomVector( hostPtr, sampleSize, rows ); 	
	genRandomVector( hostPtr, sampleSize, rows - 1 ); 	

	copy_host_device( hostPtr, spSamplingMatrix->colPtr, sizeof(int) * sampleSize, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 	

   initSamplingMatrix( rows, spSamplingMatrix, sampledLabels, srcLabels, sampleSize);
}

void sampleDataset ( SparseDataset *spSamplingMatrix, real *dataset, 
			int rows, int cols, int num_classes, 
			real *sampledDataset, int sampleSize )
{
	real alpha = 1.0; 
	real beta = 0; 

	cusparseCheckError (
        	cusparseDcsrmm( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                	sampleSize, cols, rows, spSamplingMatrix->nnz,
                        &alpha, spSamplingMatrix->descr, spSamplingMatrix->sortedVals, 
								spSamplingMatrix->rowCsrPtr, spSamplingMatrix->colPtr, 
								dataset, rows, &beta, sampledDataset, sampleSize)
                        );
}

