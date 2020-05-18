
#include <utilities/sample_matrix.h>

#include <device/extract_columns.h>
#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/gen_random.h>

#include <core/errors.h>

/*
	sample columsn from the dataset.. 

	cuSparse supports C = al * op(A) * B + be * C
	A = sparse Matrix... 

	A = Sampling Matrix ( features * sampleSize)
	B = dataset.. ( features X dataset_size )
	C = sample Dataset (features * sampleSize )
*/
void sampleRowMatrix( DEVICE_DATASET *data, SCRATCH_AREA *scratch) 
{
	
	// YET OT BE IMPLEMENTED... 
}

void sampleColumnMatrix (DEVICE_DATASET *data, SCRATCH_AREA *scratch, int sampleXY ) {

	int sampleSize = data->sampleSize; 
	int nRows = data->features; 
	int nCols = data->trainSizeX; 

	int *indices = (int *)scratch->nextHostPtr; 
	int *devIndices = (int *)scratch->nextDevPtr; 

	//generate indices here. 
	//genRandomVector( indices, sampleSize, nCols - 1 ); 
	genRandomVector( indices, sampleSize, nCols ); 

#ifdef DEBUG_FIXED
	fprintf( stderr, "Sample size for sampling: %d \n", sampleSize ); 
	for (int i = 0; i < sampleSize; i ++) indices[i] = i;
#endif

	copy_host_device (indices, devIndices, sizeof(int) * sampleSize, 
		cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	int numBlocks = (sampleSize / BLOCK_SIZE) + 
							( (sampleSize % BLOCK_SIZE) == 0 ? 0 : 1);
	
	if (numBlocks > DEVICE_NUM_BLOCKS)
		numBlocks = DEVICE_NUM_BLOCKS; 

	dim3 dimBlock( BLOCK_SIZE, 1 ); 
	dim3 dimGrid( numBlocks, 1 );

	kerExtractColumns <<< dimGrid, dimBlock>>> 
		( data->sampledTrainX, data->trainSetX, devIndices, nRows, nRows * sampleSize); 
	cudaThreadSynchronize(); 
	cudaCheckError ();

	if (sampleXY) {
		kerExtractColumns <<< dimGrid, dimBlock>>> 
			( data->sampledTrainY, data->trainSetY, devIndices, nRows, nRows * sampleSize); 
		cudaThreadSynchronize(); 
		cudaCheckError ();
	}
}

void selectColumnMatrix(DEVICE_DATASET *data, int sampleSize, int *indices, int offset ) {

	int numBlocks = (sampleSize / BLOCK_SIZE) + 
							( (sampleSize % BLOCK_SIZE) == 0 ? 0 : 1);
	
	if (numBlocks > DEVICE_NUM_BLOCKS)
		numBlocks = DEVICE_NUM_BLOCKS; 

	dim3 dimBlock( BLOCK_SIZE, 1 ); 
	dim3 dimGrid( numBlocks, 1 );

	kerExtractColumns <<< dimGrid, dimBlock >>> 
		( data->sampledTrainX, data->trainSetX, indices + offset, 
				data->features, data->features * sampleSize); 
	cudaThreadSynchronize(); 
	cudaCheckError ();

	kerExtractColumns <<< dimGrid, dimBlock>>> 
			( data->sampledTrainY, data->trainSetY, indices + offset, 
				1, sampleSize); 
	cudaThreadSynchronize(); 
	cudaCheckError ();
}
