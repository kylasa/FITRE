
#include <drivers/gradient_driver.h>

#include <device/query.h>
#include <device/cuda_environment.h>
#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/gen_random.h>
#include <device/handles.h>
#include <device/device_defines.h>

#include <functions/eval_gradient.h>
#include <functions/dev_initializations.h>

#include <core/datadefs.h>
#include <core/errors.h>

#include <drivers/dataset_driver.h> 

#include <utilities/alloc_sampled_dataset.h>
#include <utilities/sample_matrix.h>
#include <utilities/print_utils.h>
#include <utilities/utils.h>

#include <nn/read_nn.h>

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

void readVecFromFile( real *dev, real *host ) {

	//int rows = readVector( host, INT_MAX, "../../matlab/fred-nonconvex/pmatlab.txt", 0, NULL);
	int rows = readVector( host, INT_MAX, "./weights.txt", 0, NULL);
	copy_host_device( host, dev, rows * sizeof(real), cudaMemcpyHostToDevice, 
		ERROR_MEMCPY_HOST_DEVICE ); 
	
	fprintf( stderr, "Finished reading Vec (%d) from file \n", rows ); 

	for (int i = 0; i < 10; i ++)	fprintf( stderr, "%6.10f \n", host[i] ); 
}



void testGradient( NN_MODEL *model, DEVICE_DATASET *deviceData, 
				SCRATCH_AREA *scratch)
{
	real start, total;

	real *gradient, *z, *dx, *nextDevPtr, *nextHostPtr; 


   //set the weights to 0 here. 
   //cuda_memset( deviceData->weights, 0, sizeof(real) * model->pSize, ERROR_MEMSET );  

	readVecFromFile( deviceData->weights, scratch->nextHostPtr ); 
/*
   sparseRandomMatrix( model->pSize, 1, 0.1, nextHostPtr, deviceData->weights );  
   real alpha = 0.5; 
   cublasCheckError( cublasDscal( cublasHandle, model->pSize, &alpha, 
                        deviceData->weights, 1 ) );  
*/
/*
	getRandomVector( model->pSize, NULL, scratch->nextDevPtr, RAND_UNIFORM ); 
	copy_device( deviceData->weights, scratch->nextDevPtr, sizeof(real) * model->pSize, 
				ERROR_MEMCPY_DEVICE_DEVICE ); 
*/

//printVector( deviceData->weights + model->wOffsets[ model->numLayers - 1] , 100, NULL, nextHostPtr ); 

/*
	//all one vector here. 
   int b = model->pSize;  
   int numBlocks = b / BLOCK_SIZE + 
               (( b % BLOCK_SIZE  == 0) ? 0 : 1 );  
   kerInitOneVector <<< numBlocks, BLOCK_SIZE>>> 
      ( deviceData->weights, b );  
   cudaThreadSynchronize (); 
   cudaCheckError (); 
*/

   int sampleSize; 
   sampleSize = int( 0.1 * (real) deviceData->trainSizeX );
   allocSampledDataset( deviceData, sampleSize ); 
   
	fprintf( stderr, "Hessian Vec testing started .... %d \n", sampleSize ); 

   initSampledROffsets( model, sampleSize ); 
   initSampledZOffsets( model, sampleSize ); 

	fprintf( stderr, "Creating sampling matrix... \n"); 
   sampleColumnMatrix( deviceData, scratch, 0 ); 
	fprintf( stderr, "Sampling done... \n"); 

	real ll, mErr; 

	//Gradient Test


	gradient = scratch->nextDevPtr; 
	z = gradient + model->pSize; 
	dx = z  + model->zSize; 
	nextDevPtr = dx + model->rFullSize; 

	nextHostPtr = scratch->nextHostPtr; 
	scratch->nextDevPtr = nextDevPtr;

	start = Get_Time ();
	computeGradient( model, deviceData, scratch, deviceData->weights,
				NULL, NULL, gradient, &ll, &mErr, FULL_DATASET ); 
	total = Get_Timing_Info( start ); 
	fprintf( stderr, "ll: %e, error: %e \n", ll, mErr ); 
   writeVector( gradient, model->pSize, "./gpu_gradient.txt", 0, nextHostPtr );  



/*
	
	gradient = scratch->nextDevPtr; 
	z = gradient + model->pSize; 
	dx = z  + model->sampledZSize; 
	nextDevPtr = dx + model->sampledRSize; 

	nextHostPtr = scratch->nextHostPtr; 
	scratch->nextDevPtr = nextDevPtr;

	computeGradient( model, deviceData, scratch, deviceData->weights,
				z, dx, gradient, &ll, &mErr, SAMPLED_DATASET ); 
	fprintf( stderr, "ll: %e, error: %e \n", ll, mErr ); 

*/

	
}
