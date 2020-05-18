
#include <drivers/hessian_driver.h>

#include <core/datadefs.h>
#include <core/structdefs.h>
#include <core/errors.h>

#include <device/query.h>
#include <device/cuda_environment.h>
#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>

#include <functions/eval_gradient.h>
#include <functions/eval_hessian_vec.h>
#include <functions/dev_initializations.h>

#include <drivers/dataset_driver.h> 

#include <utilities/print_utils.h>
#include <utilities/utils.h>
#include <utilities/alloc_sampled_dataset.h>
#include <utilities/sample_matrix.h>

#include <nn/read_nn.h>

#include <stdio.h>
#include <stdlib.h>

#include <time.h>


void readVecFromFileHessian( real *dev, real *host, char *f ) { 

   int rows = readVector( host, INT_MAX, f, 0, NULL);
   copy_host_device( host, dev, rows * sizeof(real), cudaMemcpyHostToDevice, 
      ERROR_MEMCPY_HOST_DEVICE );  
   
   fprintf( stderr, "Finished reading Vec (%d) from file \n", rows );  

   for (int i = 0; i < 10; i ++) fprintf( stderr, "%6.10f \n", host[i] );  
}



void testHessianVec( NN_MODEL *model, DEVICE_DATASET *deviceData, 
				SCRATCH_AREA *scratch)
{
	real start, total;

	real *z, *dx, *vec, *nextDevPtr, *gradient;

	int sampleSize; 
	sampleSize = int( 0.1 * (real) deviceData->trainSizeX );
	allocSampledDataset( deviceData, sampleSize ); 

	initSampledROffsets( model, sampleSize ); 
	initSampledZOffsets( model, sampleSize ); 

	z = scratch->devWorkspace; 
	dx = z + model->sampledZSize; 
	vec = dx + model->sampledRSize; 
	gradient = vec + model->pSize; 
	nextDevPtr = gradient + model->pSize; 

	scratch->nextDevPtr = nextDevPtr; 
	scratch->nextHostPtr = scratch->hostWorkspace;
	
fprintf( stderr, "Hessian Vec testing started .... %d \n", sampleSize ); 


fprintf( stderr, "Creating sampling matrix... \n"); 
	sampleColumnMatrix( deviceData, scratch, 0 ); 
fprintf( stderr, "Sampling done... \n"); 

	readVecFromFileHessian( deviceData->weights, scratch->nextHostPtr, "./weights.txt" );
	readVecFromFileHessian( vec, scratch->nextHostPtr, "./weights2.txt" );


/*
	cuda_memset( deviceData->weights, 0, sizeof(real) * model->pSize, ERROR_MEMSET ); 
	int numBlocks = model->pSize / BLOCK_SIZE + 
						((model->pSize % BLOCK_SIZE) == 0 ? 0 : 1);	
   kerInitOneVector <<< numBlocks, BLOCK_SIZE>>> 
      ( vec, model->pSize );  
   cudaThreadSynchronize (); 
   cudaCheckError (); 
*/

/*
real nrm = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, vec, 1, &nrm) ); 
real alpha = 0.99 * (1200. / nrm ); 
cublasCheckError( cublasDscal( cublasHandle, model->pSize, &alpha, vec, 1 ) ); 


cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, vec, 1, &nrm) ); 
fprintf (stderr, "Initial guess norm is : %6.10f \n", nrm ); 
*/


	//call evaluate Model here. 
/*
	evaluateModel( model, deviceData, scratch, deviceData->weights,
			scratch->pageLckWorkspace, scratch->pageLckWorkspace, FULL_DATASET, TRAIN_DATA ); 
*/
	//sample the training set here. 
	real ll, mErr; 
	start = Get_Time (); 
	computeGradient( model, deviceData, scratch, deviceData->weights, 
			z, dx, gradient, &ll, &mErr, SAMPLED_DATASET ); 

fprintf( stderr, "Beginning to process hessian .... \n\n\n\n");
	hessianVec( model, deviceData, z, dx, vec, deviceData->weights, scratch, SAMPLED_DATASET);
	writeVector( scratch->nextDevPtr, model->pSize, "./hessianvec.txt", 0, scratch->nextHostPtr ); 

	total = Get_Timing_Info( start ); 

	fprintf( stderr, "Hessian Vec Time: %f\n", total * 1000 ); 

real temp = 0;
cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, 	
											scratch->nextDevPtr, 1, &temp )); 
fprintf( stderr, "Hessian Vec Norm: %6.15f \n", temp );
}
