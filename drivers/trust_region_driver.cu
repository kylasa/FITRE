
#include <drivers/trust_region_driver.h>

#include <solvers/sampled_trust_cg.h>
#include <solvers/params.h>

#include <core/errors.h>

#include <device/cuda_utils.h>
#include <device/gen_random.h>
#include <device/device_defines.h>
#include <device/handles.h>
#include <functions/dev_initializations.h>

#include <utilities/print_utils.h>

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>


void readVecFromFileTR( real *dev, real *host ) { 

   int rows = readVector( host, INT_MAX, "./weights.txt", 0, NULL);
   copy_host_device( host, dev, rows * sizeof(real), cudaMemcpyHostToDevice, 
      ERROR_MEMCPY_HOST_DEVICE );  
   
   fprintf( stderr, "Finished reading Vec (%d) from file \n", rows );  

   for (int i = 0; i < 10; i ++) fprintf( stderr, "%6.10f \n", host[i] );  
}

void initTrustRegionParams( TRUST_REGION_PARAMS *params, int n )
{
	//sampled_tr_cg.m file. 
	params->delta = 1200; 
	params->maxDelta = 12000; 
	params->eta1 = 0.8;
	params->eta2 = 1e-4;
	params->gamma1 = 2; 
	params->gamma2 = 1.2; 

	params->maxProps =  ULONG_MAX; 
	params->maxMatVecs = 1e15; 
	params->maxEpochs = 2000; 
	params->maxIters = 250; 

	//defaults from curves_autoencoder.m
	params->alpha = 0.01; 				// SGD Momentum
	params->hs = floor( 0.1 * n ); 	// Hessian sample size

	//no regularization 
	params->lambda = 0; 

	//loop variants here. 
	params->curIteration = 0; 
}

void testPointers( SCRATCH_AREA *scratch )
{
	real **devPtr = (real **)scratch->nextDevPtr; 
	real *host = scratch->nextHostPtr; 
	int *pageLck = (int *)scratch->nextPageLckPtr; 

	real **src = devPtr; 
	real **inv = src + 1; 
	real **next = inv + 1; 

	real *srcData = (real *)(next); 
	real *result = srcData + 9; 

	host[ 0 ] = 1; host[ 1 ] = 0; host[ 2 ] = 0; 
	host[ 3 ] = 0; host[ 4 ] = 1; host[ 5 ] = 0; 
	host[ 6 ] = 0; host[ 7 ] = 0; host[ 8 ] = 1; 

	copy_host_device( host, srcData, sizeof(real) * 9, cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	// Perform the inverse here. 
	cublasCheckError( cublasDmatinvBatched( cublasHandle, 9, 
									(const real **)src, 3, 
									inv, 3, 
									pageLck, 1 ) ); 

	copy_host_device( host, result, sizeof(real) * 9, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 

	for (int i = 0; i < 9; i ++) 
		fprintf( stderr, " %f , ", host[ i ]); 

}

void testTrustRegion (NN_MODEL *model, DEVICE_DATASET *data, 
		SCRATCH_AREA *scratch ) {

	TRUST_REGION_PARAMS trParams; 

	//testing pointers.... 
	fprintf( stderr, "Testing Pointers now.... \n" ); 
	testPointers( scratch ); 
	
	exit ( -1 ); 

	//begin here
	fprintf( stderr, "Initiating the Trust Region Test now..... \n\n\n");
	initTrustRegionParams( &trParams, data->trainSizeX );
	fprintf( stderr, "... Done parms initialization \n\n"); 

/*
   //set the weights to 0 here. 
#ifdef DEBUG_FIXED
   cuda_memset( data->weights, 0, sizeof(real) * model->pSize, ERROR_MEMSET );  
#endif


   getRandomVector( model->pSize, NULL, scratch->nextDevPtr, RAND_NORMAL ); 
   copy_device( data->weights, scratch->nextDevPtr, sizeof(real) * model->pSize, 
            ERROR_MEMCPY_DEVICE_DEVICE ); 

#ifdef DEBUG_FIXED
readVecFromFileTR( data->weights, scratch->nextHostPtr );
#endif
	
	real scale = 0.25; 
	cublasCheckError( cublasDscal( cublasHandle, model->pSize, &scale, data->weights, 1 ));
*/



/*
   int b = model->pSize;  
   int numBlocks = b / BLOCK_SIZE + 
               (( b % BLOCK_SIZE  == 0) ? 0 : 1 );  
   kerInitOneVector <<< numBlocks, BLOCK_SIZE>>> 
      ( data->weights, b );  
   cudaThreadSynchronize (); 
   cudaCheckError (); 
*/
   cuda_memset( data->weights, 0, sizeof(real) * model->pSize, ERROR_MEMSET );  

	subsampledTrustRegionCG( model, data, &trParams, scratch ); 

	fprintf( stderr, ".... Done testing of subsampledTrustRegion \n\n\n" ); 
}
