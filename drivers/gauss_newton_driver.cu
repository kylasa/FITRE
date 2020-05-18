
#include <drivers/gauss_newton_driver.h>

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


void readVecFromFileGN( real *dev, real *host ) { 

   int rows = readVector( host, INT_MAX, "./weights.txt", 0, NULL);
   copy_host_device( host, dev, rows * sizeof(real), cudaMemcpyHostToDevice, 
      ERROR_MEMCPY_HOST_DEVICE );  
   
   fprintf( stderr, "Finished reading Vec (%d) from file \n", rows );  

   for (int i = 0; i < 10; i ++) fprintf( stderr, "%6.10f \n", host[i] );  
}

void initGNTrustRegionParams( TRUST_REGION_PARAMS *params, int n )
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

	//SolverType 
	params->hessianType = GAUSS_NEWTON;
}

void testGaussNewton (NN_MODEL *model, DEVICE_DATASET *data, 
		SCRATCH_AREA *scratch ) {

	TRUST_REGION_PARAMS trParams; 

	//begin here
	fprintf( stderr, "Initiating the Trust Region Test now..... \n\n\n");
	initGNTrustRegionParams( &trParams, data->trainSizeX );
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
