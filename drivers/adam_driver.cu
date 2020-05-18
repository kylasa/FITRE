
#include <drivers/adam_driver.h>

#include <solvers/adam.h>

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


void initAdamParams( ADAM_PARAMS *params, int n )
{
	//sampled_tr_cg.m file. 
	params->step= 0.001; //learning rate
	params->beta1 = 0.9;
	params->beta2 = 0.999; 
	params->eps = 1e-8;  //eps  
	params->lambda = 0; 

	params->maxProps =  ULONG_MAX; 
	params->maxEpochs = 20; 
	params->sampleSize = floor( 256 );
}

void testAdam (NN_MODEL *model, DEVICE_DATASET *data, 
		SCRATCH_AREA *scratch ) {

	ADAM_PARAMS mParams; 

	//begin here
	fprintf( stderr, "Initiating the Trust Region Test now..... \n\n\n");
	initAdamParams( &mParams, data->trainSizeX );
	fprintf( stderr, "... Done parms initialization \n\n"); 

	//init weights to ZEROS
   cuda_memset( data->weights, 0, sizeof(real) * model->pSize, ERROR_MEMSET );  

	//init weights to Random Vector
	/*
   getRandomVector( model->pSize, NULL, scratch->nextDevPtr, RAND_NORMAL ); 
   copy_device( data->weights, scratch->nextDevPtr, sizeof(real) * model->pSize, 
            ERROR_MEMCPY_DEVICE_DEVICE ); 

	real scale = 0.25; 
	cublasCheckError( cublasDscal( cublasHandle, model->pSize, &scale, data->weights, 1 ));
	*/

	adam ( model, data, scratch, &mParams ); 

	fprintf( stderr, ".... Done testing of Adam \n\n\n" ); 
}
