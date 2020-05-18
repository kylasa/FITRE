
#include <drivers/gradient_driver.h>

#include <device/query.h>
#include <device/cuda_environment.h>
#include <device/device_defines.h>
#include <device/cuda_utils.h>


#include <functions/eval_gradient.h>

#include <core/datadefs.h>
#include <core/structdefs.h>
#include <core/errors.h>

#include <drivers/dataset_driver.h> 

#include <utilities/print_utils.h>
#include <utilities/utils.h>

#include <nn/read_nn.h>

#include <stdio.h>
#include <stdlib.h>

#include <time.h>



void testModel( NN_MODEL *model, DEVICE_DATASET *deviceData, 
				SCRATCH_AREA *scratch)
{
	real start, total;

	//Model Evaluattion here.
	real *logLikelihood = scratch->pageLckWorkspace;
	real *modelError = logLikelihood + 1;

	//set the weights to 0 here. 
	cuda_memset( deviceData->weights, 0, sizeof(real) * model->pSize, 
		ERROR_MEMSET ); 

	evaluateModel( model, deviceData, scratch, deviceData->weights,
						logLikelihood, modelError, FULL_DATASET, TRAIN_DATA ); //use entire dataset

	fprintf( stderr, "ModelEvaluation Done... \n"
						"\t\t LogLikelihood = %f \n"
						"\t\t ModelError = %f \n", *logLikelihood, *modelError ); 
}
