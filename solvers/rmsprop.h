#ifndef __RMSPROP_H__
#define __RMSPROP_H__

#include <core/structdefs.h>
#include <core/datadefs.h>

#include <nn/nn_decl.h>
#include <solvers/params.h>

typedef struct rmsprop_params {

	real step; 
	real decayRate;
	real eps;
	real lambda;

	int maxEpochs; 
	int maxProps; 
	real sampleSize; 

} RMSPROP_PARAMS; 

typedef struct rmsprop_out_params {

	FILE *out; 
	
	unsigned long int numProps; 
	int iteration; 
	real iter_time; 
	real total_time; 
	real normGrad;

	real trainLL; 
	real trainModelErr; 
	real testLL; 
	real testModelErr; 

} RMSPROP_OUT_PARAMS; 

void rmsprop (NN_MODEL *model, DEVICE_DATASET *data, 
               SCRATCH_AREA *scratch, RMSPROP_PARAMS *params);

#endif
