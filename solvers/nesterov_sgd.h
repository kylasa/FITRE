#ifndef __NESTEROV_SGD_H__
#define __NESTEROV_SGD_H__

#include <core/structdefs.h>
#include <core/datadefs.h>

#include <nn/nn_decl.h>
#include <solvers/params.h>

typedef struct nesterov_params {

	real step; 
	real momentum;
	real lambda;

	int maxEpochs; 
	int maxProps; 
	real sampleSize; 

} NESTEROV_PARAMS; 

typedef struct nest_out_params {

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

} NESTEROV_OUT_PARAMS; 

void nesterov_sgd (NN_MODEL *model, DEVICE_DATASET *data, 
               SCRATCH_AREA *scratch, NESTEROV_PARAMS *params);

#endif
