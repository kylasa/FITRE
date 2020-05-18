#ifndef __MOMENTUM_SGD_H__
#define __MOMENTUM_SGD_H__

#include <core/structdefs.h>
#include <core/datadefs.h>

#include <nn/nn_decl.h>
#include <solvers/params.h>

typedef struct momentum_params {

	real alpha; 
	real beta;
	real lambda;

	int maxEpochs; 
	int maxProps; 
	real sampleSize; 

} MOMENTUM_PARAMS; 

typedef struct sgd_out_params {

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

} SGD_OUT_PARAMS; 

void momentum_sgd (NN_MODEL *model, DEVICE_DATASET *data, 
               SCRATCH_AREA *scratch, MOMENTUM_PARAMS *params);

#endif
