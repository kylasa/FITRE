#ifndef __ADAGRAD_H__
#define __ADAGRAD_H__

#include <core/structdefs.h>
#include <core/datadefs.h>

#include <nn/nn_decl.h>
#include <solvers/params.h>

typedef struct adagrad_params {

	real step; 
	real eps;
	real lambda;

	int maxEpochs; 
	int maxProps; 
	real sampleSize; 

} ADAGRAD_PARAMS; 

typedef struct adagrad_out_params {

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

} ADAGRAD_OUT_PARAMS; 

void adagrad (NN_MODEL *model, DEVICE_DATASET *data, 
               SCRATCH_AREA *scratch, ADAGRAD_PARAMS *params);

#endif
