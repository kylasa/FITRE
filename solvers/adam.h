#ifndef __ADAM_H__
#define __ADAM_H__

#include <core/structdefs.h>
#include <core/datadefs.h>

#include <nn/nn_decl.h>
#include <solvers/params.h>

typedef struct adam_params {

	real step; 
	real beta1, beta2; 
	real eps;
	real lambda;

	int maxEpochs; 
	int maxProps; 
	real sampleSize; 

} ADAM_PARAMS; 

typedef struct adam_out_params {

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

} ADAM_OUT_PARAMS; 

void adam (NN_MODEL *model, DEVICE_DATASET *data, 
               SCRATCH_AREA *scratch, ADAM_PARAMS *params);

#endif
