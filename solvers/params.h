#ifndef __H_PARAMS__
#define __H_PARAMS__

#include <core/datadefs.h>

typedef enum hessian_type {
	TRUE_HESSIAN = 0, 
	GAUSS_NEWTON
} HESSIAN_TYPE; 

typedef struct trust_region_params { 

	real delta; 
	real minDelta;
	real maxDelta; 
	real eta1; 
	real eta2;
	real gamma1;
	real gamma2;
	unsigned long int maxProps; 
	int maxMatVecs; 
	int maxEpochs; 
	int maxIters; 

	int hessianType;
	
	real alpha; 
	int curIteration; 
	real hs; 

	real lambda;

} TRUST_REGION_PARAMS;

typedef struct steihaug_params {
	
	real 	tolerance; 
	int	maxIterations; 
	real	alpha; 

} STEIHAUG_PARAMS; 

#endif
