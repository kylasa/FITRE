
#include <functions/dev_loglikelihood.h>

GLOBAL void kerNNComputeLogLikelihoodSoftmax
	( real *y, real *z, int numElements, real *out)
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (myIdx < numElements)
		out[ myIdx ] = y[ myIdx ] * log( z[ myIdx ] ); 
}

GLOBAL void kerNNComputeLogLikelihoodLinear
	( real *y, real *z, int numElements, real *out)
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (myIdx < numElements)
		out[ myIdx ] =  pow(y[ myIdx ] -  z[ myIdx ], 2.0); 
}

GLOBAL void kerNNComputeLogLikelihoodLogistic
	(real *x, real *y, int numElements, real *out)
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	real lx;
	real lxx; 

	if (myIdx < numElements){

		lx = x[ myIdx ];
		if ( lx >= 0) 
			lxx = 1; 
		else 
			lxx = 0;

		out[ myIdx ] =  lx * ( y[ myIdx ] - lxx ) 
							- log(1 + exp(lx - 2.0 * lx * lxx ));
	}
		
}
