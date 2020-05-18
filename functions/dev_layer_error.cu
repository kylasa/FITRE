
#include <functions/dev_layer_error.h>

GLOBAL void kerNNComputeLayerError 
		( real *y, real *z, int numElements, real scale, real *out)
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (myIdx < numElements)
		out[ myIdx ] = scale * (y[ myIdx ] - z[ myIdx ]); 
}

GLOBAL void kerNNBackPropLogisticErrors
		( real *err, real *xi, int numElements)
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (myIdx < numElements)
		err[ myIdx ] = err[ myIdx ] * xi [ myIdx ] * ( 1.0 - xi[ myIdx ] );  
}

GLOBAL void kerNNBackPropTanHErrors
		( real *err, real *xi, int numElements)
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (myIdx < numElements)
		err[ myIdx ] *= (1.0 + xi [ myIdx ]) * ( 1.0 - xi[ myIdx ] );  
}

//First derivative of softplus function here. 
// f(x) = log( 1 + exp(x) )
// f'(x) = 1 / (1 + exp(-x) )
GLOBAL void kerNNBackPropSOFTPLUS( 
		real *err, real *xi, int count ){ 
		
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	real t; 
	if ( myIdx < count ) {
		/*
		//err[ myIdx ] *= 1.0 / ( 1.0 + exp( - xi[ myIdx ] ) ); 
		t = exp( xi[ myIdx ] ); 
		err[ myIdx ] *= (t - 1.0) / t; 
		*/
		err[ myIdx ] *= 1./(1. + exp( - xi[ myIdx ] ) ); 
	}

} 
