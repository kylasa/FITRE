
#include <functions/swish.h>
#include <utilities/reduce.h>
#include <utilities/print_utils.h>
#include <device/device_defines.h>

#include <device/cuda_utils.h>
#include <core/errors.h>

/*
	y = x / (1 + exp( -x ) )
*/

GLOBAL void kerNNSwish( real *input, real *output, int count)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 
	real t = 0;

	if (idx < count){
		t = input[ idx ]; 	
		output[ idx ] = t / (1 + exp( -t )); 
	}
}

/* 
Computes the derivative of the Swish Function 

y = x / (1 + exp( -x ) )
y' = y + t( 1 - y)
t = 1 / (1 + exp( -x ))

*/

GLOBAL void kerNNBackPropSwish( real *zin, real *zout, real *output, int count)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 
	real t = 0; 	

	if (idx < count){
		t = 1. / (1. + exp( -zin[ idx ] )); 
		output[ idx ] *= (zout[ idx ] + t * ( 1. - zout[ idx ] )); 
	}
}

GLOBAL void kerNNROpSwish( real *input, real *output, int count)
{
	int idx = threadIdx.x * blockDim.x * blockIdx.x; 
	real t = 0; 
	real fx = 0; 

	if (idx < count) {
		t = 1. / (1. + exp( -input[ idx ] ) ); 		
		fx = input[ idx ] * t; 
		output[ idx ] *= ( fx + t * ( 1. - fx) ); 
	}
}

/*
Computes the second derivative of swish function

y'' =  (1 - t) * [ y' + t (1 - y ) ]

*/

GLOBAL void kerNNSecondDerivSwish( real *zin, real *zout, real *y_p,
	real *output, int count )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	real t = 0; 

	if (idx < count){
		t = 1. / (1. + exp( -zin[ idx ])); 
		output[ idx ] = (1. - t) * (zout[ idx ] + 2. * t * (1. - zout[idx])); 
	}

	/*
	real y_prime = 0;
	if (idx < count) {
		y_prime = y_p[ idx ]; 
		t = 1. / (1. + exp( -zin[ idx ])); 
		output[ idx ] = (1. - t) * (y_prime + t * (1. - zout[ idx ] ) ); 
	}
	*/
}
