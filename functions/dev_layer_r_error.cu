
#include <functions/dev_layer_r_error.h>

//based on the input... to activation layer
GLOBAL void kerNNROpSOFTPLUS( 
	real *err, real *xi, int count )
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 

	if (idx < count ){
		/*
		real t = 1 + exp( - xi[ idx ] )	;
		t = 1. / t;  
		err *= t * ( 1. - t ); 
		*/
      //real t = exp( xi[ idx ] );  
      //err[ idx ] *= ((t - 1.0) / t) * (t); 
      //err[ idx ] *= ((t - 1.0) / t);

		err[ idx ] *= 1. / (1. + exp( - xi[ idx ] ) ); 
	}
}

GLOBAL void kerNNROpSOFTPLUSWithZ( 
		real *rError, real *delta, real *rz, real *xi, int count ){

	int idx = threadIdx.x + blockDim.x * blockIdx.x; 

	if (idx < count){
		//real t = exp( xi[ idx ] ); 
		real t_minus = exp( -xi[ idx ] ); 
		
		//rError[ idx ] += ( (t - 1.0) / t ) * (1./t) * rz[ idx ] * delta[ idx ]; 
		//rError[ idx ] += (1./(1. + t)) * (1./(1. + t_minus)) * rz[ idx ] * delta[ idx ]; 
		//rError[ idx ] += (1./(1. + t_minus)) * (t_minus/(1. + t_minus)) * rz[ idx ] * delta[ idx ]; 
		rError[ idx ] += (1./(1. + t_minus)) * (t_minus/(1. + t_minus)) * rz[ idx ] * delta[ idx ]; 
	}
}
