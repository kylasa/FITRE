
#include <functions/dev_activations.h> 

#include <core/datadefs.h>
#include <device/device_defines.h>

//x = max( 0, x)
GLOBAL void kerNNApplyRELU
		( real *input, int numElements )
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	real x = 0; 

	//int imgIdx = myIdx / (channels * sizePerSlice);
	//int myChannel = (myIdx / sizePerSlice) % channels; 

	if (myIdx < numElements){
		x = input[ myIdx ]; 
		if (x < 0) input [myIdx] = 0;
		//input[ myIdx ] += bias[ myChannel ];
	}
}

GLOBAL void kerNNApplySOFTPLUS
		( real *input, int numElements )
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	//int imgIdx = blockIdx.y; 
	real x = 0; 

	//x = log( 1 + exp(x) )
	if (myIdx < numElements){
		x = input[ myIdx ]; 
		if (x < 20)
			input[ myIdx ] = log( 1 + exp( x ) );
		else
			input[ myIdx ] = x;
	}
}

GLOBAL void kerNNApplyELU
		( real *input, int numElements, real a )
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	real x = 0; 

	if (myIdx < numElements){
		x = input[ myIdx ]; 
		if (x < 0){
			input[ myIdx ] = a * (exp(x) - 1);
		}
	}
}



GLOBAL void kerNNApplyLogistic
		( real *input, int numElements)
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	real x = 0;
	
	if (myIdx < numElements) {
		x = input[ myIdx ]; 
		input[ myIdx ] = 1. / (1. + exp( - x ) ); 		
	}
}

GLOBAL void kerNNApplyTanH
		( real *input, int numElements)
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (myIdx < numElements) {
		input[ myIdx ] = tanh( input[ myIdx ] ) ; 		
	}
}

//divide exp(.) by sum of column exp(.). 
GLOBAL void kerNNApplyExp
		( real *input, int numElements)
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (myIdx < numElements) {
		input[ myIdx ] = exp( input[ myIdx ] ) ; 		
	}
}

GLOBAL void kerInitVector
		( real *vec, int numElements, real val) 
{
		int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
		if (myIdx < numElements) vec[ myIdx ] = val;
}

GLOBAL void kerNNComputeSoftmax
		( real *input, int rows, int cols, real *vec) 
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	int myRow = 0;  //TODO
	int myCol = 0; //TODO

	if (myRow < rows && myCol < cols) {
		input[ myIdx ] /= vec[ myRow ]; 	
	}
}
