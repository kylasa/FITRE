
#include <functions/dev_hadamard.h>

GLOBAL void ker_hadamard( real *input, int count, real *output)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	if (idx < count){
		output[ idx ] *= input[ idx ];
	}
}

GLOBAL void ker_hadamard_2 (
	real *m1, real *m2, real *m3, real *m4, 
	int count, real *output )
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	if (idx < count){
		output[ idx ] = m1[ idx ] * m2[ idx ] + m3[ idx ] * m4[ idx ];
	}
}
