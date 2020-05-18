
#include <functions/dev_mse_error.h> 

GLOBAL void kerNNComputeModelError
	(real *y, real *z, int numElements, real *out )
{
   int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 

   if (myIdx < numElements)
      out[ myIdx ] = pow( y[ myIdx ] -  z[ myIdx ], 2.0) ; 
	
}
