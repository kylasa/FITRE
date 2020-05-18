
#include <functions/dev_elem_sqr.h> 

GLOBAL void kerElemSqr
	(real *x, int numElements, real *out )
{
   int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 

   if (myIdx < numElements)
      out[ myIdx ] = x[ myIdx ] * x[ myIdx ]; 
	
}
