
#include <functions/dev_elem_sqr_decay.h> 

GLOBAL void kerElemSqrDecay
	(real *x, real *y, real decay, int numElements, real *out )
{
   int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 

   if (myIdx < numElements)
      out[ myIdx ] = decay * x[ myIdx ] + (1. - decay) * y[ myIdx ] * y[ myIdx ];
}
