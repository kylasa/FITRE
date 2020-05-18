
#include <functions/dev_elem_scale.h> 

GLOBAL void kerElemInvScale
	(real *x, real *y, real eps, int numElements, real *out )
{
   int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 

   if (myIdx < numElements)
      out[ myIdx ] = x[ myIdx ] / (sqrt( y[ myIdx ] ) + eps);
}
