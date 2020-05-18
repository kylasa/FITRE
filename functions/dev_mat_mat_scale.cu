
#include <functions/dev_mat_mat_scale.h>

#include <device/device_defines.h>
#include <core/datadefs.h>

GLOBAL void kerUtilsMatMatScale 
	(real *x, real *y, int numElements, real *out)
{
	int myIdx = blockDim.x * blockIdx.x + threadIdx.x; 

	if (myIdx < numElements) {
		out[ myIdx ] = x[ myIdx ] * y[ myIdx ]; 
	}
}
