
#include <functions/dev_initializations.h>

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void kerInitOneVector
	( real *input, int numElements )
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (myIdx < numElements)
		input[ myIdx ] = 1.0;
}
