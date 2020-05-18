
#ifndef __H_DEVICE_LOGLIKELIHOOD__
#define __H_DEVICE_LOGLIKELIHOOD__

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void kerNNComputeLogLikelihoodSoftmax
	( real *y, real *z, int numElements, real *out);

GLOBAL void kerNNComputeLogLikelihoodLinear
	( real *y, real *z, int numElements, real *out);

GLOBAL void kerNNComputeLogLikelihoodLogistic
	(real *x, real *y, int numElements, real *out);

#endif
