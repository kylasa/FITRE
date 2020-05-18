#ifndef __H_SWISH__
#define __H_SWISH__

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void kerNNSwish( real *input, real *output, int count);

GLOBAL void kerNNROpSwish( real *input, real *output, int count);

GLOBAL void kerNNBackPropSwish( real *zin, real *zout, real *output, int count);

GLOBAL void kerNNSecondDerivSwish( real *zin, real *zout, real *y_p,
	real *output, int count );

#endif
