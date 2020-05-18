#ifndef __H_REDUCE__
#define __H_REDUCE__

#include <device/device_defines.h>
#include <core/datadefs.h>

GLOBAL void reduce(const real *, real *, const size_t );
GLOBAL void reduce_grid(const real *, real *, const size_t );
//template <unsigned int blockSize>
GLOBAL void reduce6( real *, real *, unsigned int); 
GLOBAL void kerNormInf( real *in, real *out, unsigned int count);


#endif
