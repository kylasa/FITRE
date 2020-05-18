
#ifndef __H_TRANSPOSE__
#define __H_TRANSPOSE__

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void ker_transpose( real *input, int count, 
   int channels, int height, int width, int samples, real *output);

GLOBAL void ker_transpose_rc( real *input, int count, 
   int channels, int height, int width, int samples, real *output);

#endif
