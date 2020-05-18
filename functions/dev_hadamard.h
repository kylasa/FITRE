
#ifndef __H_HADAMARD__
#define __H_HADAMARD__

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void ker_hadamard (real *input, int count, real *output);

GLOBAL void ker_hadamard_2 (
   real *m1, real *m2, real *m3, real *m4, 
   int count, real *output );

#endif
