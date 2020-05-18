
#ifndef __H_DEV_MAT_VEC_SCALE__
#define __H_DEV_MAT_VEC_SCALE__

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void kerUtilsMatRowVecScale( 
      real *x, int rows, int cols, real *y, real *out );

#endif
