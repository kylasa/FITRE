
#ifndef __H_DEV_MAT_VEC_ADDITION__
#define __H_DEV_MAT_VEC_ADDITION__

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void kerUtilsAddColumnToMatrix 
      ( real *mat, int rows, int cols, real *vec);

#endif

