
#ifndef __H_DEV_LAYER_ERROR__
#define __H_DEV_LAYER_ERROR__

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void kerNNComputeLayerError 
      ( real *x, real *y, int numElements, real scale, real *out);

GLOBAL void kerNNBackPropLogisticErrors
      ( real *err, real *xi, int numElements);

GLOBAL void kerNNBackPropTanHErrors
      ( real *err, real *xi, int numElements);

GLOBAL void kerNNBackPropSOFTPLUS( 
      real *err, real *xi, int count );

#endif
