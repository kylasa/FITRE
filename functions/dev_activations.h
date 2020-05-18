
#ifndef __H_DEV_ACTIVATIONS__
#define __H_DEV_ACTIVATIONS__

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void kerNNApplyRELU
      ( real *input, int numElements ); 

GLOBAL void kerNNApplySOFTPLUS
      ( real *input, int numElements );

GLOBAL void kerNNApplyELU
      ( real *input, int numElements, real a); 

GLOBAL void kerNNApplyLogistic
      ( real *input, int numElements);

GLOBAL void kerNNApplyTanH
      ( real *input, int numElements);

GLOBAL void kerNNApplyExp
      ( real *input, int numElements);

GLOBAL void kerInitVector
      ( real *vec, int numElements, real val) ;

GLOBAL void kerNNComputeSoftmax
      ( real *input, int rows, int cols, real *vec);

#endif
