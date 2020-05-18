
#ifndef __H_R_LAYER_ERROR__
#define __H_R_LAYER_ERROR__

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void kerNNROpSOFTPLUS( 
      real *err, real *xi, int count );

GLOBAL void kerNNROpSOFTPLUSWithZ( 
      real *rError, real *delta, real *rz, real *xi, int count );

#endif
