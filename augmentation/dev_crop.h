
#ifndef __AUGMENT_CROP__
#define __AUGMENT_CROP__

#include <core/datadefs.h>

#include <core/datadefs.h>
#include <device/device_defines.h>

void randomCrop( real *input, int samples ,int height, int width, int channels, 
      int padding, real *output, real *devPtr, real *hostPtr, real defaultValue, real *probs );

#endif
