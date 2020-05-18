
#ifndef __AUGMENT_ROTATE__
#define __AUGMENT_ROTATE__

#include <core/datadefs.h>

#include <core/datadefs.h>
#include <device/device_defines.h>



GLOBAL void ker_rotate_right ( real *input, 
   int numImages, int height, int width, int channels, 
   real *output );

GLOBAL void ker_rotate_left ( real *input, 
   int numImages, int height, int width, int channels, 
   real *output );

void rotate( real *input, real *output, int samples, int channels, 
   int height, int width, real *probs, real *devPtr, real *hostPtr );

#endif
