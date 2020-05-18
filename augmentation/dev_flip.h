
#ifndef __AUGMENT_FLIP__
#define __AUGMENT_FLIP__

#include <core/datadefs.h>

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void ker_horizontal_flip (real *input, 
   int numImages, int height, int width, int channels, 
   real *output );

GLOBAL void ker_vertical_flip (real *input, 
   int numImages, int height, int width, int channels, 
   real *output );

GLOBAL void ker_random_flip (real *input, 
   int numImages, int height, int width, int channels, 
   real *output, real *probs );

void flipData( real *input, real *output, int samples, int channels, 
   int height, int width, real *probs, real *devPtr, real *hostPtr );

#endif
