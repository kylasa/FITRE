
#ifndef __H_BACKPROP_CONVOLUTION__
#define __H_BACKPROP_CONVOLUTION__

#include <core/datadefs.h>

void reshapeMatrix( real *input, int rightChannels, int leftChannels, 
      int chunk, real *output );

void backpropConvolution( real *delta, int dHeight, int dWidth, int dChannels,
   real *filter, int fHeight, int fWidth,
   int height, int width, int padding, int channels,
    int samples, real *delta_1, real *devPtr, real *hostPtr);


#endif
