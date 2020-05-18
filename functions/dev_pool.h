
#ifndef __H_POOLING__
#define __H_POOLING__

#include <core/datadefs.h>

void applyPoolLayer( real *input, int samples, int channels, int height, int width, 
   int ksize, int stride, int padding, int poolFun, real *output, real rOpScale );

void applyROpPoolLayer( real *input, real *z, int samples, int channels, int height, int width, 
   int ksize, int stride, int padding, int poolFun, real *output, real rOpScale );

void computePoolDerivative( real *delta, int sSize,
   int channels, real *output, int kernelSize, int samples );

void computePoolDerivative_in( real *delta, int outSize, int inSize,
   int channels, real *output, int kernelSize, int stride, int padding, int samples );

/*
void computeMaxPoolDerivative( real *delta, real *z_in, int outSize, int inSize,
   int channels, real *output, int kernelSize, int stride, int padding, int samples );
*/

void computeMaxPoolDerivative( real *delta, real *z_in, int channels, 
   int height, int width, int kernel, int stride, int padding, 
   int p_height, int p_width, int samples, real *output );

#endif
