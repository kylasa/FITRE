
#ifndef __H_EVAL_CONVOLUTION__
#define __H_EVAL_CONVOLUTION__

#include <core/datadefs.h>
#include <nn/nn_decl.h>

/*
void applyConvolutionLayer(real *input, int samples, 
   int in_channels, int height, int width, 
   int ksize, int pad, int stride, 
	int col_height, int col_width,
   real *weights, real *bias, 
   real *output, int out_channels, 
   int actFun, int pkSize, int pkStride, int pkPad, int poolFun, 
   real *devScratch, real *hostPtr);
*/

void applyConvolutionLayer(real *input, int samples, 
   int in_channels, int height, int width, 
   int ksize, int pad, int stride, 
   int col_height, int col_width, 
   real *weights, real *bias,
   real *output, int out_channels, 
   int actFun, int pkSize, int pkStride, int pkPad, int poolFun, 
   int poolOutHeight, int poolOutWidth,  
   BATCH_NORM_TYPES performBatchNorm, real epsilon, 
   int activationOffset, int poolOffset, int batchNormOffset, 
	int meansOffset, int variancesOffset, 
   real *devScratch, real *hostPtr, EVAL_TYPE forTesting, int runningMeansOffset, int runningVariancesOffset);

/*
void applyROpConvolutionLayer(real *input, real *prev_z, int offset, 
	real *z, real *rz_in, int samples, 
   int in_channels, int height, int width, 
   int ksize, int pad, int stride, 
   real *weights, real *bias,
   real *vweights, real *vbias, 
   real *rx, real *rz, int out_channels, 
   int actFun, int pkSize, int pkStride, int pkPad, int poolFun,  
   real *devScratch, real *hostPtr, int performBatchNorm);
*/

void applyROpConvolutionLayer(real *input, real *prev_z, int offset,
   real *z, real *rz_in, int samples,  
   int in_channels, int height, int width,
   int ksize, int pad, int stride,  
   int col_height, int col_width, 
   real *weights, real *bias,
   real *vweights, real *vbias, 
   real *rx, real *rz, int out_channels,
   int actFun, int pkSize, int pkStride, int pkPad, int poolFun,
   int activationOffset, int poolOffset, int batchNormOffset, int outputOffset,
   int convVolumn, int activationVolumn, int poolVolumn, int batchNormVolumn, 
   real *devScratch, real *hostPtr, BATCH_NORM_TYPES performBatchNorm, real epsilon, int batchSize);

void testImgConv(real *input, int in_channels, int height, int width, 
   int ksize, int pad, int stride, int out_channels, int poolFun, 
   real *weights, real *bias, int samples, real *devScratch);


#endif
