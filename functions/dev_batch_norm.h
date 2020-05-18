
#ifndef __H_BATCH_NORMALIZATION__
#define __H_BATCH_NORMALIZATION__

#include <core/datadefs.h>
#include <device/device_defines.h>

#include <nn/nn_decl.h>

#define BATCH_NORM_EPSILON		1e-5
#define BATCH_NORM_MOMENTUM	0.9

void initMeanVariances( real *means, real *variances, int channels );


void computeBatchMeanVariance( real *input, int height, int width, int channels, int samples,
   real *output, real *batch_mean, real *batch_variance,  real *devPtr );

void computeZHat( real *input, int rows, int channels,
   real *means, real *variance, real epsilon, real *output );

void computeBatchNormForward( real *input, int height, int width, int channels, int samples,
   real *output, int meansOffset, int variancesOffset, real epsilon, real *devPtr, real *hostPtr, 
	EVAL_TYPE forTesting, real batchNormMomentum, int datasetMeanOffset, int datasetVarianceOffset ) ;

void computeBatchNormDerivative( 
   real *delta, int height, int width, int channels, int samples, 
   real *means, real *variance, real *output, 
   real *zout, real epsilon, real *devPtr, real *dx, real *hostPtr );

void computeROpBatchNormForward (real *z, real *z_out, real *rz_in, real *rz_out, real *devPtr, real *hostPtr, 
   real epsilon, int height, int width, int channels, int samples, int batchSize );

void computeROpBatchNormForwardTest (real *z, real *z_out, real *rz_in, real *rz_out, real *devPtr, real *hostPtr, 
   real epsilon, int height, int width, int channels, int samples, int batchSize );


void computeROpBatchNormBackward (
   real *z_in, real *z_out,
   real *rz_in, real *rz_out,
   real *delta, real *rdelta,
   real *dx_temp, real epsilon,
	real *means, real *variances, 
   int height, int width, int channels, int samples, int batchSize,
   real *devPtr, real *hostPtr );

#endif
