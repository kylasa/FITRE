#ifndef __SOFTMAX_LOSS_H__
#define __SOFTMAX_LOSS_H__

#include <core/datadefs.h>

void computeDistributionErrors ( real *probs, real *distTarget, real *distError, 
   int rows, int num_classes, real *devPtr, real *hostPtr );

real computeProbsAndCrossEntropyLoss( real *input, real *target, 
   int rows, int num_classes, 
   real *probs, real *devPtr, real *pageLckPtr, real *host );

void computeCrossEntropyError( real *input, int rows, int cols, 
   real *target, real *errors );

void computeROpCrossEntropyError_simple( real *rz, real *probs, 
   int rows, int numclasses, 
   real *rError );

void computeROpCrossEntropyError( real *rz, real *probs,  real *target, 
   int rows, int numclasses, real *rError, real *devPtr );

#endif
