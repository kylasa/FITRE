#ifndef __H_CNN_HESSIAN_VEC__
#define __H_CNN_HESSIAN_VEC__

#include <core/structdefs.h>
#include <core/datadefs.h>

#include <nn/nn_decl.h>

void cnnHv ( CNN_MODEL *model, DEVICE_DATASET *data, 
   real *z, real *probs, real *lossFuncErrors, real *dx, 
   real *vector, real *hv, 
   int s, int curBatchSize,
   real *devPtr, real *hostPtr, real weightDecay );

#endif
