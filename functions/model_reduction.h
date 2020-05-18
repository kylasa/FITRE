
#ifndef __MODEL_REDUCTION_H__
#define __MODEL_REDUCTION_H__

#include <core/datadefs.h>
#include <core/structdefs.h>

#include <nn/nn_decl.h>

real computeQuadraticModel( CNN_MODEL *model, DEVICE_DATASET *data, 
   real *z, real *probs, real *lossFuncErrors, real *dx, 
   real *gradient, real *vector, real delta, int offset, int curBatchSize, real weightDecay, 
   real *devPtr, real *hostPtr, real *pageLckPtr );

#endif
