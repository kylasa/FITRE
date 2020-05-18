
#ifndef __H_CNN_GRADIENT__
#define __H_CNN_GRADIENT__

#include <core/structdefs.h>
#include <nn/nn_decl.h>
#include <core/datadefs.h>

real computeCNNGradient(CNN_MODEL *model, DEVICE_DATASET *data, SCRATCH_AREA *scratch,
   real *z, real *dx, real *probs, real *lossFuncErrors, 
   real *gradient, 
   int offset, int curBatchSize, real weightDecay );

#endif
