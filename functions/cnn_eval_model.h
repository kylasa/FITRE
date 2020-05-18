
#ifndef __H_CNN_MODEL_EVAL__
#define __H_CNN_MODEL_EVAL__

#include <core/datadefs.h>
#include <core/structdefs.h>
#include <nn/nn_decl.h>

real evaluateCNNModel (CNN_MODEL *model, DEVICE_DATASET *data, 
   SCRATCH_AREA *scratch, real *z, real *probs, real *lossFuncErrors, 
   int offset, int curBatchSize);

#endif
