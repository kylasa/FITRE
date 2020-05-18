
#ifndef __H_CNN_FORWARD_PASS__
#define __H_CNN_FORWARD_PASS__

#include <core/datadefs.h>
#include <core/structdefs.h>
#include <nn/nn_decl.h>

long cnnForwardMemRequired(CNN_MODEL *model );

real cnnForward(CNN_MODEL *model, DEVICE_DATASET *data, 
      SCRATCH_AREA *scratch, real *z, real *probs, real *errors, int s, int curBatchSize, 
		EVAL_TYPE forTesting );

#endif
