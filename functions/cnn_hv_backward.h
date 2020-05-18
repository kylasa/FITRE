
#ifndef __H_CNN_HV_BACKWARD__
#define __H_CNN_HV_BACKWARD__

#include <core/datadefs.h>
#include <core/structdefs.h>

#include <nn/nn_decl.h>

long cnnROpBackwardMemRequired( CNN_MODEL *model);

void cnnROpBackward( CNN_MODEL *model, DEVICE_DATASET *data, SCRATCH_AREA *scratch, 
      real *z, real *dx, real *lossFuncErrors, real *rError, real *rz,
      real *vector, real *hv, 
      int s, int curBatchSize, real *devPtr, real *hostPtr);
#endif
