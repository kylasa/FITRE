#ifndef __H_CNN_HV_FORWARD__
#define __H_CNN_HV_FORWARD__

#include <core/datadefs.h>
#include <core/structdefs.h>
#include <nn/nn_decl.h>

long cnnROpForwardMemRequired(CNN_MODEL *model); 

void cnnROpForward(CNN_MODEL *model, DEVICE_DATASET *data, 
   SCRATCH_AREA *scratch, real *z, real *vector, 
   real *rz, int s, int curBatchSize, 
   real *devPtr, real *hostPtr ) ;

#endif
