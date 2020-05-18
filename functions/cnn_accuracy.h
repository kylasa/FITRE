
#ifndef __CNN_ACCURACY_H__
#define __CNN_ACCURACY_H__

#include <nn/nn_decl.h>

#include <core/datadefs.h>
#include <core/structdefs.h>

#include <device/device_defines.h>


real computeAccuracy( real *probs, real *target, 
   int rows, int numClasses, real *devPtr, real *pageLckPtr );

void computeTestGeneralizationErrors( CNN_MODEL *model, DEVICE_DATASET *data, 
   SCRATCH_AREA *scratch, real *z, real *probs, real *errors, 
	real *likelihood, real *accuracy );

void computeTrainGeneralizationErrors( CNN_MODEL *model, DEVICE_DATASET *data, HOST_DATASET *host, 
   SCRATCH_AREA *scratch, real *z, real *probs, real *errors, 
   real *likelihood, real *accuracy  );

#endif
