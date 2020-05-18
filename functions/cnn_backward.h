
#ifndef __H_CNN_BACKWARD__
#define __H_CNN_BACKWARD__


#include <core/structdefs.h>
#include <nn/nn_decl.h>

#include <core/datadefs.h>

long cnnBackwardMemRequired( CNN_MODEL *model );

void cnnBackward(CNN_MODEL *model, DEVICE_DATASET *data, 
      real *devPtr, real *z, real *gradient,
      real *dx, real *delta, real *delta_new, 
		int offset, int batchSize, real *hostPtr ) ;

#endif
