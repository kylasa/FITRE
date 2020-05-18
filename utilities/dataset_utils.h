
#ifndef __H_DATASET_UTILS__
#define __H_DATASET_UTILS__

#include <core/datadefs.h>
#include <core/structdefs.h>

void selectHostMatrix( HOST_DATASET *host, real *trainX, real *trainY, int numSamples, int *indices, int offset, real *hostPtr ); 

#endif
