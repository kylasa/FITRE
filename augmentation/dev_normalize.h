
#ifndef __AUGMENT_NORMALIZE__
#define __AUGMENT_NORMALIZE__

#include <core/datadefs.h>
#include <core/structdefs.h>
#include <device/device_defines.h>

   
void normalizeCIFAR10( real *input, int samples, int channels, int height, int width, 
   real *devPtr, real *hostPtr, DATASET_TYPE datasetType );


#endif
