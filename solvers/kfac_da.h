
#ifndef __H_KFAC_DA__
#define __H_KFAC_DA__

#include <core/datadefs.h>
#include <core/structdefs.h>

#include <nn/nn_decl.h>

void augmentData( CNN_MODEL *model, DEVICE_DATASET *data, int offset, int currentBatchSize, 
	real *devPtr, real *hostPtr, int enable, DATASET_TYPE datasetType );


#endif
