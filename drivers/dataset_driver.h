
#ifndef __H_DATASET_DRIVER__
#define __H_DATASET_DRIVER__

#include <core/datadefs.h>
#include <core/structdefs.h>

#include <nn/nn_decl.h>

int getGaussDataset(CNN_MODEL *curvesModel, 
         HOST_DATASET *host, DEVICE_DATASET *device);

int testDatasetRead (NN_MODEL *curvesModel, 
         HOST_DATASET *host, DEVICE_DATASET *device);

int getCIFAR10 (CNN_MODEL *model, HOST_DATASET *host, 
   DEVICE_DATASET *device, SCRATCH_AREA *scratch);

int getCIFAR100 (CNN_MODEL *model, HOST_DATASET *host, 
   DEVICE_DATASET *device, SCRATCH_AREA *scratch);

int getTinyImageNet (CNN_MODEL *model, HOST_DATASET *host, 
   DEVICE_DATASET *device, SCRATCH_AREA *scratch );

#endif
