
#ifndef __KFAC_UTILS_H__
#define __KFAC_UTILS_H__

#include <core/structdefs.h>
#include <core/datadefs.h>
#include <nn/nn_decl.h>

#include <solvers/kfac_structs.h>

void computeKFACStorageIndices (CNN_MODEL *model, KFAC_CURVATURE_INFO *kfacInfo) ;

void updateDeltasWithDistribution (CNN_MODEL *model,
   real *z, real *dx, DEVICE_DATASET *data, int offset, int samples, int numClasses, 
   real *devPtr, real *hostPtr) ;

#endif
