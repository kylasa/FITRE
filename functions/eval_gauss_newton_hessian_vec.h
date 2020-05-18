
#ifndef __H_GAUSS_NEWTON_HESSIAN_VEC__
#define __H_GAUSS_NEWTON_HESSIAN_VEC__

#include <nn/nn_decl.h>
#include <core/datadefs.h>
#include <core/structdefs.h>

void gaussNewtonHessianVec ( NN_MODEL *model, DEVICE_DATASET *data, 
         real *z, real *dx, real *vec, real *weights, SCRATCH_AREA *scratch, 
		DATASET_SIZE allData);

#endif
