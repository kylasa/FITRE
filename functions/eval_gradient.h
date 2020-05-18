
#ifndef __H_EVAL_GRADIENT__
#define __H_EVAL_GRADIENT__

#include <core/datadefs.h>
#include <nn/nn_decl.h>
#include <core/structdefs.h>

void nnForwardPass(NN_MODEL *model, DEVICE_DATASET *data, 
            real *weights, real *z, real *errTerm, 
            real *logLikelihood, real *modelError, 
            real *hostPtr, real *devPtr, DATASET_SIZE dSize, DATA_TYPE d);

void evaluateModel (NN_MODEL *model, DEVICE_DATASET *data, SCRATCH_AREA *scratch,
            real *w, real *logLikelihood, real *modelError, DATASET_SIZE dSize, DATA_TYPE s ); 

void computeGradient( NN_MODEL *model, 
				DEVICE_DATASET *data, SCRATCH_AREA *scratch, real *weights,
				real *z, real *dx, real *gradient, real *ll, real *err, DATASET_SIZE dSize);

void applyLayerActivation (int actFunction, 
         real *W, int wRows, int wCols,  
         real *b, int bRows, 
         real *z, int zRows, int zCols, 
         real *output, real *scratch, real *hostPtr );

#endif
