#ifndef __H_UTILS_NN__
#define __H_UTILS_NN__

#include <nn/nn_decl.h>
#include <core/structdefs.h>

void getDimensions( int height, int width, int padding, int stride, int kernel,
   int *h, int *w);


void computeParamSize( CNN_MODEL *model );
void computeWeightOffsets( CNN_MODEL *model );
void computeWeightBiasOffsets( CNN_MODEL *model );
void computeZOffsets( CNN_MODEL *model, int height, int width, int batchSize );


#endif
