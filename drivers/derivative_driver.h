
#ifndef __H_DERIVATIVE_DRIVER__
#define __H_DERIVATIVE_DRIVER__

#include <core/structdefs.h>
#include <nn/nn_decl.h>

void runDerivativeTest( NN_MODEL *, DEVICE_DATASET *, SCRATCH_AREA *); 
void runCNNDerivativeTest( CNN_MODEL *, DEVICE_DATASET *, SCRATCH_AREA *); 

#endif
