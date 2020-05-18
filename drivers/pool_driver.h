#ifndef __H_POOL_DRIVER__
#define __H_POOL_DRIVER__

#include <core/structdefs.h>
#include <nn/nn_decl.h>

void testPoolDerivative(CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch ); 

void testPoolForwardPass (CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch ); 

#endif
