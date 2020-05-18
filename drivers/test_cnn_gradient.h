
#ifndef __H_TEST_CNN_GRADIENT__
#define __H_TEST_CNN_GRADIENT__

#include <core/datadefs.h>
#include <core/structdefs.h>
#include <nn/nn_decl.h>

void testCNNGradient( CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch ); 

#endif
