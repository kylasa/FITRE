#ifndef __H_CONVOLUTION_DRIVER__
#define __H_CONVOLUTION_DRIVER__

#include <core/structdefs.h>
#include <nn/nn_decl.h>

void testConvolution (CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch ); 

void testBackPropConvolution (CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch ); 

#endif
