#ifndef __H_CNN_DRIVER__
#define __H_CNN_DRIVER__

#include <core/structdefs.h>
#include <nn/nn_decl.h>

void testCNN (CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch ); 

#endif
