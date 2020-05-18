
#ifndef __H_MEM_DRIVER__
#define __H_MEM_DRIVER__

#include <core/structdefs.h>
#include <nn/nn_decl.h>

void testCudaMemcpy2D( SCRATCH_AREA *scratch );

void getMemRequired( CNN_MODEL *model );

#endif
