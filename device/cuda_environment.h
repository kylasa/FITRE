#ifndef _H_CUDA_ENVIRONMENT__
#define _H_CUDA_ENVIRONMENT__

#include <core/structdefs.h>

void cuda_env_init (SCRATCH_AREA *, int);
void cuda_allocate_workspace (SCRATCH_AREA *scratch, int device);

void cuda_env_cleanup (SCRATCH_AREA *);

#endif
