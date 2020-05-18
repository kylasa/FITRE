
#ifndef __H_SAMPLED_TRUST_CG__
#define __H_SAMPLED_TRUST_CG__

#include <core/structdefs.h>
#include <core/datadefs.h>

#include <nn/nn_decl.h>
#include <solvers/params.h>

void subsampledTrustRegionCG( NN_MODEL *nnModel, DEVICE_DATASET *data, 
		TRUST_REGION_PARAMS *params, SCRATCH_AREA *scratch ); 

#endif
