
#ifndef __H_KFAC_TRUST_REGION_DRIVER__
#define __H_KFAC_TRUST_REGION_DRIVER__

#include <nn/nn_decl.h>
#include <core/structdefs.h>

void testKFACTrustRegion ( CNN_MODEL *model, DEVICE_DATASET *data, HOST_DATASET *host, 
		SCRATCH_AREA *scratch, real dampGamma, real trMaxRadius, int checkGrad, 
		int master, int slave, int inverseFreq, int epochs, int dataset, real regLambda, int initialization ); 

#endif
