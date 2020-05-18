
#ifndef __KFAC_TRUST_REGION_H__
#define __KFAC_TRUST_REGION_H__

#include <core/structdefs.h>
#include <core/datadefs.h>

#include <nn/nn_decl.h>
#include <solvers/params.h>
#include <solvers/kfac_structs.h>

void subsampledTrustRegionKFAC( CNN_MODEL *model, DEVICE_DATASET *data, HOST_DATASET *host, 
      KFAC_CURVATURE_INFO *kfacInfo, TRUST_REGION_PARAMS *trParams, 
      SCRATCH_AREA *scratch, int master, int slave ) ;

#endif
