
#ifndef __KFAC_NATURAL_GRADIENT_H__
#define __KFAC_NATURAL_GRADIENT_H__

#include <core/datadefs.h>
#include <nn/nn_decl.h>
#include <solvers/kfac_structs.h>

void computeNaturalGradient (CNN_MODEL *model, 
   KFAC_CURVATURE_INFO *kfacInfo,
   real *devPtr, real *hostPtr );

#endif
