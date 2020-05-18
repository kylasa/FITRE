
#ifndef __H_DEV_HESSIAN_HELPERS__
#define __H_DEC_HESSIAN_HELPERS__

#include <device/device_defines.h>
#include <core/datadefs.h>


GLOBAL void eval_backprop_hessian( 
   real *z, real *Rdz, real *Rx, real *dx, real *out, int rows );

GLOBAL void eval_gauss_newton_backprop( 
   real *z, real *Rdz, real *out, int rows );

#endif
