
#include <functions/dev_hessian_helpers.h>

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void eval_backprop_hessian( 
	real *z, real *Rdz, real *Rx, real *dx, real *out, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < count) {
		out[ idx ] = (1.0 - z[ idx ])	 * z[ idx ] * Rdz[ idx ] + 
						Rx[ idx ] * (1.0 - 2 * z[ idx ]) * dx[ idx ]; 
	}
}

GLOBAL void eval_gauss_newton_backprop( 
	real *z, real *Rdz, real *out, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < count) {
		out[ idx ] = (1.0 - z[ idx ])	 * z[ idx ] * Rdz[ idx ];
	}
}
