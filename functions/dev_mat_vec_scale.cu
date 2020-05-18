

#include <functions/dev_mat_vec_scale.h>
#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void kerUtilsMatRowVecScale( 
		real *x, int rows, int cols, real *y, real *out )
{
	int myIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	int myColId = myIdx / rows;

	if (myIdx < (rows * cols)){
		out[ myIdx ] = x[ myIdx ] * y[ myColId ]; 
	}
}
