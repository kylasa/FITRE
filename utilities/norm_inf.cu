
#include <utilities/norm_inf.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/reduce.h>

#include <core/errors.h>

void norm_inf_host( real *dev, real *host, int len, real *res, real *idx, real *scratch)
{
	copy_host_device( host, dev, len * sizeof(real), 
		cudaMemcpyDeviceToHost, ERROR_DEBUG);   

	*res = fabs( host[0] ); *idx = 0; 
	for (int i = 1; i < len; i ++) {
		if ( *res < fabs( host[ i ] ) ){ 
			*res = fabs( host[ i ] );
			*idx = i; 
		}
	}
}


void norm_inf( real *dev, real *host, int len, real *res, real *idx, real *scratch)
{
/*
	int blocks = len /  (8 * BLOCK_SIZE) + 
					(( len % (8 * BLOCK_SIZE)) == 0 ? 0 : 1 ); 

	kerNormInf <<< blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(real) >>> 
		( dev, scratch, len); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	//Now fine the greattest among the number of blocks results. 
	kerNormInf <<< 1, blocks, BLOCK_SIZE * sizeof(real) >>> 
			( scratch, res, blocks ); 	
	cudaThreadSynchronize (); 
	cudaCheckError (); 

*/

	copy_host_device( host, dev, len * sizeof(real), 
		cudaMemcpyDeviceToHost, ERROR_DEBUG);   

	*res = fabs( host[0] ); *idx = 0; 
	for (int i = 1; i < len; i ++) {
		if ( *res < fabs( host[ i ] ) ){ 
			*res = fabs( host[ i ] );
			*idx = i; 
		}
	}
}
