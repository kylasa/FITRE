
#include <functions/dev_mat_vec_addition.h> 

#include <device/device_defines.h>

GLOBAL void kerUtilsAddColumnToMatrix 
		( real *mat, int rows, int cols, real *vec) 
{
		int tid = blockIdx.x * blockDim.x + threadIdx.x; 

		int myRow = tid % rows;  
		int myCol = tid / rows;  

		if ( (myRow < rows) && (myCol < cols) )
			mat[ myRow + myCol * rows ] += vec[ myRow ] ;

}
