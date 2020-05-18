
#include <device/extract_columns.h>

#include <device/device_defines.h>
#include <core/datadefs.h>


GLOBAL void kerExtractColumns( 
	real *tgt, real *src, int *indices, int rows, int numElements) 
{
	int stride = gridDim.x * blockDim.x; 
	int tid = blockDim.x * blockIdx.x + threadIdx.x; 

	for (int j = tid; j < numElements; j += stride) {
		int colId = j / rows; 
		int rowId = j % rows; 

		tgt[ j ] = src[ indices[ colId ] * rows + rowId ]; 
	}
}
