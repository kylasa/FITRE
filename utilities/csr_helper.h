
#ifndef __H_CSR_HELPER__
#define __H_CSR_HELPER__

#include <core/datadefs.h>
#include <core/sparsedefs.h>

void convertToCSR( SparseDataset *data, int rows, int cols, real *devPtr); 

#endif
