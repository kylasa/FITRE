
#ifndef __H_SPARSE_DEFS__
#define __H_SPARSE_DEFS__


#include <core/datadefs.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <host_defines.h>


typedef struct spData {
   int *rowPtr, *colPtr, *rowCsrPtr; 
   real *valPtr; 

   int nnz; 

   real *sortedVals;
   int *P; 

   cusparseMatDescr_t descr;  

} SparseDataset; 


#endif
