
#include <utilities/alloc_sparse_mat.h>

#include <device/cuda_utils.h>
#include <core/errors.h>

void allocSparseMatrix( SparseDataset *spData, int rows, int nnz )
{
   cuda_malloc( (void **) &spData->P,     rows * sizeof(int), 1, ERROR_MEM_ALLOC ); 
   cuda_malloc( (void **) &spData->sortedVals,  rows * sizeof(real), 0, ERROR_MEM_ALLOC ); 
   cuda_malloc( (void **) &spData->rowPtr,   rows * sizeof(int), 0, ERROR_MEM_ALLOC ); 
   cuda_malloc( (void **) &spData->colPtr,   rows * sizeof(int), 0, ERROR_MEM_ALLOC ); 
   cuda_malloc( (void **) &spData->valPtr,   rows * sizeof(real), 0, ERROR_MEM_ALLOC ); 
   cuda_malloc( (void **) &spData->rowCsrPtr,   (rows + 1) * sizeof(int), 1, ERROR_MEM_ALLOC ); 
}
