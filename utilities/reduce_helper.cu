

#include <utilities/reduce_helper.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>
#include <device/reduce.h>

#include <functions/dev_initializations.h>

void reduce_cublas( real *input, real *output, real *temp, int rows, int cols )
{
/*
	int blocks = (rows * cols) / (BLOCK_SIZE * 2 * 8 ) + 
			(((rows * cols) % (BLOCK_SIZE * 2 * 8) == 0) ? 0 : 1);
   reduce6<<< blocks, BLOCK_SIZE, BLOCK_SIZE* sizeof(real)  >>> 
      (input, temp, rows * cols);  
   cudaThreadSynchronize (); 
   cudaCheckError (); 
   
   reduce6<<< 1, BLOCK_SIZE, BLOCK_SIZE * sizeof(real)>>> 
      ( temp,  output, blocks); 
   cudaThreadSynchronize (); 
   cudaCheckError (); 
*/

   //one Vectors
   int blocks = cols / BLOCK_SIZE +
                  (( cols % BLOCK_SIZE ) == 0 ? 0 : 1);
   kerInitOneVector <<< blocks, BLOCK_SIZE >>>
         (temp, cols);
   cudaThreadSynchronize ();
   cudaCheckError ();

   real alpha = 1, beta = 0;
   cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N,
                                    rows, cols, &alpha,
                                    input, rows, temp, 1, &beta,
                                    temp + cols, 1 ) );

   cublasCheckError( cublasDdot( cublasHandle, rows, temp + cols, 1, temp, 1, output ) );

}
