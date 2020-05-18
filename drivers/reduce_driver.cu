
#include <drivers/reduce_driver.h>

#include <device/cuda_utils.h>
#include <device/reduce.h>
#include <device/handles.h>

#include <functions/dev_initializations.h>

#include <utilities/utils.h>
#include <utilities/reduce_helper.h>
#include <utilities/reduce.h>

#include <core/datadefs.h>
#include <core/errors.h>


/*
void sum_cublas( real *input, real *output, int count, real *temp )
{
	int rows = 1000; 
	int cols = count / rows;  // 20000	

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
	
	//sum the rows here. 
	reduce <<< 1, BLOCK_SIZE, WARP_SIZE * sizeof(real) >>> 	
		( temp + cols, output, rows ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	cublasCheckError( cublasDdot( cublasHandle, rows, temp + cols, 1, temp, 1, output ) ); 
}

*/
void sum( real *input, real *output, int count, int threadsPerBlock, int gridBlocksX )
{
	reduce_grid <<< (gridBlocksX, 1), (threadsPerBlock, 1,1), WARP_SIZE * sizeof(real) >>> 
		(input, input + count, count ); 
	cudaThreadSynchronize ();
	cudaCheckError (); 

	reduce <<< 1, threadsPerBlock, WARP_SIZE * sizeof(real) >>> 
		(input + count, output, gridBlocksX ); 
	cudaThreadSynchronize ();
	cudaCheckError (); 
}

void reduce_nvidia( real *in, real *out, int count) {
	int blocks1 = count / (2 * 1024 );
	reduce6<<< blocks1, 1024, 1024  * sizeof(real)  >>> 
		(in, in + count, count ); 
	cudaThreadSynchronize ();
	cudaCheckError (); 
	
	int blocks2 = blocks1 / ( 2 * 1024 );
	reduce6<<< blocks2, 1024, 1024 * sizeof(real)>>> 
		( in + count,  in + count + blocks1, blocks1 ); 
	cudaThreadSynchronize ();
	cudaCheckError (); 

	reduce6 <<< 2, 4, 1024 * sizeof(real) >>> 
		(in + count + blocks1, out, blocks2); 
	cudaThreadSynchronize ();
	cudaCheckError (); 
}

void testReduce (SCRATCH_AREA *s)
{
	int count = 32 * 1024 * 1024; 

	real *hostPtr = s->hostWorkspace; 
	real *devPtr = s->devWorkspace; 
	real *result = devPtr + count; 
	real *page = s->pageLckWorkspace; 

	real start, total; 

	for (int i = 0; i < count; i ++ ) hostPtr[i] = 1; 

	copy_host_device( hostPtr, devPtr, sizeof(real) * count, cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
	cuda_memset ( result, 0, sizeof(real) * 1024, ERROR_MEMSET ); 
		
/*
	start = Get_Time (); 
	sum( devPtr, page, count, BLOCK_SIZE, 56); 
	total = Get_Timing_Info (start); 

	fprintf( stderr, "\n\n\n\n");
	fprintf( stderr, "SUM of 20 Millions ones is %10.2f, in %f (msecs) \n", *page, (total * 1000.)); 
*/

	start = Get_Time (); 
	reduce_cublas( devPtr, page, devPtr + count, 1024, 32 * 1024); 
	total = Get_Timing_Info (start); 

	fprintf( stderr, "SUM (CUBLAS) of 20 Millions ones is %10.2f, in %f (msecs) \n", *page, (total * 1000.)); 
	*page = 100; 

	start = Get_Time (); 
	reduce_nvidia( devPtr, page, count); 
	total = Get_Timing_Info (start); 

	fprintf( stderr, "SUM (NVIDIA) of 32 Millions ones is %10.2f, in %f (msecs) \n", *page, (total * 1000.)); 

	start = Get_Time (); 
	myreduce( devPtr, count, devPtr + count, page + 1 ); 
	total = Get_Timing_Info (start); 

	fprintf( stderr, "SUM (My IMPL) of 32 Millions ones is %10.2f, in %f (msecs) \n", *(page + 1), (total * 1000.)); 

}
