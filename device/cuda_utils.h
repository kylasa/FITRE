#ifndef	__CUDA_UTILS_H_
#define __CUDA_UTILS_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusolverDn.h>
#include <cusolver_common.h>
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>


void cuda_malloc (void **, size_t, int , int);
void cuda_malloc_host( void **, size_t, int, int );

void cuda_free (void *, int);
void cuda_free_host (void *, int);
void cuda_memset (void *, int , size_t , int );

void copy_host_device (void *, void *, size_t , enum cudaMemcpyKind, int);
void copy_device (void *, void *, size_t, int );

void print_device_mem_usage ();
void deviceMemAllocTest ();

#define cusolverCheckError( cusolverStatus ) __cusolverCheckError  (cusolverStatus, __FILE__, __LINE__)
inline void __cusolverCheckError( cusolverStatus_t cusolverStatus, const char *file, const int line )
{
	if (cusolverStatus != CUSOLVER_STATUS_SUCCESS){
		fprintf( stderr, "failed ..%s:%d -- error code %d \n", file, line, cusolverStatus );
		exit (-1);
	}
}

#define cusparseCheckError(cusparseStatus) __cusparseCheckError (cusparseStatus, __FILE__, __LINE__)
inline void __cusparseCheckError( cusparseStatus_t cusparseStatus, const char *file, const int line )
{
if (cusparseStatus!= CUSPARSE_STATUS_SUCCESS)
{
	//fprintf (stderr, "failed .. %s:%d -- error code %d \n", __FILE__, __LINE__, cusparseStatus);
	fprintf (stderr, "failed .. %s:%d -- error code %d \n", file, line, cusparseStatus);
	exit (-1);
}
return;
}


#define cublasCheckError(cublasStatus) __cublasCheckError (cublasStatus, __FILE__, __LINE__)
inline void __cublasCheckError( cublasStatus_t cublasStatus, const char *file, const int line )
{
if (cublasStatus!= CUBLAS_STATUS_SUCCESS)
{
	fprintf (stderr, "failed .. %s:%d -- error code %d \n", file, line, cublasStatus);
	exit (-1);
}
return;
}

#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf (stderr, "Failed .. %s:%d -- gpu erro code %d:%s\n", file, line, err, cudaGetErrorString( err ) );
		exit( -1 );
	}
 
	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	/*
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err )
	{
		exit( -1 );
	}
	*/
	return;
}

#define curandCheckError(curandStatus) __curandCheckError (curandStatus, __FILE__, __LINE__)
inline void __curandCheckError( curandStatus_t curandStatus, const char *file, const int line )
{
        if (curandStatus!= CURAND_STATUS_SUCCESS)
        {
                fprintf (stderr, "failed .. %s:%d -- error code %d \n", __FILE__, __LINE__, curandStatus);
                exit (-1);
        }
        return;
}


#endif
