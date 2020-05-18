
#ifndef _H_HANDLES__
#define _H_HANDLES__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cusolver_common.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <curand.h>
#include <time.h>


extern cublasHandle_t cublasHandle; 
extern cusparseHandle_t cusparseHandle;

extern cusolverDnHandle_t cusolverHandle;

extern curandGenerator_t curandGeneratorHandle;

#endif
