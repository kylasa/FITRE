
#ifndef __KFAC_INVERSES_H__
#define __KFAC_INVERSES_H__

/*
	Used to compute the inverses of the 
	Z 
	and 
	delta
	Matrices
*/

#include <core/datadefs.h>
#include <core/structdefs.h>
#include <solvers/kfac_structs.h>
#include <solvers/kfac_thread.h>
#include <nn/nn_decl.h>

#include <cusolverDn.h>
#include <cusolver_common.h>

void printNorms( CNN_MODEL *model, KFAC_CURVATURE_INFO *kfac_info );
void initKFACData( CNN_MODEL *model, KFAC_CURVATURE_INFO *kfac_info );


void updateRunningStats( real *src, real *update, int numElements, real momentum );

void computeOmegaZ( KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model, DEVICE_DATASET *data, 
  int samples, real *dampledInput, real *dampedZ, real *omegaZInv, int *zOffsets, int* zztOffsets, 
	real *devPtr, real *pageLckPtr, cublasHandle_t blasHandle, cudaStream_t curStream, cusolverDnHandle_t curHandle );

void computeLambdaDelta( KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model, 
   int samples, real *dx, real *dampedLambda, real *lambdaGInv, int *zOffsets, int *zztOffsets, 
	real *devPtr, real *pageLckPtr, cublasHandle_t blasHandle, cudaStream_t curStream, cusolverDnHandle_t curHandle ) ;

void computeSpeedUpLambdaDelta( KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model,
   int samples, real *dx, real *dampedLambda, int *zOffsets, int *zztOffsets, int iterCount );

void computeSpeedUpOmegaZ( KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model, DEVICE_DATASET *data, 
   int samples, real *z, int *zOffsets, 
   real *dampedInput, real *dampedZ, int *zztOffsets, int iterCount);


void computeKFACInverses( KFAC_CURVATURE_INFO *kfac_info, 
                              KFAC_THREAD_INFO *kfacThreadInfo, 
                              CNN_MODEL *model, 
                           int batchNo, real *devPtr, real *pageLckPtr, int iter, int miniBatchNo ) ;


#endif
