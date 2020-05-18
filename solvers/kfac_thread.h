
#ifndef __KFAC_THREAD_H__
#define __KFAC_THREAD_H__

#include <pthread.h>
#include <semaphore.h>

#include <solvers/kfac_structs.h>
#include <nn/nn_decl.h>

#include <core/datadefs.h>

#include <cusolverDn.h>
#include <cusolver_common.h>

typedef struct kfac_thread_info { 

	pthread_mutex_t	initiateMutex; 
	pthread_cond_t		initiateCVariable; 

	//pthread_mutex_t	resMutex; 
	//pthread_cond_t		resCVariable; 
	sem_t				resSemaphore; 
	sem_t				readySemaphore; 

	pthread_t		kfac_thread; 

	//Storage.. 
	int				slaveDevice; 
	int				masterDevice; 

	real* 			slaveWorkspace; 
	real* 			slavePageLckPtr; 

	int				workComplete; 
	int 				signalled; 
	int				batchNo; 

	int				curBatchSize; 

	cudaStream_t	slaveStream; 

	cublasHandle_t		threadBlasHandle; 
	cusolverDnHandle_t threadHandle;

} KFAC_THREAD_INFO; 

typedef struct kfac_thread_args { 

	KFAC_THREAD_INFO 		*kfacThreadInfo; 
	KFAC_CURVATURE_INFO 	*kfacInfo; 
	CNN_MODEL				*model; 

} KFAC_THREAD_ARGS; 


void prepSlaveDevice( KFAC_THREAD_INFO *kfacThreadInfo, KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model );

void copyResultToMaster( KFAC_THREAD_INFO *kfacThreadInfo, KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model );

void createKFACThread( KFAC_THREAD_INFO *kfacThreadInfo, KFAC_THREAD_ARGS *kfacThreadArgs );

void initKFACThread( KFAC_THREAD_INFO *kfacThreadInfo );




#endif
