
#include <solvers/kfac_thread.h>
#include <solvers/kfac_inverses.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <device/handles.h>
#include <device/cuda_utils.h>

#include <core/memsizes.h>
#include <core/errors.h>


void initKFACThread( KFAC_THREAD_INFO *kfacThreadInfo )
{
	kfacThreadInfo->kfac_thread = NULL; 

	kfacThreadInfo->initiateMutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER; 
	kfacThreadInfo->initiateCVariable = (pthread_cond_t)PTHREAD_COND_INITIALIZER; 

	if (sem_init( &kfacThreadInfo->resSemaphore, 0, 0) != 0) { 
		fprintf( stderr, "We have a problem with the semaphore initialization... \n"); 
		exit( -1 ); 
	}

	if (sem_init( &kfacThreadInfo->readySemaphore, 0, 0) != 0) { 
		fprintf( stderr, "We have a problem with the semaphore initialization... \n"); 
		exit( -1 ); 
	}


#ifdef DEBUG_KFAC_THREAD
	fprintf( stderr, "KFAC-THREAD done with initialization... \n"); 
#endif
}

void initiateSlaveDevice( KFAC_THREAD_INFO *kfacThreadInfo, KFAC_CURVATURE_INFO *kfacInfo, CNN_MODEL *model )
{
   size_t memFree, memTotal, memAlloc; 
	size_t slaveDevMem = sizeof(real) * ( 2 * model->zztSize + 
									kfacInfo->OmegaZOffsets[ model->cLayers + model->lLayers ] + 
									kfacInfo->LambdaGOffsets[ model->cLayers + model->lLayers ] + 
									model->zSize); 

	//Allocate memory for the ZZT and ZInverse Matrices, same for delta's as well. 
	cudaSetDeviceFlags( cudaDeviceBlockingSync ); 
	cudaCheckError (); 

	cudaSetDevice( kfacThreadInfo->slaveDevice ); 
	cudaDeviceReset( ); 
	cudaDeviceSynchronize (); 

   cudaMemGetInfo( &memFree, &memTotal );  
   memAlloc = memFree - 500 * 1024 * 1024; 

   fprintf( stderr, "Slave Device Workspace size: %ld (MB), %ld (GB)\n\n", 
                           memAlloc / (1024 * 1024), memAlloc / (1024 * 1024 * 1024) );  


	cublasCheckError( cublasCreate( &kfacThreadInfo->threadBlasHandle ) ); 

	//cudaStreamCreateWithFlags( & kfacThreadInfo->slaveStream, cudaStreamNonBlocking ); 
	cudaStreamCreate( & kfacThreadInfo->slaveStream ); 
   cusolverCheckError( cusolverDnCreate( &kfacThreadInfo->threadHandle) );
	cusolverDnSetStream( kfacThreadInfo->threadHandle, kfacThreadInfo->slaveStream ); 


	cuda_malloc( (void **)&kfacThreadInfo->slaveWorkspace, memAlloc, 0, ERR_MEM_ALLOC  );  
   cuda_malloc_host ((void **)&kfacThreadInfo->slavePageLckPtr, PAGE_LOCKED_WORKSPACE_SIZE, 0, ERR_MEM_ALLOC );

#ifdef DEBUG_KFAC_THREAD
	fprintf( stderr, "KFAC-THREAD done with Slave Device Initialization... %d\n", kfacThreadInfo->slaveDevice); 
#endif
}

void prepSlaveDevice( KFAC_THREAD_INFO *kfacThreadInfo, KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model )
{
   int *lambdaOffsets = kfac_info->LambdaGOffsets; 
   int *omegaOffsets = kfac_info->OmegaZOffsets; 

	real *masterDampedInput	= kfac_info->dampedInput; 
	real *masterOmegaZ 		= kfac_info->dampedZ; 
	real *masterLambda 		= kfac_info->dampedLambda; 

	real *slaveDampedInput 	= kfacThreadInfo->slaveWorkspace; 
	real *slaveOmegaZ 		= slaveDampedInput + kfac_info->dampedInputSize; 
	real *slaveLambda 		= slaveOmegaZ + model->zztSize; 
	real *slaveZInv 			= slaveLambda + model->zztSize; 
	real *slaveGInv 			= slaveZInv + omegaOffsets [ model->cLayers + model->lLayers ]; 
	real *slaveDevPtr 		= slaveGInv + lambdaOffsets[ model->cLayers + model->lLayers ]; 

	// Damped Input Size
	cudaMemcpyPeer( slaveDampedInput, kfacThreadInfo->slaveDevice, 
							masterDampedInput, kfacThreadInfo->masterDevice, 
							sizeof(real) * kfac_info->dampedInputSize ); 

	//copy ZZT to the slave device here. 	
	cudaMemcpyPeer( slaveOmegaZ, kfacThreadInfo->slaveDevice, 
							masterOmegaZ, kfacThreadInfo->masterDevice, 
							sizeof(real) * model->zztSize ); 

	//copy GGT to the slave Device here. 
	cudaMemcpyPeer( slaveLambda, kfacThreadInfo->slaveDevice, 
							masterLambda, kfacThreadInfo->masterDevice, 
							sizeof(real) * model->zztSize ); 

#ifdef DEBUG_KFAC_THREAD
	fprintf( stderr, "KFAC-THREAD copyied data to the slave device ... \n"); 
#endif
}

void copyResultToMaster( KFAC_THREAD_INFO *kfacThreadInfo, KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model )
{
   int *lambdaOffsets 		= kfac_info->LambdaGOffsets; 
   int *omegaOffsets 		= kfac_info->OmegaZOffsets; 

	real *masterZInv			= kfac_info->OmegaZInv; 
	real *masterGInv 			= kfac_info->LambdaGInv; 

	real *slaveDampedInput 	= kfacThreadInfo->slaveWorkspace; 
	real *slaveOmegaZ 		= slaveDampedInput + kfac_info->dampedInputSize; 
	real *slaveLambda 		= slaveOmegaZ + model->zztSize; 
	real *slaveZInv 			= slaveLambda + model->zztSize; 
	real *slaveGInv 			= slaveZInv + omegaOffsets [ model->cLayers + model->lLayers ]; 
	real *slaveDevPtr 		= slaveGInv + lambdaOffsets[ model->cLayers + model->lLayers ]; 

	//copy ZZT to the slave device here. 	
	cudaMemcpyPeer( masterZInv, kfacThreadInfo->masterDevice, 
							slaveZInv, kfacThreadInfo->slaveDevice, 
							sizeof(real) * omegaOffsets[ model->cLayers + model->lLayers ] ); 

	//copy GGT to the slave Device here. 
	cudaMemcpyPeer( masterGInv, kfacThreadInfo->masterDevice, 
							slaveGInv, kfacThreadInfo->slaveDevice, 
							sizeof(real) * lambdaOffsets[ model->cLayers + model->lLayers ] ); 

#ifdef DEBUG_KFAC_THREAD
	fprintf( stderr, "KFAC-THREAD copied data back to the master... \n"); 
#endif
}

void* kfacThreadFunc( void *args )
{

	KFAC_THREAD_ARGS *threadArgs = (KFAC_THREAD_ARGS *)args; 

	KFAC_THREAD_INFO *kfacThreadInfo 	= threadArgs->kfacThreadInfo; 
	KFAC_CURVATURE_INFO *kfacInfo 		= threadArgs->kfacInfo ;
	CNN_MODEL *model	 						= threadArgs->model;  

	real *slaveOmegaZ, *slaveLambda, *slaveZInv, *slaveGInv, *slaveDevPtr, *slaveDampedInput; 
	int *omegaOffsets = kfacInfo->OmegaZOffsets; 
	int *lambdaOffsets = kfacInfo->LambdaGOffsets; 

	initiateSlaveDevice( kfacThreadInfo, kfacInfo, model ); 

	sem_post( &kfacThreadInfo->readySemaphore ); 

	while (1) {
		//WAIT FOR THE SIGNAL TO PROCEED
		pthread_mutex_lock( &kfacThreadInfo->initiateMutex ); 

			pthread_cond_wait( &kfacThreadInfo->initiateCVariable, &kfacThreadInfo->initiateMutex ); 
			
		pthread_mutex_unlock( &kfacThreadInfo->initiateMutex ); 

		// WE ARE ALLOWED TO COMPUTE THE INVERSES. 
		// SLAVE WORK UNIT HERE

#ifdef DEBUG_KFAC_THREAD
		fprintf( stderr, "KFAC-THREAD Starting with Inverse comptuation... \n\n"); 
#endif

		slaveDampedInput 	= kfacThreadInfo->slaveWorkspace; 
		slaveOmegaZ 		= slaveDampedInput + kfacInfo->dampedInputSize; 
		slaveLambda 		= slaveOmegaZ + model->zztSize; 
		slaveZInv 			= slaveLambda + model->zztSize;
		slaveGInv 			= slaveZInv + omegaOffsets [ model->cLayers + model->lLayers ]; 
		slaveDevPtr			= slaveGInv + lambdaOffsets[ model->cLayers + model->lLayers ];

		computeOmegaZ( kfacInfo, model, NULL, kfacThreadInfo->curBatchSize, 
							slaveDampedInput, slaveOmegaZ, slaveZInv, 
							model->zOffsets, model->zztOffsets, 
							slaveDevPtr, kfacThreadInfo->slavePageLckPtr, 
							kfacThreadInfo->threadBlasHandle, kfacThreadInfo->slaveStream, kfacThreadInfo->threadHandle ); 							
								
		computeLambdaDelta( kfacInfo, model, kfacThreadInfo->curBatchSize, NULL, slaveLambda, slaveGInv, 
									model->zOffsets, model->zztOffsets, slaveDevPtr, kfacThreadInfo->slavePageLckPtr, 
									kfacThreadInfo->threadBlasHandle, kfacThreadInfo->slaveStream, kfacThreadInfo->threadHandle ); 							
	
		// Make sure that the cudaDeviceBlockingSync is set on this device. 
		//cudaStreamSynchronize( kfacThreadInfo->slaveStream ); 
		cudaDeviceSynchronize ();

		// SLAVE WORK UNIT HERE
		//fprintf( stderr, "KFAC-THREAD Done with Inverses... signalling the main thread.. \n\n"); 

		//Signal the master here. 
		kfacThreadInfo->workComplete = 1;
		sem_post( &kfacThreadInfo->resSemaphore ); 
	}
}


void createKFACThread( KFAC_THREAD_INFO *kfacThreadInfo, KFAC_THREAD_ARGS *kfacThreadArgs )
{
	pthread_create( &kfacThreadInfo->kfac_thread, NULL, kfacThreadFunc, (void *) kfacThreadArgs ); 

#ifdef DEBUG_KFAC_THREAD
	fprintf( stderr, "KFAC-THREAD done with creating of the thread... \n"); 
#endif
}
