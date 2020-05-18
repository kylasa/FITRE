#include <device/cuda_environment.h>
#include <device/cuda_utils.h>
#include <device/handles.h>
#include <utilities/utils.h>
#include <core/datadefs.h>
#include <core/structdefs.h>
#include <core/memsizes.h>
#include <core/errors.h>

#include <stdlib.h>
#include <time.h>

void cuda_set_heap_limits()
{
	const size_t malloc_limit = DEVICE_WORKSPACE_SIZE + (100 * 1024 * 1024);
	
	//cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, malloc_limit);
	//cudaCheckError ();

	size_t malloced = 0;
	cuCtxGetLimit(&malloced, CU_LIMIT_MALLOC_HEAP_SIZE);
	cudaCheckError ();

fprintf(stderr, "Heap Malloc Limit set: %lu\n", malloc_limit);
fprintf(stderr, "Heap Current Limit :%lu\n", malloced);

}

void cuda_env_init(SCRATCH_AREA *scratch, int gpu){

   cudaSetDeviceFlags( cudaDeviceScheduleBlockingSync );
	cudaSetDevice (gpu);
	cudaDeviceReset ();
	cudaDeviceSynchronize ();

	cublasCheckError( cublasCreate( &cublasHandle ) );
	cusparseCheckError( cusparseCreate( &cusparseHandle ) );
	curandCheckError( curandCreateGenerator( &curandGeneratorHandle, CURAND_RNG_PSEUDO_DEFAULT ) );
   curandCheckError ( curandSetPseudoRandomGeneratorSeed ( curandGeneratorHandle , time(NULL) )) ;
	cusolverCheckError( cusolverDnCreate( &cusolverHandle ) ); 

	srand( time(NULL) ); 

	allocate_memory( (void **)&scratch->hostWorkspace, (size_t)HOST_WORKSPACE_SIZE );
	scratch->nextHostPtr = scratch->hostWorkspace;
}

void cuda_allocate_workspace (SCRATCH_AREA *scratch, int device) 
{
	size_t memFree, memTotal, memAlloc; 

   //cudaDeviceProp deviceProp;
   //cudaGetDeviceProperties(&deviceProp, device);
	cudaMemGetInfo( &memFree, &memTotal ); 

	memAlloc = memFree - 500 * 1024 * 1024; 

	fprintf( stderr, "Master Device Workspace size: %ld (MB), %ld (GB)\n\n", 
									memAlloc / (1024 * 1024), memAlloc / (1024 * 1024 * 1024) ); 

	cuda_malloc( (void **)&scratch->devWorkspace, memAlloc, 0, ERR_MEM_ALLOC  );
	cuda_malloc_host ((void **)&scratch->pageLckWorkspace, PAGE_LOCKED_WORKSPACE_SIZE, 0, ERR_MEM_ALLOC );

	scratch->nextDevPtr = scratch->devWorkspace;
	scratch->nextPageLckPtr = scratch->pageLckWorkspace;
}

void cuda_env_cleanup (SCRATCH_AREA *scratch){
	release_memory( (void **)&scratch->hostWorkspace );
	cuda_free ((void *)scratch->devWorkspace, ERR_MEM_FREE);
	cuda_free_host ( (void *)scratch->pageLckWorkspace, ERR_MEM_FREE );

	//release_memory( (void **)&dscratch);


	curandCheckError ( curandDestroyGenerator ( curandGeneratorHandle ) );
}
