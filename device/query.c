
#include <device/query.h>
#include <device/cuda_utils.h> 

#include <device/dev_params.h>

void getDeviceParameters( ) 
{
	cudaDeviceProp deviceProp;
	int warpSize; 
	int maxThreadsPerSM;
	int maxThreadsPerBlock;
	int smCount;
	int dev = 0; 

	cudaSetDevice(dev);
	cudaGetDeviceProperties(&deviceProp, dev);
	cudaCheckError ();

	warpSize = deviceProp.warpSize;
	maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
	smCount = deviceProp.multiProcessorCount;
	maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

	fprintf( stderr, "warpSize: %d, maxThreadsPerSM: %d, maxThreadsPerBlock: %d\n", 
				warpSize, maxThreadsPerSM, maxThreadsPerBlock ); 

	fprintf( stderr, "SM Count: %d \n", smCount ); 
	fprintf( stderr, "Max Threads per SM: %d \n", maxThreadsPerSM ); 
	fprintf( stderr, "Max Grid Size: %d, %d \n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1] ); 
	fprintf( stderr, "Max ThreadsDim: %d, %d, %d \n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2] ); 
	

	
	//update the globals here. 
	BLOCK_SIZE = maxThreadsPerBlock;
	//BLOCK_SIZE = 256; 
	WARP_SIZE = warpSize;
	DEVICE_NUM_BLOCKS = (smCount * maxThreadsPerSM * 6) / maxThreadsPerBlock; 

	fprintf( stderr, "Ideal Block Count: %d \n", DEVICE_NUM_BLOCKS ); 
	fprintf( stderr, "BLOCK_SIZE: %d \n", BLOCK_SIZE ); 

	fprintf( stderr, " ****************** Platform Report ************* \n"); 
	fprintf( stderr, " Size( double ): %zu  \n", sizeof(double) ); 
	fprintf( stderr, " Size( float ): %zu  \n", sizeof(float) ); 
	fprintf( stderr, " Size( int ): %zu  \n", sizeof(int) ); 
	fprintf( stderr, " ******************* **************************** \n"); 
}
