
#include <utilities/dataset_utils.h>
#include <utilities/utils.h>
#include <utilities/print_utils.h>

#include <device/cuda_utils.h>

#include <core/errors.h>

void selectHostMatrix( HOST_DATASET *host, real *trainX, real *trainY, int numSamples, int *indices, int offset, real *hostPtr )
{
	for (int i = 0; i < numSamples; i ++) { 	

		real *device = trainX + i * host->features; 
		real *src = host->trainSetX + 
						indices[ offset + i ] * host->features; 

		copy_host_device( src, device, sizeof(real) * host->features, 
							cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );  

		hostPtr[ i ] = host->trainSetY [ indices[ offset + i ] ]; 
	} 

	copy_host_device( hostPtr, trainY, sizeof(real) * numSamples, cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
}

