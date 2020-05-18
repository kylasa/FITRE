
#include <augmentation/dev_normalize.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

#include <core/errors.h>

#include <utilities/print_utils.h>


GLOBAL void ker_normalize (real *input, 
	int samples, int height, int width, int channels, 
	real *means, real *std, real nrConstant ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int imgIdx = blockIdx.y; 

	int count = height * width * channels; 

	int chId, chIdx, row, col, loc; 

	if (idx < count){
			chId = idx / (height * width ); 
			chIdx = idx % (height * width );

			row = chIdx % height; 
			col = chIdx / height; 

			loc = chId * height * width * samples + 
					imgIdx * height * width + 
					col * height + row ;

			input [loc ] /= nrConstant;
			input [loc ] = (input[ loc ] - means[ chId ]) / std[ chId ]; 
	}
}

/*
   real means_cifar10[3] = { 0.4914, 0.4822, 0.4465 };  
   real std_cifar10[3] = { 0.247, 0.243, 0.261 };

	real means_imagenet[ 3 ] = {0.4914, 0.4822, 0.4465}; 
	real std_imagenet[ 3 ] = {0.2023, 0.1994, 0.2010};

110 CIFAR10 Statistics:  (tensor([0.4914, 0.4822, 0.4465]), tensor([0.2470, 0.2435, 0.2616]))
111 CIFAR100 Statistics:  (tensor([0.5071, 0.4865, 0.4409]), tensor([0.2673, 0.2564, 0.2762]))
112 ImageNet Statistics:  (tensor([0.4799, 0.4478, 0.3972]), tensor([0.2769, 0.2690, 0.2820]))
*/
	
void normalizeCIFAR10( real *input, int samples, int channels, int height, int width, 
	real *devPtr, real *hostPtr, DATASET_TYPE datasetType )
{
   int blocks = ( channels * height * width + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 
   dim3 blockXYZ( blocks, samples, 1);  

   real mean_cifar10[3] = {0.4914, 0.4822, 0.4465}; 
   real std_cifar10[3] = {0.2470, 0.2435, 0.2616 };

   real mean_cifar100[3] = {0.5071, 0.4865, 0.4409}; 
   real std_cifar100[3] = {0.2673, 0.2564, 0.2762};

   real mean_imagenet[3] = {0.4799, 0.4478, 0.3972};
   real std_imagenet[3] = {0.2769, 0.2690, 0.2820};

   real *mean, *std;

   switch( datasetType ) { 
      case CIFAR10:  
         mean = mean_cifar10;
         std = std_cifar10; 
         break;
      case CIFAR100:  
         mean = mean_cifar100;
         std = std_cifar100; 
         break;
      case IMAGENET:  
         mean = mean_imagenet;
         std = std_imagenet; 
         break;

      default: 
         fprintf( stderr, "Unknown DatasetType here... \n\n"); 
         exit( -1 );  
   } 


	copy_host_device( mean, devPtr, sizeof(real) * 3, 
		cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	copy_host_device( std, devPtr + 3, sizeof(real) * 3, 
		cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

   ker_normalize <<< blockXYZ, BLOCK_SIZE>>> 
      ( input, samples, height, width, channels, devPtr, devPtr + 3, 255. );  
   cudaDeviceSynchronize (); 
   cudaCheckError (); 

}
