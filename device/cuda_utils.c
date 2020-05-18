
#include <device/cuda_utils.h>

void cuda_malloc (void **ptr, size_t size, int memset, int err_code) {

    cudaError_t retVal = cudaSuccess;
    retVal = cudaMalloc (ptr, size);
    if (retVal != cudaSuccess) {
		fprintf (stderr, "Failed to allocate memory on device for the res: %d...  exiting with code: %d size: %ld, %s \n", 
							err_code, retVal, size, cudaGetErrorString(retVal));
		exit (err_code);
    }  

    if (memset) {
        retVal = cudaMemset (*ptr, 0, size);
        if (retVal != cudaSuccess) {
			fprintf (stderr, "Failed to memset memory on device... exiting with code %d, %s\n", 
							err_code, cudaGetErrorString( retVal ));
			exit (err_code);
        }
    }  
}

void cuda_malloc_host (void **ptr, size_t size, int memset, int err_code) {

    cudaError_t retVal = cudaSuccess;
    retVal = cudaMallocHost (ptr, size);
    if (retVal != cudaSuccess) {
		fprintf (stderr, "Failed to allocate memory on device for the res: %d...  exiting with code: %d size: %ld, %s \n", 
							err_code, retVal, size, cudaGetErrorString(retVal) );
		exit (err_code);
    }  

    if (memset) {
        retVal = cudaMemset (*ptr, 0, size);
        if (retVal != cudaSuccess) {
			fprintf (stderr, "Failed to memset memory on device... exiting with code %d, %s\n", 
							err_code, cudaGetErrorString( retVal ));
			exit (err_code);
        }
    }  
}



void cuda_free (void *ptr, int err_code) {

    cudaError_t retVal = cudaSuccess;
    if (!ptr) return;

    retVal = cudaFree (ptr);

    if (retVal != cudaSuccess) {
		fprintf (stderr, "Failed to release memory on device for res %d... exiting with code %d -- Address %ld, %s\n", 
						err_code, retVal, (long int)ptr, cudaGetErrorString( retVal ));
        return;
    }  
}

void cuda_free_host (void *ptr, int err_code) {

    cudaError_t retVal = cudaSuccess;
    if (!ptr) return;

    retVal = cudaFreeHost (ptr);

    if (retVal != cudaSuccess) {
		fprintf (stderr, "Failed to release memory on device for res %d... exiting with code %d -- Address %ld, %s\n", 
						err_code, retVal, (long int)ptr, cudaGetErrorString( retVal ));
        return;
    }  
}


void cuda_memset (void *ptr, int data, size_t count, int err_code){
    cudaError_t retVal = cudaSuccess;

    retVal = cudaMemset (ptr, data, count);
    if (retVal != cudaSuccess) {
	 	fprintf (stderr, "ptr passed is %ld, value: %ld \n", (long int)ptr, &ptr);
	 	fprintf (stderr, " size to memset: %ld \n", count);
		fprintf (stderr, " target data is : %d \n", data);
		fprintf (stderr, "Failed to memset memory on device... exiting with code %d, cuda code %d, %s\n", 
							err_code, retVal, cudaGetErrorString( retVal ));
		exit (err_code);
    }
}

void copy_host_device (void *host, void *dev, size_t size, enum cudaMemcpyKind dir, int resid)
{
	cudaError_t	retVal = cudaErrorNotReady;

	if (dir == cudaMemcpyHostToDevice)
		retVal = cudaMemcpy (dev, host, size, cudaMemcpyHostToDevice);
	else
		retVal = cudaMemcpy (host, dev, size, cudaMemcpyDeviceToHost);

	if (retVal != cudaSuccess) {
		fprintf (stderr, "could not copy resource %d from host to device: reason %d:%s \n",
							resid, retVal, cudaGetErrorString( retVal ));
		exit (resid);
	}
}

void copy_device (void *dest, void *src, size_t size, int resid)
{
	cudaError_t	retVal = cudaErrorNotReady;

	retVal = cudaMemcpy (dest, src, size, cudaMemcpyDeviceToDevice);
	if (retVal != cudaSuccess) {
		fprintf (stderr, "could not copy resource %d from device to device: reason %d \n",
							resid, retVal);
		exit (resid);
	}
}

void print_device_mem_usage ()
{
   size_t total, free;
   cudaMemGetInfo (&free, &total);
   if (cudaGetLastError () != cudaSuccess )
   {
      fprintf (stderr, "Error on the memory call \n");
		return;
   }

   fprintf (stderr, "Total %ld Mb %ld gig %ld , free %ld, Mb %ld , gig %ld \n", 
                     total, total/(1024*1024), total/ (1024*1024*1024), 
                     free, free/(1024*1024), free/ (1024*1024*1024) );
}

void deviceMemAllocTest ()
{
	void *devPtr; 
	size_t size = 1;
	size_t MY_HEAP = size << (30 + 3); 

	fprintf( stderr, "Beginning device mem alloc test...... \n\n");

	do {
		cudaMalloc(&devPtr, size*sizeof(double));
		if(devPtr == NULL) {
			fprintf(stderr, "couldn't allocate %lu bytes.\n", size * sizeof(double));
			break;
		} else {
			fprintf(stderr, "\t\tAllocated %lu bytes.\n", size * sizeof(double));
		}
		cudaFree(devPtr);
		size*=2;

		if (size > MY_HEAP) break;
	} while(1);

	fprintf( stderr, "Done with Device Malloc Test..... \n");

	
	fprintf( stderr, "trying to allocate 8 Gigs.... \n");
	cudaMalloc( &devPtr, MY_HEAP );
	if (devPtr == NULL) {
		fprintf( stderr, "Failed to allocate 8 Gigs.... \n"); 
	} else {
		fprintf( stderr, "Success !!!! 8 Gigs \n"); 
		cudaFree( devPtr );
	}

	fprintf( stderr, "trying to allocate 8 Gigs....cuda_malloc \n");
	cuda_malloc( &devPtr, (size_t(1) << 33), 0, 0xFF); 
	if (devPtr == NULL) {
		fprintf( stderr, "Failed to allocate 8 Gigs.... \n"); 
	} else {
		fprintf( stderr, "Success !!!! 8 Gigs \n"); 
		cudaFree( devPtr ); 
	}
}

