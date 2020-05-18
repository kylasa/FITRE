
#include <utilities/alloc_sampled_dataset.h> 

#include <core/structdefs.h>
#include <device/cuda_utils.h>
#include <core/errors.h>
#include <core/datadefs.h>

void allocSampledDataset( DEVICE_DATASET *data, int sampleSize )
{
	data->sampleSize = sampleSize; 

 cuda_malloc( (void **) &data->sampledTrainX, 
	sampleSize * data->features * sizeof(real), 1, ERROR_MEM_ALLOC );  	

 cuda_malloc( (void **) &data->sampledTrainY, 
	sampleSize * data->features * sizeof(real), 1, ERROR_MEM_ALLOC );  	

}
