#ifndef _H_STRUCT_DEFS__
#define _H_STRUCT_DEFS__

#include <core/datadefs.h>
#include <core/sparsedefs.h>

typedef enum data_type {
      TEST_DATA = 0,  
      TRAIN_DATA = 1 
   } DATA_TYPE; 

typedef enum DATASET_SIZE {
   FULL_DATASET = 0,  
   SAMPLED_DATASET = 1 
   } DATASET_SIZE; 

typedef enum dataset_type { 
	CIFAR10 = 10, 
	CIFAR100 = 20, 
	IMAGENET = 30 
} DATASET_TYPE; 



typedef struct scratch_space{ 
	real *hostWorkspace; 
	real *devWorkspace;
	real *pageLckWorkspace;

	real *nextDevPtr; 
	real *nextHostPtr;
	real *nextPageLckPtr;
} SCRATCH_AREA;

typedef struct host_dataset{ 
	
	real *trainSetX; 
	real *trainSetY; 
	real *testSetX; 
	real *testSetY; 
	
	int trainSizeX; 
	int trainSizeY; 
	int testSizeX; 
	int testSizeY; 

	int features;
	int height; 
	int width; 
	int numClasses; 
	DATASET_TYPE datasetType; 

}HOST_DATASET; 

typedef struct dev_dataset{ 

	real *trainSetX; 
	real *trainSetY; 
	real *testSetX; 
	real *testSetY; 

	real *weights; 

	int numClasses; 
	DATASET_TYPE datasetType; 

	int features;
	int height; 
	int width;
	int testSizeX; 
	int testSizeY; 
	int trainSizeX;
	int trainSizeY;
	
	//Sample related members. 
	int sampleSize; 
	real *sampledTrainX; 
	real *sampledTrainY;	
	real *currentBatch;

	SparseDataset spSamplingMatrix;

} DEVICE_DATASET;

#endif
