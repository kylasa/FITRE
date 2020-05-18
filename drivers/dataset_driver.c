
#include <core/datadefs.h>
#include <core/errors.h>
#include <core/memsizes.h>
#include <core/sparsedefs.h>
#include <core/structdefs.h>

#include <nn/nn_decl.h>

#include <utilities/dataset.h>

#include <drivers/dataset_driver.h>

#include <stdio.h>
#include <stdlib.h>

int getGaussDataset(CNN_MODEL *curvesModel, 
			HOST_DATASET *host, DEVICE_DATASET *device)
{
	fprintf( stderr, "Reading the curves dataset here. ... \n"); 
	fprintf( stderr, "\n\n"); 

	CONV_LAYER c = curvesModel->convLayer[ 0 ]; 

	readDataset( 
			"/scratch/skylasa/solvers/code/RC-NON-CONVEX-CNN-MULTI/matlab/gauss-dataset.txt", 
			"/scratch/skylasa/solvers/code/RC-NON-CONVEX-CNN-MULTI/matlab/gauss-labels.txt", 
			NULL, NULL, host, curvesModel->batchSize, c.inChannels, c.height * c.width ); 

	host->numClasses = 10; 
	initialize_device_data ( host, device); 

	fprintf( stderr, "Done reading the dataset .... \n"); 

	return 0;
}


int testDatasetRead (NN_MODEL *curvesModel, 
			HOST_DATASET *host, DEVICE_DATASET *device)
{
	fprintf( stderr, "Reading the curves dataset here. ... \n"); 
	fprintf( stderr, "\n\n"); 

	readDataset( "/scratch/skylasa/solvers/raw-data/curves/X_TRAIN.txt", 
					"/scratch/skylasa/solvers/raw-data/curves/Y_TRAIN.txt", 
					"/scratch/skylasa/solvers/raw-data/curves/X_TEST.txt", 
					"/scratch/skylasa/solvers/raw-data/curves/Y_TEST.txt", 
					host, 0, 0, 0); 

	initialize_device_data ( host, device); 

	fprintf( stderr, "Done reading the dataset .... \n"); 

	return 0;
}

int getCIFAR10 (CNN_MODEL *model, HOST_DATASET *host, 
	DEVICE_DATASET *device, SCRATCH_AREA *scratch )
{
	fprintf( stderr, "Reading CIFAR_10 dataset... \n"); 

	host->height = 32; 
	host->width = 32; 
	host->datasetType = device->datasetType = CIFAR10; 
	readCIFARDataset( "/scratch/gilbreth/skylasa/datasets/cifar/cifar-10-batches-bin/", 
					"data_batch_", 5, "test_batch.bin",
					host, scratch, 255., model->batchSize, 1, host->width, host->height); //normalize by 255

	initialize_device_data( host, device ); 

	fprintf( stderr, "Done reading the dataset .... \n"); 

	return 0;
}

int getCIFAR100 (CNN_MODEL *model, HOST_DATASET *host, 
	DEVICE_DATASET *device, SCRATCH_AREA *scratch )
{
	fprintf( stderr, "Reading CIFAR_100 dataset... \n"); 

	host->height = 32; 
	host->width = 32; 
	host->datasetType = device->datasetType = CIFAR100; 
	readCIFARDataset( "/scratch/gilbreth/skylasa/datasets/cifar/cifar-100-binary/", 
					"train_", 1, "test.bin",
					host, scratch, 255., model->batchSize, 2, host->height, host->width); //normalize by 255

	initialize_device_data( host, device ); 

	fprintf( stderr, "Done reading the dataset .... \n"); 

	return 0;
}

int getTinyImageNet (CNN_MODEL *model, HOST_DATASET *host, 
	DEVICE_DATASET *device, SCRATCH_AREA *scratch )
{
	fprintf( stderr, "Reading Tiny ImageNet dataset... \n"); 

	host->height = 64; 
	host->width = 64; 
	host->datasetType = device->datasetType = IMAGENET; 
	readCIFARDataset( "/scratch/gilbreth/skylasa/datasets/tiny-imagenet-200/files/", 
					"train_", 19, "test_",
					host, scratch, 255., model->batchSize, 3, host->height, host->width ); //normalize by 255

	initialize_device_image ( host, device ); 

	fprintf( stderr, "Done reading the dataset .... \n"); 

	return 0;
}
