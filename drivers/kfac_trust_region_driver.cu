
#include <drivers/kfac_trust_region_driver.h>

#include <solvers/kfac_trust_region.h>
#include <solvers/params.h>

#include <core/errors.h>

#include <device/cuda_utils.h>
#include <device/gen_random.h>
#include <device/device_defines.h>
#include <device/handles.h>
#include <functions/dev_initializations.h>

#include <utilities/print_utils.h>

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>


void readKFACVecFromFileTR( real *dev, real *host ) { 

   int rows = readVector( host, INT_MAX, "./weights.txt", 0, NULL);
   copy_host_device( host, dev, rows * sizeof(real), cudaMemcpyHostToDevice, 
      ERROR_MEMCPY_HOST_DEVICE );  
   
   fprintf( stderr, "Finished reading Vec (%d) from file \n", rows );  

   for (int i = 0; i < 10; i ++) fprintf( stderr, "%6.10f \n", host[i] );  
}

void testRandomNumbers (SCRATCH_AREA *scratch) 
{
	int *idx = (int *)scratch->nextHostPtr; 

	int m, n; 
	m = 10; 
	n = 10; 

	genRandomVector( idx, m, n ); 	

	real tmp = 0; 
	for (int i = 0; i < n; i ++) 
		tmp += (real) idx[ i ]; 

	fprintf( stderr, "\n\n"); 
	
	for (int i = 0; i < n; i ++) 
		fprintf( stderr, " %d ", idx[ i ] ); 

	fprintf( stderr, "\n\n"); 


	fprintf( stderr, " Sum of numbers: %f, %f \n", tmp, (real)(n * (n + 1)) / 2. ); 
}

void initKFACTrustRegionParams( TRUST_REGION_PARAMS *params, int n )
{
	params->eta1 = 0.8;
	params->eta2 = 1e-4;
	params->gamma1 = 2; 
	params->gamma2 = 1.2; 
	params->maxProps =  ULONG_MAX; 
	params->maxMatVecs = 1e15; 
	params->maxEpochs = 1; 


	//sampled_tr_cg.m file. 
	params->delta = 1.; 
	params->maxDelta = 100; 
	params->minDelta = 1e-6; 
	params->maxIters = 200; 

	//defaults from curves_autoencoder.m
	params->alpha = 0.01; 				// SGD Momentum

	//no regularization 
	params->lambda = 0; 

	//loop variants here. 
	params->curIteration = 0; 
}

void initKFACInfo ( KFAC_CURVATURE_INFO *kfacInfo )
{
	kfacInfo->OmegaZZT = NULL; 
	kfacInfo->LambdaGGT = NULL; 

	kfacInfo->OmegaZInv = NULL; 
	kfacInfo->LambdaGInv = NULL; 

	kfacInfo->vec = NULL; 
	kfacInfo->nGradient = NULL; 

	for (int i = 0; i < MAX_LAYERS; i ++) {
		kfacInfo->OmegaZOffsets[ i ] = kfacInfo->LambdaGOffsets[ i ] = 0; 
	}

	kfacInfo->stats_decay = 0.8; 
	kfacInfo->momentum = 0; 

	kfacInfo->regLambda = 1e-3; 
	kfacInfo->dampGamma = 100; 

	kfacInfo->checkGrad = 1; 
}

void matrixInverse (real *matrix, real *inverse, int rows, real *devPtr, int *pageLckPtr )
{
   int n = rows; 

   real *ZPtr = matrix;
   real *ZInvPtr = inverse; 

   int *infoArray = (int *)pageLckPtr; 
   int *pivotArray = (int *)(devPtr + n*n); 
   int *nextDevPtr = pivotArray + rows; 

   real **ZArr = (real **) nextDevPtr;
   real **ZInvArr = ZArr + 1;  

   copy_device( devPtr, matrix, sizeof(real) * n * n, ERROR_MEMCPY_DEVICE_DEVICE );  

   //In place LU factorization here. 
   copy_host_device( &ZPtr, ZArr, sizeof(real *), cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );  
   cublasCheckError( cublasDgetrfBatched( cublasHandle, n, ZArr, n, pivotArray, infoArray, 1 ) );  

   if (infoArray[ 0 ] != 0) {
      fprintf( stderr, "ComputeMatrixInverse: Problem with LU Decomposition .... \n"); 
      exit ( -1 );  
   }   

   // out of place inverseion here. 
   copy_host_device( &ZInvPtr, ZInvArr, sizeof(real *), cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );  
   cublasCheckError( cublasDgetriBatched( cublasHandle, n, (const real **)ZArr, n, pivotArray, ZInvArr, n, infoArray, 1 ) );  

   if (infoArray[ 0 ] != 0) {
      fprintf( stderr, "ComputeMatrixInverse: Problem with Matrix Inversion.... \n"); 
      exit ( -1 );  
   }   
}

void testKFACPointers( SCRATCH_AREA *scratch )
{
   //real **devPtr = (real **)scratch->nextDevPtr;
   real *host = scratch->nextHostPtr;
   int *pageLck = (int *)scratch->nextPageLckPtr;

	int rows = 5000; 
	int matsize = rows * rows; 

	/*
   real **src = devPtr;
   real **inv = src + 1;
   real **next = inv + 1;

   real *srcData = (real *)(next);
   real *result = srcData + matsize;

   host[ 0 ] = 3; host[ 1 ] = 0; host[ 2 ] = 0;
   host[ 3 ] = 0; host[ 4 ] = 3; host[ 5 ] = 0;
   host[ 6 ] = 0; host[ 7 ] = 0; host[ 8 ] = 3;

	memset( host, 0, sizeof(real) * matsize ); 
	*/

	for (int i = 0; i < rows; i ++) 
		host[ i * rows + i ] = 1; 


	real *devPtr = scratch->nextDevPtr; 
	real *nextDevPtr = devPtr + 2 * matsize; 
	copy_host_device( host, devPtr, sizeof(real) * rows * rows, cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
	matrixInverse( devPtr, devPtr + rows*rows, rows, nextDevPtr, pageLck ); 

	/*
	*src = srcData; 
	copy_host_device( &srcData, src, sizeof(real *), cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	*inv = result; 
	copy_host_device( &result, inv, sizeof(real *), cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

   copy_host_device( host, srcData, sizeof(real) * matsize, cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );

   // Perform the inverse here. 
   cublasCheckError( cublasDmatinvBatched( cublasHandle, rows,
                           (const real **)src, rows,
                           inv, rows,
                           pageLck, 1 ) );
   copy_host_device( host, result, sizeof(real) * matsize, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST );
	*/

   copy_host_device( host, devPtr + matsize, sizeof(real) * matsize, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST );
   for (int i = 0; i < rows; i ++)
      fprintf( stderr, " %f , ", host[ i * rows + i ]);

}


void testKFACTrustRegion (CNN_MODEL *model, DEVICE_DATASET *data, HOST_DATASET *host, 
		SCRATCH_AREA *scratch, 
		real dampGamma, real trMaxRadius, int checkGrad, int master, int slave, int inverseFreq, int epochs, int dataset, real regLambda, int initialization ) {

	char *baseFolder = "/scratch/gilbreth/skylasa/weights/";
	char paramsFolder[ 1024 ]; 
	char paramsFile[ 1024 ]; 
	
	TRUST_REGION_PARAMS trParams; 
	KFAC_CURVATURE_INFO kfacInfo; 


	//begin here
	fprintf( stderr, "Initiating the Trust Region Test now..... \n\n\n");
	initKFACTrustRegionParams( &trParams, data->trainSizeX );
	initKFACInfo ( &kfacInfo ); 

	kfacInfo.dampGamma = dampGamma; 
	kfacInfo.checkGrad = checkGrad; 
	kfacInfo.inverseFreq = inverseFreq; 
	kfacInfo.regLambda = regLambda; 
	trParams.maxDelta = trMaxRadius; 
	trParams.maxIters = epochs; 
	

	data->sampleSize = data->trainSizeX; 
	fprintf( stderr, "... Done parms initialization \n\n"); 

	memset( paramsFile, 0, 1024 ); 
	memset( paramsFolder , 0, 1024 ); 

	switch( dataset ) { 
		case 1: 
			sprintf( paramsFolder, "%scifar10/", baseFolder ); 
			break;
		case 2: 
			sprintf( paramsFolder, "%scifar100/", baseFolder ); 
			break;
		case 3: 
			sprintf( paramsFolder, "%simagenet/", baseFolder ); 
			break;

		default: 
			fprintf( stderr, "Unknown datasetType.... %d \n\n\n", dataset ); 
			exit( -1 ); 
	}

	switch( model->name ) {
		case CNN_LENET: 
			if (model->bias != 0)
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "lenet_default_bias.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "lenet_kaiming_bias.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "lenet_xavier_bias.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			else
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "lenet_default.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "lenet_kaiming.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "lenet_xavier.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 

			break;

		case CNN_ALEXNET: 
			if (model->bias != 0)
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "alexnet_default_bias.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "alexnet_kaiming_bias.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "alexnet_xavier_bias.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			else
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "alexnet_default.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "alexnet_kaiming.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "alexnet_xavier.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			break;

		case CNN_VGG11NET: 
			if (model->bias != 0)
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg11_default_bias.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg11_kaiming_bias.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg11_xavier_bias.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			else
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg11_default.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg11_kaiming.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg11_xavier.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			break;

		case CNN_VGG13NET: 
			if (model->bias != 0)
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg13_default_bias.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg13_kaiming_bias.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg13_xavier_bias.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			else
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg13_default.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg13_kaiming.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg13_xavier.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			break;

		case CNN_VGG16NET: 
			if (model->bias != 0)
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg16_default_bias.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg16_kaiming_bias.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg16_xavier_bias.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			else
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg16_default.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg16_kaiming.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg16_xavier.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			break;
			
		case CNN_VGG19NET: 
			if (model->bias != 0)
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg19_default_bias.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg19_kaiming_bias.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg19_xavier_bias.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			else
				switch( initialization ) { 
					case 0: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg19_default.params" ); 
						break;

					case 1: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg19_kaiming.params" ); 
						break;

					case 2: 
						sprintf( paramsFile, "%s%s", paramsFolder, "vgg19_xavier.params" ); 
						break;

					default: 
						fprintf( stderr, "unknown initialization.... please check \n\n"); 
						exit ( -1 ); 
				} 
			break;

		default: 
			fprintf( stderr, "Unknown initialization param file... \n");
			exit( -1 ); 
	}

	fprintf( stderr, "Reading the params file: %s \n", paramsFile ); 
	readVector( scratch->nextHostPtr, model->pSize, paramsFile, 0, NULL ); 

	copy_host_device( scratch->nextHostPtr, data->weights, sizeof(real) * model->pSize, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	fprintf( stderr, "Starting the KFAC Algorithm now.... \n"); 
	subsampledTrustRegionKFAC( model, data, host, &kfacInfo, &trParams, scratch, master, slave ); 

	fprintf( stderr, ".... Done testing of KFAC Trust Region Done. \n\n\n" ); 
}
