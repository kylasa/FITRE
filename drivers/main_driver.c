
#include <device/query.h>
#include <device/cuda_environment.h>
#include <device/cuda_utils.h>

#include <core/datadefs.h>

#include <drivers/dataset_driver.h> 
#include <drivers/gradient_driver.h>
#include <drivers/hessian_driver.h>
#include <drivers/model_driver.h>
#include <drivers/derivative_driver.h>
#include <drivers/reduce_driver.h>
#include <drivers/trust_region_driver.h>
#include <drivers/kfac_trust_region_driver.h>
#include <drivers/gauss_newton_driver.h>

#include <drivers/momentum_driver.h>
#include <drivers/adagrad_driver.h>
#include <drivers/nesterov_driver.h>
#include <drivers/adam_driver.h>
#include <drivers/rmsprop_driver.h>

#include <drivers/norm_driver.h>
#include <drivers/cg_driver.h>
#include <drivers/convolution_driver.h>
#include <drivers/test_cnn_gradient.h>
#include <drivers/cnn_driver.h>
#include <drivers/conv_cnn_driver.h>
#include <drivers/cnn_driver_new.h>
#include <drivers/reshape_driver.h>
#include <drivers/pool_driver.h>
#include <drivers/mem_driver.h>

#include <drivers/augmentation_driver.h>

#include <utilities/print_utils.h>
#include <utilities/utils.h>
#include <utilities/mem_estimate.h>

#include <nn/read_nn.h>
#include <nn/read_alexnet.h>
#include <nn/read_vgg.h>

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

//#define  CIFAR_BATCH_SIZE 500

cublasHandle_t cublasHandle; 
cusparseHandle_t cusparseHandle;
cusolverDnHandle_t cusolverHandle;
curandGenerator_t curandGeneratorHandle;

int BLOCK_SIZE; 
int WARP_SIZE; 
int DEVICE_NUM_BLOCKS;
int __THREADS_PER_SAMPLE__ ;

int main( int argc, char **argv )
{
	CNN_MODEL lenetModel; 
	NN_MODEL curvesModel; 
	HOST_DATASET hostData; 
	DEVICE_DATASET deviceData;
	SCRATCH_AREA scratch;

	int CIFAR_BATCH_SIZE; 
	real dampGamma; 	
	real trMaxRadius; 
	real regLambda; 
	int checkGrad; 
	int enableBias; 
	int enableBatchNorm; 
	int MASTER_GPU; 
	int SLAVE_GPU;
	int inverseFreq; 
	int network; 
	int epochs;
   int dataset; 
	int initialization; 

   fprintf( stderr, "Total No. of args passed: %d \n", argc );
   if (argc < 15) {
      fprintf( stderr, "Please enter sufficient parameters and retry... \n\n\n ");
      fprintf( stderr, "<exe> <testcase> batchSize dampGamma trMaxRadius checkGrad enableBias enableBatchNorm master slave freq network epochs dataset lambda init\n\n\n");
      exit ( -1 );
   } 

	int testNo = atoi( argv[1] ); 
	if (testNo > 1000) {
		deviceMemAllocTest ();
		return 1;
	}


	CIFAR_BATCH_SIZE  = atoi( argv[ 2 ] ); 
	dampGamma = atof( argv[ 3 ] ); 
	trMaxRadius = atof( argv[ 4 ] ); 
	checkGrad = atoi( argv[ 5 ] ); 
	enableBias = atoi( argv[ 6 ] ); 
	enableBatchNorm = atoi( argv[ 7 ] ); 
	MASTER_GPU = atoi( argv[ 8 ] ); 	
	SLAVE_GPU = atoi( argv[ 9 ] ); 
	inverseFreq = atoi( argv[ 10 ] ); 
	network = atoi( argv[ 11 ] ); 
	epochs = atoi( argv[ 12 ] ); 
   dataset = atoi( argv[ 13 ] );
	regLambda = atof( argv[ 14 ] ); 

	if (argc == 16)
		initialization = atoi( argv[ 15 ] ); 
	else
		initialization = 0; 


	//DEVICE initialization
	getDeviceParameters ();
	cuda_env_init( &scratch, MASTER_GPU ); 


   //Dataset initialization
   switch( dataset ) {
      case 1:
         lenetModel.batchSize = CIFAR_BATCH_SIZE;
			hostData.datasetType = CIFAR10; 
         getCIFAR10( &lenetModel, &hostData, &deviceData, &scratch );
			__THREADS_PER_SAMPLE__ = 1; 
         break;

      case 2:
         lenetModel.batchSize = CIFAR_BATCH_SIZE;
			hostData.datasetType = CIFAR100; 
         getCIFAR100( &lenetModel, &hostData, &deviceData, &scratch );
			__THREADS_PER_SAMPLE__ = 1; 
         break;

		case 3: 
         lenetModel.batchSize = CIFAR_BATCH_SIZE;
			hostData.datasetType = IMAGENET; 
         getTinyImageNet( &lenetModel, &hostData, &deviceData, &scratch );
			__THREADS_PER_SAMPLE__ = 1; 
         break;

      default:
         fprintf( stderr, " Unknown dataset parameter: %d \n", dataset );
         exit( -1 );
   }



   switch( network ) {
      case 1:
         readLenetCNN( &lenetModel, 3, deviceData.height, deviceData.width, CIFAR_BATCH_SIZE, enableBias, enableBatchNorm);
         break;

      case 2:
         readAlexNetCNN( &lenetModel, CIFAR_BATCH_SIZE, deviceData.height, deviceData.width, deviceData.numClasses, enableBias, enableBatchNorm  );
         break;

      case 3:
         readVGG11( &lenetModel, CIFAR_BATCH_SIZE, deviceData.height, deviceData.width, deviceData.numClasses, 3, enableBias, enableBatchNorm, hostData.datasetType );
         break;

      case 4:
         readVGG13( &lenetModel, CIFAR_BATCH_SIZE, deviceData.height, deviceData.width, deviceData.numClasses, 3, enableBias, enableBatchNorm, hostData.datasetType );
         break;

      case 5:
         readVGG16( &lenetModel, CIFAR_BATCH_SIZE, deviceData.height, deviceData.width, deviceData.numClasses, 3, enableBias, enableBatchNorm, hostData.datasetType );
         break;

      case 6:
         readVGG19( &lenetModel, CIFAR_BATCH_SIZE, deviceData.height, deviceData.width, deviceData.numClasses, 3, enableBias, enableBatchNorm, hostData.datasetType );
         break;

      default:
         fprintf( stderr, "Unknown NETWORK Name: %d \n", network );
         exit( -1 );
   }

	//Weights is allocated here. 
	cnnInitializations( &lenetModel, &deviceData ); 


	// now allocate the remaining workspace here. 
	cuda_allocate_workspace( &scratch, MASTER_GPU ); 

	switch(  testNo ) {
		case 0: 
			testHessianVec( &curvesModel, &deviceData, &scratch );
		break;

		case 1: 
			testGradient( &curvesModel, &deviceData, &scratch ); 
		break;

		case 2: 
			testModel (&curvesModel, &deviceData, &scratch ); 
		break;

		case 3:
			//runDerivativeTest( &curvesModel, &deviceData, &scratch ); 
			runCNNDerivativeTest( &lenetModel, &deviceData, &scratch ); 
		break;

		case 4:
			estimateMem( &curvesModel, &deviceData ); 
		break;

		case 5: 
			print_device_mem_usage ();
		break;

		case 6: 
			getDeviceParameters (); 
		break;

		case 7:
			testReduce (&scratch); 
		break;

		case 8: 
			//testTrustRegion( &curvesModel, &deviceData, &scratch ); 
			testKFACTrustRegion( &lenetModel, &deviceData, &hostData, &scratch, dampGamma, trMaxRadius, checkGrad, MASTER_GPU, SLAVE_GPU, inverseFreq, epochs, dataset, regLambda, initialization ); 
		break;

		case 9: 
			testGaussNewton ( &curvesModel, &deviceData, &scratch ); 
		break;

		case 11: 
			testMomentumSGD( &curvesModel, &deviceData, &scratch ); 
		break;

		case 12: 
			testAdagrad( &curvesModel, &deviceData, &scratch ); 
		break;

		case 13: 
			testNesterov( &curvesModel, &deviceData, &scratch ); 
		break;

		case 14: 
			testRMSProp ( &curvesModel, &deviceData, &scratch ); 
		break;

		case 15: 
			testAdam ( &curvesModel, &deviceData, &scratch ); 
		break;


		case 19: 
			testNorm( &scratch );
		break;

		case 20: 
			testCG( &curvesModel, &deviceData, &scratch );
		break;

		case 100: 
			testConvCNN( &lenetModel, &deviceData, &scratch ); 	
			break;

		case 101: 
			testConvolution( &lenetModel, &deviceData, &scratch ); 
			break;

		case 102: 
			testBackPropConvolution ( &lenetModel, &deviceData, &scratch ); 	
			break;

		case 103: 
			testReshape( &lenetModel, &deviceData, &scratch ); 
			break;

		case 104: 
			testAugmentation( &lenetModel, &deviceData, &scratch ); 
			break;

		case 110: 
			testCNN( &lenetModel, &deviceData, &scratch ); 
			//testCNNGradient( &lenetModel, &deviceData, &scratch ); 
			break;

		case 120: 
			//testPoolForwardPass( &lenetModel, &deviceData, &scratch ); 
			testPoolDerivative( &lenetModel, &deviceData, &scratch ); 
			break;

		case 130: 
			//getMemRequired( &lenetModel ); 
			testCudaMemcpy2D( &scratch ); 

			break;
			

		default:
			fprintf( stderr, "Input the right test case... \n"); 
	}

	return 1;
}
