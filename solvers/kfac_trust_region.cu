
#include <solvers/kfac_trust_region.h>

#include <solvers/kfac_utils.h>
#include <solvers/kfac_natural_gradient.h>
#include <solvers/params.h>
#include <solvers/kfac_inverses.h>
#include <solvers/kfac_da.h>
#include <solvers/kfac_thread.h>

#include <utilities/sample_matrix.h>
#include <utilities/alloc_sampled_dataset.h>
#include <utilities/dataset.h>
#include <utilities/dataset_utils.h>
#include <utilities/print_utils.h>
#include <utilities/utils.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/gen_random.h>
#include <device/handles.h>

#include <core/errors.h>

#include <functions/cnn_gradient.h>
#include <functions/cnn_forward.h>
#include <functions/cnn_accuracy.h>
#include <functions/model_reduction.h>
#include <functions/cnn_eval_model.h>
#include <functions/cnn_backward.h>
#include <functions/softmax_loss.h>
#include <functions/dev_batch_norm.h>

int getWordAlignment( unsigned int count ){
	return (count % 32);
}

void copyAndPrint( real *host, real *dev )
{
	copy_host_device( host, dev, sizeof(real) * 3, 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 

	for(int i = 0; i < 3; i ++)
		fprintf( stderr, "%f ", host[ i ] ); 
	fprintf( stderr, "\n"); 
}

void printDataset( SCRATCH_AREA *scratch, DEVICE_DATASET *data, int offset) 
{
	real *hostPtr = scratch->nextHostPtr; 
	real *devPtr = scratch->nextDevPtr; 	

	real *dataset = data->trainSetX + offset * data->features; 
	fprintf( stderr, "First Image.... \n"); 
	copyAndPrint( hostPtr, dataset ); 
	copyAndPrint( hostPtr, dataset + 1024 * 256 ); 
	copyAndPrint( hostPtr, dataset + 2 * 1024 * 256); 

	fprintf( stderr, "Last Image.... \n"); 
	copyAndPrint( hostPtr, dataset + 1024 * 255 ); 
	copyAndPrint( hostPtr, dataset + 1024 * 255 + 1024 * 256  ); 
	copyAndPrint( hostPtr, dataset + 2 * 1024 * 256 + 1024 * 255); 
}


void initBatchNormData( CNN_MODEL *model, real *z, int *zOffsets )
{
	for (int i = 0; i < model->cLayers; i ++ ){
		CONV_LAYER c = model->convLayer[ i ]; 
		if (c.batchNorm != PERFORM_NO_BATCH_NORM) {
			initMeanVariances( 
					z + zOffsets[ i + 1 ] + c.batchNormOffset + c.runningMeansOffset, 
					z + zOffsets[ i + 1 ] + c.batchNormOffset + c.runningVariancesOffset, 
									 c.outChannels ); 
		}
	}
}

void generateParamReport(CNN_MODEL *model, KFAC_CURVATURE_INFO *kfacInfo, TRUST_REGION_PARAMS *trParams )
{
	fprintf( stderr, "********** KFAC_SOLVER_REPORT ***********\n"); 
	fprintf( stderr, "BatchSize: %d\n", model->batchSize ); 
	fprintf( stderr, "MaxTRRadius: %f\n", trParams->maxDelta ); 
	fprintf( stderr, "DampGamma: %f\n", kfacInfo->dampGamma ); 
	fprintf( stderr, "CheckGrad: %d \n", kfacInfo->checkGrad ); 
	fprintf( stderr, "BatchNorm: %d \n", model->enableBatchNorm ); 
	fprintf( stderr, "Bias: %d \n", model->bias ); 
	fprintf( stderr, "InverseFreq: %d \n", kfacInfo->inverseFreq); 
	fprintf( stderr, "Lambda: %f \n", kfacInfo->regLambda); 
	fprintf( stderr, "*****************************************\n"); 	
}


void subsampledTrustRegionKFAC( CNN_MODEL *model, DEVICE_DATASET *data, HOST_DATASET *host, 
      KFAC_CURVATURE_INFO *kfacInfo, TRUST_REGION_PARAMS *trParams, 
		SCRATCH_AREA *scratch, int master, int slave ) { 

	//local variables... 
   real *hostPtr = scratch->nextHostPtr; 
   real *devPtr = scratch->nextDevPtr; 
   real *pageLckPtr = scratch->nextPageLckPtr;

	//Initialization Functions here. 
fprintf( stderr, "1. Computing the KFAC Storage indices... \n\n"); 
	computeKFACStorageIndices( model, kfacInfo ); 


	real *omegaZZT, *lambdaGGT, *omegaZInv, *lambdaGInv, *temp; 
	omegaZZT = lambdaGGT = omegaZInv = lambdaGInv = temp = NULL; 

	real *dampedInput, *dampedZ, *dampedLambda; 
	dampedInput = dampedZ = dampedLambda = NULL; 

	//Device Area Variables
	int *shuffleIndices				= (int *)devPtr;

	if (kfacInfo->inverseFreq == 1) { 
		omegaZInv					= devPtr + data->trainSizeX;
		lambdaGInv					= omegaZInv + kfacInfo->OmegaZOffsets[ model->cLayers + model->lLayers ]; 
		temp							= lambdaGInv + kfacInfo->LambdaGOffsets[ model->cLayers + model->lLayers ]; 
	}  else { 	
		dampedInput						= devPtr + data->trainSizeX;
		dampedZ							= dampedInput + data->features * model->batchSize ;
		dampedLambda					= dampedZ + model->zztSize;
		omegaZInv						= dampedLambda + model->zztSize; 
		lambdaGInv						= omegaZInv + kfacInfo->OmegaZOffsets[ model->cLayers + model->lLayers ]; 
		temp							   = lambdaGInv + kfacInfo->LambdaGOffsets[ model->cLayers + model->lLayers ]; 
	} 

	real *gradient						= temp; 
	real *distGradient				= gradient + model->pSize; 
	real *vector						= distGradient + model->pSize; 	
	real *prevWeights					= vector + model->pSize; 
	real *naturalGradient			= prevWeights + model->pSize; 
	real *z								= naturalGradient + model->pSize; 
	real *dx								= z + model->zSize;								// 1.5
	real *probs							= dx + model->zSize; 
	real *lossFuncErrors				= probs + model->batchSize * data->numClasses; 



	real *nextDevPtr					= lossFuncErrors + model->batchSize * data->numClasses; 
	real *scratchBegin				= nextDevPtr; 
	real *distributionDX; 
	real *distErrors_1; 
	real *distErrors_2; 
	real *distTarget;
	

/*
	real *distributionDX				= dx + model->zSize; 							// 1.5
	real *lossFuncErrors				= distributionDX + model->zSize; 			// 1.5
	real *probs 						= lossFuncErrors + model->maxDeltaSize; 	// 1.5
	real *distErrors_1				= probs + model->maxDeltaSize; 				// 1.5
	real *distErrors_2				= distErrors_1 + model->maxDeltaSize; 		// 1.5
	real *distTarget					= distErrors_2 + model->maxDeltaSize; 		// 1.5
	real *nextDevPtr					= distTarget + model->batchSize; 
*/

	//Host Area Variables
	int *hostIndices					= (int *)hostPtr;
	real *nextHostPtr 				= hostPtr + data->trainSizeX; 

	//PageLock Area Variables
	real *trainModelError			= pageLckPtr; 
	real *testModelError				= trainModelError + 1; 
	real *nrmNatGrad					= testModelError + 1; 
	real *nrmGradient					= nrmNatGrad + 1; 
	real *nextPageLckPtr				= nrmGradient + 1; 


	//Miscellaneous Variables here. 
	real 	start, total; 
	real  stInverses, totalInverses, sumInverses, KFACDistTotal, stKFACDist ; 
	real  stGrad, totalGrad, sumGrad; 
	real  stHessVec, totalHessVec, sumHessVec; 
	real 	trainingLikelihood, newTrainingLikelihood; 
	real	trainingAccuracy, newTrainingAccuracy; 
	real 	testAccuracy, testLikelihood; 
	real	kfacStep, gradientStep;
	real 	mErrKfac, mErrGradient; 	
	real	alpha; 
	real	rho; 
	unsigned int	offset, curBatchSize, miniBatchNo; 
	int	components					= model->pSize; 
	real	assertCondition; 
	real 	nrmWeights; 

	real stAugment, stNGrad, stAccuracy; 
	real totalAugment, totalNGrad, totalAccuracy; 
	real sumAugment, sumNGrad, sumAccuracy; 
	real stRandom, sumRandom; 
	real stZZT, totalZZT, sumZZT; 

	real loopStart, loopTotal; 
	real loopSecond, loopSecondTotal; 
	real tempTime; 

	//THREADED Sturctures here. 
	KFAC_THREAD_INFO					kfacThreadInfo; 
	KFAC_THREAD_ARGS					kfacThreadArgs; 

	//KFAC Struct Initializations here..
	kfacInfo->OmegaZZT 				= NULL; 
	kfacInfo->LambdaGGT 				= NULL; 
	kfacInfo->OmegaZInv 				= omegaZInv; 
	kfacInfo->LambdaGInv 			= lambdaGInv; 
	kfacInfo->vec 						= vector; 
	kfacInfo->nGradient 				= naturalGradient; 
	kfacInfo->gradient				= gradient; 

	kfacInfo->dampedInput			= dampedInput; 
	kfacInfo->dampedZ					= dampedZ; 
	kfacInfo->dampedLambda			= dampedLambda; 

	generateParamReport( model, kfacInfo, trParams ); 

	//begin code here. 
fprintf( stderr, "Allocating Sampling Dataset... %d\n", model->batchSize); 
   allocSampledDataset( data, 2 * model->batchSize);

	// Initialization before the code. 
   scratch->nextDevPtr = nextDevPtr;
   scratch->nextHostPtr = nextHostPtr;
   scratch->nextPageLckPtr = nextPageLckPtr;



	//Threaded Implementation here. 
	kfacThreadArgs.kfacThreadInfo  = & (kfacThreadInfo); 
	kfacThreadArgs.kfacInfo 		 = kfacInfo; 
	kfacThreadArgs.model				 = model;

	kfacThreadInfo.masterDevice	= master; 
	kfacThreadInfo.slaveDevice		= slave; 

	kfacThreadInfo.slaveWorkspace		= NULL; 
	kfacThreadInfo.slavePageLckPtr 	= NULL; 

	kfacThreadInfo.workComplete		= 0; 
	kfacThreadInfo.signalled			= 0; 
	kfacThreadInfo.batchNo				= 0; 

/*
	if (kfacInfo->inverseFreq != 1) { 
		initKFACThread( &kfacThreadInfo ); 
		createKFACThread( &kfacThreadInfo, &kfacThreadArgs ); 
		sem_wait( &kfacThreadInfo.readySemaphore ); 
	}
*/

	//main Loop
	miniBatchNo = 0; 
	cuda_memset( prevWeights, 0, sizeof(real) * model->pSize, ERROR_MEMSET ); 

// TODO
// TODO DO WE NEED THIS CLEANUP>... 
//	initKFACData( model, kfacInfo ); 
//fprintf( stderr, "OmegaZInv and LambdaGInv initialized... \n"); 
// TODO
// TODO

	initBatchNormData( model, z, model->zOffsets  ); 

	for (int iter = 0; iter < trParams->maxIters; iter ++) {

#ifdef DEBUG_TRUST_REGION
fprintf( stderr, "Iteration: %d, -------- BEGIN --------- \n", iter); 
#endif

		offset = 0; 
		curBatchSize = model->batchSize; 

		stInverses = totalInverses = sumInverses = 0; 
		stGrad = totalGrad = sumGrad = 0; 
		stHessVec = totalHessVec = sumHessVec = 0; 

		stAugment = stNGrad = stAccuracy = 0; 
		totalAugment = totalNGrad = totalAccuracy = 0; 
		sumAugment = sumNGrad = sumAccuracy = 0; 
	
		stRandom = sumRandom = 0; 
		stZZT = totalZZT = sumZZT = 0; 
	
		loopStart = loopTotal = 0; 
		loopSecond = loopSecondTotal = 0; 

		KFACDistTotal = 0; 

		start = Get_Time (); 

		stRandom = Get_Time (); 
   	genRandomVector( hostIndices, data->trainSizeX, data->trainSizeX );  
		copy_host_device( hostIndices, shuffleIndices, sizeof(int) * data->trainSizeX, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
		sumRandom += Get_Timing_Info( stRandom );


		while (offset < data->trainSizeX ) {

			loopSecond = Get_Time (); 

			if ((offset + model->batchSize) < data->trainSizeX ) {
				curBatchSize = model->batchSize; 
			} else {
				//curBatchSize = data->trainSizeX - offset; 
				break;
			}


#ifdef DEBUG_TRUST_REGION
fprintf( stderr, "\n\n\n\n****Iteration: %d, Offset: %d, BatchSize: %d ****\n", iter, offset, curBatchSize ); 
#endif

#ifdef DEBUG_TRUST_REGION
fprintf( stderr, "Intiating Data Augmentation... \n" ); 
#endif

			//get the sample here. 
			stAugment = Get_Time (); 

			switch( host->datasetType ) { 
				case CIFAR10: 
				case CIFAR100: 
					selectColumnMatrix( data, curBatchSize, shuffleIndices, offset ); 
					break;
	
				case IMAGENET: 
					selectHostMatrix( host, data->sampledTrainX, data->sampledTrainY, 
							curBatchSize, hostIndices, offset, nextHostPtr ); 
					break; 

				default: 
					fprintf( stderr, "Problem with the dataset initialization.... \n\n"); 
					exit( -1 ); 
			} 
			augmentData( model, data, offset, curBatchSize, nextDevPtr, nextHostPtr , 1, data->datasetType ); 

			totalAugment = Get_Timing_Info( stAugment ); 
			sumAugment += totalAugment; 

			//compute Gradient
			stGrad = Get_Time (); 

/*
			trainingLikelihood = computeCNNGradient( model, data, scratch, z, dx, probs, lossFuncErrors, 
																	gradient, offset, curBatchSize, kfacInfo->regLambda ); 
*/

			scratch->nextDevPtr = scratchBegin;

			trainingLikelihood = cnnForward( model, data, scratch, z, probs, lossFuncErrors, 
											offset, curBatchSize, MODEL_TRAIN ); 

			distErrors_1 = scratchBegin;
			distErrors_2 = distErrors_1 + model->maxDeltaSize; 
			nextDevPtr = distErrors_2 + model->maxDeltaSize; 
			copy_device( distErrors_1, lossFuncErrors, sizeof(real) * model->batchSize * data->numClasses, 
								ERROR_MEMCPY_DEVICE_DEVICE ); 

			cnnBackward( model, data, nextDevPtr, z, gradient, dx, distErrors_1, distErrors_2, 
								offset, curBatchSize, nextHostPtr ); 

			alpha = kfacInfo->regLambda; 
			cublasCheckError( cublasDaxpy( cublasHandle, model->pSize, &alpha,  
														data->weights, 1, gradient, 1) ); 


			totalGrad = Get_Timing_Info( stGrad ); 
			sumGrad += totalGrad; 

#ifdef DEBUG_TRUST_REGION
cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, gradient, 1, nextPageLckPtr ) ); 
fprintf( stderr, "Done with Gradient evaluation....Norm(grad, 2): %f \n", *nextPageLckPtr ); 
fprintf( stderr, "Model Value: %f\n\n", trainingLikelihood ); 
cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, data->weights, 1, nextPageLckPtr ) ); 
fprintf( stderr, "Weights norm is : %f \n", *nextPageLckPtr ); 
#endif

			// do the backward pass with distribution predictions here. 
			// this is used to compute Natural Gradient
			stKFACDist = Get_Time (); 

			distErrors_1 = scratchBegin;
			distErrors_2 = distErrors_1 + model->maxDeltaSize; 
			distributionDX = distErrors_2 + model->maxDeltaSize;
			distTarget = distributionDX + model->zSize; 
			nextDevPtr = distTarget + model->batchSize; 

			computeDistributionErrors( probs, distTarget, distErrors_1, curBatchSize, data->numClasses, 
												nextDevPtr, nextHostPtr ); 

			nextDevPtr = distributionDX + model->zSize; 

			cnnBackward( model, data, nextDevPtr, z, distGradient, distributionDX, distErrors_1, distErrors_2, 
								offset, curBatchSize, nextHostPtr ); 

			KFACDistTotal += Get_Timing_Info( stKFACDist ); 

			//Compute Lambda and Deltas here. 

			stZZT = Get_Time (); 
			stInverses = Get_Time (); 

			// Store the Z and Dx variables here. 
			if ((dampedInput != NULL) && ( dampedZ != NULL )) { 
					computeSpeedUpOmegaZ( kfacInfo, model, data, curBatchSize, z, model->zOffsets, 
						dampedInput, dampedZ, model->zztOffsets, iter + offset ); 
					computeSpeedUpLambdaDelta( kfacInfo, model, curBatchSize, distributionDX, 
						dampedLambda, model->zOffsets, model->zztOffsets, iter + offset ); 
			}

			// Single Threaded Implementation for every mini batch
			if ((kfacInfo->inverseFreq == 1) || ((iter == 0) && (miniBatchNo == 0)) ){

				computeOmegaZ( kfacInfo, model, data, curBatchSize, NULL, z, kfacInfo->OmegaZInv, 
					model->zOffsets, model->zztOffsets, 
					nextDevPtr, nextPageLckPtr, cublasHandle, NULL, cusolverHandle  ); 
				computeLambdaDelta( kfacInfo, model, curBatchSize, distributionDX, NULL, kfacInfo->LambdaGInv, 
					model->zOffsets, model->zztOffsets, 
					nextDevPtr, nextPageLckPtr, cublasHandle, NULL, cusolverHandle ); 

			}  else if ((miniBatchNo % kfacInfo->inverseFreq) == 0){ 

				computeOmegaZ( kfacInfo, model, NULL, curBatchSize, dampedInput, dampedZ, kfacInfo->OmegaZInv, 
					model->zOffsets, model->zztOffsets, 
					nextDevPtr, nextPageLckPtr, cublasHandle, NULL, cusolverHandle  ); 
				computeLambdaDelta( kfacInfo, model, curBatchSize, NULL, dampedLambda, kfacInfo->LambdaGInv, 
					model->zOffsets, model->zztOffsets, 
					nextDevPtr, nextPageLckPtr, cublasHandle, NULL, cusolverHandle ); 

			} 

			totalZZT = Get_Timing_Info ( stZZT ); 
			sumZZT += totalZZT ;
			totalInverses = Get_Timing_Info( stInverses ); 
			sumInverses += totalInverses; 

#ifdef DEBUG_TRUST_REGION
fprintf( stderr, "Done with computing Omega and Delta Inverses... \n"); 
#endif

			//compute Natural Gradient
			//printNorms( model, kfacInfo ); 
			stNGrad = Get_Time (); 

			nextDevPtr = scratchBegin; 

			computeNaturalGradient( model, kfacInfo, nextDevPtr, nextHostPtr ); 

			totalNGrad = Get_Timing_Info( stNGrad ); 
			sumNGrad += totalNGrad; 

			//assert the KFAC matrix here. 
			loopStart = Get_Time (); 

#ifdef DEBUG_TRUST_REGION
cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, kfacInfo->nGradient, 1, nextPageLckPtr )); 
fprintf( stderr, "Done with Natural Gradient ....Norm( natGradient, 2): %f \n", *nextPageLckPtr ); 
#endif


			cublasCheckError( cublasDdot( cublasHandle, model->pSize, 
				kfacInfo->nGradient, 1, kfacInfo->nGradient, 1, &assertCondition )); 
			if (assertCondition < 0) {
				fprintf (stderr, " **** KFAC MATRIX IS NOT PSD.... \n\n\n"); 
				exit( -1 ); 
			}

			tempTime = Get_Timing_Info( loopStart ); 
			loopTotal += tempTime; 
//fprintf( stderr, "TrustRegion: Loop Variant Timings: %f \n", temp ); 


			// compute step size (eta)
			//*nrmNatGrad = 0; 
			//cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, kfacInfo->nGradient, 1, nrmNatGrad ) );  
			//kfacStep = trParams->delta / (*nrmNatGrad); 
			stHessVec = Get_Time( ); 

			//This needs probs, lossFuncErrors and, dx

			nextDevPtr = scratchBegin;

			mErrKfac = computeQuadraticModel( model, data, z, probs, lossFuncErrors, dx, 
								gradient, kfacInfo->nGradient, trParams->delta, offset, curBatchSize, kfacInfo->regLambda, 
								nextDevPtr, nextHostPtr, nextPageLckPtr ); 

#ifdef DEBUG_TRUST_REGION
fprintf( stderr, "Done with KFAC Direction Model Reduction... %f\n", mErrKfac); 
#endif

			// using Gradient
			//*nrmGradient = 0; 
			//cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, gradient, 1, nrmGradient ) );  
			//gradientStep = trParams->delta / (*nrmGradient); 
			mErrGradient = 0; 
			if (kfacInfo->checkGrad) {

				nextDevPtr = scratchBegin;

				mErrGradient = computeQuadraticModel( model, data, z, probs, lossFuncErrors, dx, 
									gradient, gradient, trParams->delta, offset, curBatchSize, kfacInfo->regLambda,
									nextDevPtr, nextHostPtr, nextPageLckPtr ); 
			}

#ifdef DEBUG_TRUST_REGION
fprintf( stderr, "Done with Gradient Direction Model Reduction... %f\n", mErrGradient); 
#endif

			totalHessVec = Get_Timing_Info( stHessVec ); 
			sumHessVec += totalHessVec; 




			// Trust Region part here. 
			if (!kfacInfo->checkGrad || (mErrKfac < mErrGradient )) {
				// Use the KFAC direction here. 
				// Log Likelihood and Accuracy with new set of weights. 
				// Update Weights using kFAC direction. 

				// add Momentum term to KFAC direction. 
				alpha = kfacInfo->momentum; 
				cublasCheckError( cublasDaxpy( cublasHandle, components, &alpha, 
											prevWeights, 1, kfacInfo->nGradient, 1 ) ); 

				// Step Increment to the weights. 
				alpha = -1.; 
				cublasCheckError( cublasDaxpy( cublasHandle, components, 
										&alpha, kfacInfo->nGradient, 1, data->weights, 1 ) ); 
				copy_device( prevWeights, kfacInfo->nGradient, sizeof(real) * components, 
										ERROR_MEMCPY_DEVICE_DEVICE ); 

				// New Statistics. 
				stAccuracy = Get_Time (); 

				nextDevPtr = scratchBegin;
				scratch->nextDevPtr = nextDevPtr; 

				newTrainingLikelihood = evaluateCNNModel( model, data, scratch, z, probs, 
													lossFuncErrors, offset, curBatchSize ); 
				newTrainingAccuracy = computeAccuracy( probs, data->sampledTrainY, curBatchSize, 
												data->numClasses, nextDevPtr, nextPageLckPtr ); 
				totalAccuracy = Get_Timing_Info( stAccuracy ); 
				sumAccuracy += totalAccuracy; 


				// Model Reduction Term
				rho = ( newTrainingLikelihood - trainingLikelihood ) / (mErrKfac - 1e-16);
#ifdef DEBUG_TRUST_REGION
cublasCheckError( cublasDnrm2( cublasHandle, components, kfacInfo->nGradient, 1, &nrmWeights ) ); 
fprintf( stderr, "Iteration: %d, Offset: %d, Rho (KFAC Direction): %f, Nrm(dir, 2): %e \n", iter, offset, rho, nrmWeights ); 
#endif

			} else {
				// Use the Gradient direction here. 
				// Log Likelihood and Accuracy with new set of weights. 

				// add Momentum Term here. 
				alpha = kfacInfo->momentum; 
				cublasCheckError( cublasDaxpy( cublasHandle, components, &alpha, 
											prevWeights, 1, gradient, 1 ) ); 

				// Step Increment the weights. 
				alpha = -1.; 
				cublasCheckError( cublasDaxpy( cublasHandle, components, &alpha, 
											gradient, 1, data->weights, 1 ) ); 
				copy_device( prevWeights, gradient, sizeof(real) * components , 
										ERROR_MEMCPY_DEVICE_DEVICE ); 

				// New statistics. 
				stAccuracy = Get_Time (); 

				nextDevPtr = scratchBegin;
				scratch->nextDevPtr = nextDevPtr; 

				newTrainingLikelihood = evaluateCNNModel( model, data, scratch, z, probs, 
														lossFuncErrors, offset, curBatchSize ); 
				newTrainingAccuracy = computeAccuracy( probs, data->sampledTrainY, curBatchSize, 
														data->numClasses, nextDevPtr, nextPageLckPtr ); 
				totalAccuracy = Get_Timing_Info( stAccuracy ); 
				sumAccuracy += totalAccuracy ;


				// Model Reduction Term
				rho = ( newTrainingLikelihood - trainingLikelihood ) / ( mErrGradient - 1e-16); 
#ifdef DEBUG_TRUST_REGION
cublasCheckError( cublasDnrm2( cublasHandle, components, gradient, 1, &nrmWeights ) ); 
fprintf( stderr, "Iteration: %d, Offset: %d, Rho (Gradient Direction): %e, Nrm(dir, 2): %e \n", 
								iter, offset, rho, nrmWeights ); 
#endif
			}

			// Incerment/Decrement trust region radius, depending 
			// on projected model reduction. 
			if (rho > 0.75) 
				trParams->delta = min( trParams->maxDelta, 2. * trParams->delta ); 

			if ( rho < 0.25 )
				trParams->delta = max( trParams->minDelta, 0.5 * trParams->delta ); 

			if ((rho < 1e-4) || (newTrainingLikelihood > (10. * trainingLikelihood))) {
#ifdef DEBUG_TRUST_REGION
				fprintf( stderr, "KFAC-TRUST-REGION: Rejecting this step ...Iteration: %d, Offset: %d \n", 
												iter, offset ); 
#endif

				//Revert back the model weights here. 
				alpha = 1.; 
				cublasCheckError( cublasDaxpy( cublasHandle, components, &alpha, 
											prevWeights, 1, data->weights, 1 ) ); 				
			}

#ifdef DEBUG_TRUST_REGION
cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, data->weights, 1, &nrmWeights ) ); 
			if (mErrKfac < mErrGradient ) {
fprintf( stderr, "Iter: %d, Offset: %d, %e, %e, %e, %e, %e, kfac, %e \n", iter, offset, trainingLikelihood, newTrainingLikelihood, rho, mErrKfac, nrmWeights, trParams->delta ); 
			} else {
fprintf( stderr, "Iter: %d, offset: %d, %e, %e, %e, %e, %e, grad, %e \n", iter, offset, trainingLikelihood, newTrainingLikelihood, rho, mErrGradient, nrmWeights, trParams->delta ); 
			}
#endif

			// Increment the pointers here. 
			offset += curBatchSize; 
			miniBatchNo ++; 

			//exit( -1 ); 


			loopSecondTotal += Get_Timing_Info( loopSecond ); 


		} // Pass over the Dataset. 
#ifdef DEBUG_TRUST_REGION
		fprintf( stderr, "Done with Iteration: %d \n\n", iter ); 
#endif

		total = Get_Timing_Info( start ); 

		// Statistics over here. 
		// Training Loss/Accuracy over the entire dataset here. 
		// Time for each epoch

		nextDevPtr = scratchBegin;
		scratch->nextDevPtr = nextDevPtr; 

		computeTestGeneralizationErrors( model, data, scratch, 
										z, probs, lossFuncErrors, &testLikelihood, &testAccuracy ); 

		computeTrainGeneralizationErrors( model, data, host, scratch, 
										z, probs, lossFuncErrors, &trainingLikelihood, &trainingAccuracy ); 
		
		cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, data->weights, 1, &nrmWeights ) ); 

		fprintf( stderr, " Iteration: %d\t, TrainLikelihood: %f\t, TrainAccuracy: %3.2f\t, TestLikelihood: %f\t, TestAccuracy: %3.2f\t, Weight: %e\t, IterTime: %6.3f\t, ZZTTime: %6.3f\t, InverseTime: %6.3f\t, GradTime: %6.3f\t, HvTime: %6.3f\t, KFACDistPass: %6.3f\t, Augmentation: %6.3f\t, NatGrad: %6.3f\t, AccuracyTime: %6.3f\t, Random: %6.3f\t, Loop Variant: %6.3f, second: %6.3f, TR Radius: %f \n\n", iter, trainingLikelihood, trainingAccuracy, testLikelihood, testAccuracy, nrmWeights, total, sumZZT, sumInverses, sumGrad, sumHessVec, KFACDistTotal, sumAugment, sumNGrad, sumAccuracy, sumRandom, loopTotal, loopSecondTotal, trParams->delta ); 


	} // Epochs pass
	
}
