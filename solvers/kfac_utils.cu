
#include <solvers/kfac_utils.h>
#include <solvers/kfac_structs.h>

#include <functions/cnn_backward.h>

void computeKFACStorageIndices (CNN_MODEL *model,  KFAC_CURVATURE_INFO *kfacInfo) 
{
	int *omegaOffsets, *lambdaOffsets; 

	omegaOffsets = kfacInfo->OmegaZOffsets; 
	lambdaOffsets = kfacInfo->LambdaGOffsets; 	

	int inChannels, kSize, outChannels; 

	for (int i = 0; i < MAX_LAYERS; i ++) 
		omegaOffsets[ i ] = lambdaOffsets[ i ] = 0; 		

	for (int i = 0; i < model->cLayers; i ++) {
		//ZZT and Inverse Sizes
		// inChannels * kSize * kSize
		inChannels = model->convLayer[ i ].inChannels; 
		outChannels = model->convLayer[ i ].outChannels; 
		kSize = model->convLayer[ i ].kSize; 
		omegaOffsets[ i+1 ] = ( inChannels * kSize * kSize + 1) * (inChannels * kSize * kSize + 1) + 
			omegaOffsets[ i ]; 

		//GGT and Inverse Sizes; 
		// outChannels * outChannels
		lambdaOffsets[ i+1 ] = outChannels * outChannels + 
			lambdaOffsets[ i ]; 
	}

	for (int i = 0; i < model->lLayers; i ++) {
		//ZZT and Inverse Sizes
		// in * in
		inChannels = model->fcLayer[ i ].in;
		outChannels = model->fcLayer[ i ].out; 
		omegaOffsets[ model->cLayers + i + 1 ] = (inChannels + 1) * (inChannels + 1) + 
			omegaOffsets[ model->cLayers + i ]; 

		//GGT and Inverse Sizes
		// out * out
		lambdaOffsets[ model->cLayers + i + 1 ] = outChannels * outChannels + 
			lambdaOffsets[ model->cLayers + i ]; 
	}

	fprintf( stderr, " ------ Omega Offsets.......... \n\n"); 
	for (int i = 0; i < model->cLayers; i ++)
		fprintf( stderr, " Layer: %d ---- Size: %d, %d \n", i, omegaOffsets[ i ], lambdaOffsets[ i ] ); 
	fprintf( stderr, "\n" ); 
	for (int i = 0; i <= model->lLayers; i ++)
		fprintf( stderr, " Layer: %d ---- Size: %d,  \n", i, omegaOffsets[ model->cLayers + i ], lambdaOffsets[ model->cLayers + i ]  ); 
	fprintf( stderr, "\n" ); 
}

/*
	Do the backward Pass with the Distribution Targets
*/

void updateDeltasWithDistribution (CNN_MODEL *model,
	real *z, real *dx, DEVICE_DATASET *data, int offset, int samples, int numClasses, 
	real *devPtr, real *hostPtr) 
{

	real *gradient = devPtr; 	
	real *errors	= gradient + model->pSize; 
	real *errors_1 = errors + model->maxDeltaSize; 
	real *nextDevPtr = errors_1 + model->maxDeltaSize; 

	real *nextHostPtr = hostPtr; 

   cnnBackward( model, data, nextDevPtr, z, gradient, dx, errors, errors_1, 
         offset, samples, nextHostPtr );  
}
