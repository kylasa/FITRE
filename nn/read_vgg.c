
#include <nn/read_vgg.h>

#include <nn/utils.h>
#include <nn/read_nn.h>
#include <nn/nn_decl.h>

#include <core/errors.h>
#include <core/structdefs.h>
#include <device/cuda_utils.h>


#include <stdlib.h>
#include <stdio.h>

/*
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
*/

void makeLayer( CONV_LAYER *conv, POOL_LAYER *pool, POOL_TYPE poolFun, int inChannels, int outChannels, 
						int height, int width, int *outHeight, int *outWidth, int batchSize, BATCH_NORM_TYPES batchnorm  )
{
	int convOutHeight, convOutWidth; 

	conv->height 			= height; 
	conv->width 			= width; 
	conv->stride 			= 1; 
	conv->padding 			= 1; 
	conv->kSize 			= 3; 
	conv->inChannels 		= inChannels; 
	conv->outChannels 	= outChannels; 

   conv->batchNorm 			= batchnorm;

	getDimensions( conv->height, conv->width, conv->padding, conv->stride, conv->kSize, &convOutHeight, &convOutWidth ); 
   conv->outHeight 	= convOutHeight;  
   conv->outWidth 	= convOutWidth;  

   pool->type 			= poolFun;
   pool->pSize			= 2;  
   pool->stride 		= 2;
   pool->padding 		= 0;
   pool->height 		= conv->outHeight;  
   pool->width 		= conv->outWidth;  

	if ( poolFun == NO_POOL ){
   	pool->outHeight 	= pool->height; 
   	pool->outWidth 	= pool->width; 
		*outHeight = pool->height; 
		*outWidth = pool->width; 
	} else {
   	getDimensions( convOutHeight, convOutWidth, 0, 2, 2, outHeight, outWidth);
   	pool->outHeight 	= *outHeight; 
   	pool->outWidth 	= *outWidth; 
	}


   if (conv->batchNorm == PERFORM_NO_BATCH_NORM){
		//Offsets. 
   	conv->activationOffset 	= conv->outHeight * conv->outWidth * conv->outChannels * batchSize; 

		if (poolFun == NO_POOL) {
			conv->poolOffset = INT_MAX; 
			conv->outputOffset = conv->activationOffset; 
		} else {
			conv->poolOffset = 2 * conv->activationOffset; 
      	conv->outputOffset = conv->poolOffset;
		}
		conv->batchNormOffset = INT_MAX; 
      conv->meansOffset = conv->variancesOffset = INT_MAX;  
		conv->runningMeansOffset = conv->runningVariancesOffset = INT_MAX; 
   } else {
		if (poolFun == NO_POOL){
      	conv->batchNormOffset 	= conv->outHeight * conv->outWidth * conv->outChannels * batchSize;
      	conv->meansOffset 		= conv->outHeight * conv->outWidth * conv->outChannels * batchSize; 
      	conv->variancesOffset 	= conv->meansOffset + conv->outChannels;
			conv->runningMeansOffset = conv->variancesOffset + conv->outChannels; 
			conv->runningVariancesOffset = conv->runningMeansOffset + conv->outChannels;

			conv->activationOffset	= 2 * conv->batchNormOffset + 4 * conv->outChannels; 
			conv->poolOffset			= INT_MAX; 
      	conv->outputOffset 		= conv->activationOffset;

		} else {

			conv->batchNormOffset = conv->outHeight * conv->outWidth * conv->outChannels * batchSize; 
			conv->meansOffset		 = conv->outHeight * conv->outWidth * conv->outChannels * batchSize; 
			conv->variancesOffset = conv->meansOffset + conv->outChannels; 
			conv->runningMeansOffset = conv->variancesOffset + conv->outChannels; 
			conv->runningVariancesOffset = conv->runningMeansOffset + conv->outChannels;

			conv->activationOffset = 2 * conv->batchNormOffset + 4 * conv->outChannels; 
			conv->poolOffset			= conv->activationOffset + conv->outHeight * conv->outWidth * conv->outChannels * batchSize; 

      	conv->outputOffset 		= conv->poolOffset; 
		}

   }

   //volumn Terms here. 
   conv->convVolumn 				= conv->outHeight * conv->outWidth * conv->outChannels;

	if (conv->batchNorm != PERFORM_NO_BATCH_NORM)
		conv->batchNormVolumn = INT_MAX;
	else
		conv->batchNormVolumn		= conv->convVolumn; 

   conv->activationVolumn 		= conv->convVolumn;

	if (poolFun == NO_POOL){
		conv->poolVolumn			= INT_MAX; 
	} else {
   	conv->poolVolumn 				= pool->outHeight * pool->outWidth * conv->outChannels;
	}
}

void readTestVGG( CNN_MODEL *model, int batchSize, int height, int width, int numClasses, int inChannels, int bias, BATCH_NORM_TYPES bn )
{
	int l = 0; 

	int t_h, t_w; 
	CNN_ACTIVATION_FUNCTIONS vggAct = CNN_ACT_SOFTPLUS; 

	fprintf( stderr, "First Layer... \n"); 
	l = 0; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			2, 3, height, width, &t_h, &t_w, batchSize, bn ); 

	fprintf( stderr, "Second Layer... \n"); 
	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			3, 4, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->cLayers = l; 

	l = 0; 
	model->fcLayer[ l ].in = 4 * 2 * 2; 
	model->fcLayer[ l ].out = numClasses; 
	model->fcLayer[ l ].actFun = CNN_ACT_NONE; 
	model->fcLayer[ l ].offset = batchSize * numClasses; 

	l ++; 
	model->lLayers = l; 

	//Bias
	model->bias = 1; 

   //BATCH SIZE
   model->batchSize = batchSize; 

   //compute pSize; 
   computeParamSize( model );  

   //compute Weights/Bias offsets here. 
	if (model->bias != 0)
   	computeWeightBiasOffsets( model );  
	else
   	computeWeightOffsets( model );  


   //compute zOffsets here. 
   computeZOffsets( model, height, width, model->batchSize ); 

	//Report Here. 
   fprintf( stderr, "======================\n"); 
   fprintf( stderr, " W and B Offsets... \n"); 
   for (int i = 0; i < model->cLayers; i ++) 
      fprintf( stderr, "%8d\t\t%8d \n", model->wOffsets[ i ], model->bOffsets[ i ] );  
   fprintf( stderr, "\n"); 
   for (int i = 0; i < model->lLayers; i ++) 
      fprintf( stderr, "%8d\t\t%8d \n", model->wOffsets[ i + model->cLayers ], model->bOffsets[ i + model->cLayers ] );  
   fprintf( stderr, "\n"); 
   fprintf( stderr, "\n"); 

   fprintf( stderr, "Z Offsets ... \n" );  
   for (int i = 0; i <= model->cLayers + model->lLayers + 1; i ++) 
      fprintf( stderr, "%8d \n", model->zOffsets[ i ] );  
   fprintf( stderr, "\n"); 
   fprintf( stderr, "MaxDeltaSize: %d \n\n", model->maxDeltaSize );  
   fprintf( stderr, "\n"); 

   fprintf( stderr, "Params size: %d \n", model->pSize );  
   fprintf( stderr, "Z size: %d \n", model->zSize );  
   fprintf( stderr, "======================\n"); 

	fprintf( stderr, " *** MODEL SUMMARY *** \n"); 
	fprintf( stderr, " ********************* \n"); 
	for (int i = 0; i < model->cLayers; i ++) {
		CONV_LAYER c = model->convLayer[ i ]; 
		POOL_LAYER p = model->poolLayer[ i ]; 
		fprintf( stderr, "Layer: %d, Activation: %d\n", i, model->actFuns[ i ] ); 
		fprintf( stderr, "\t\tinC: %d, outC: %d, H: %d, W: %d, OH: %d, OW: %d \n", 
									c.inChannels, c.outChannels, c.height, c.width, c.outHeight, c.outWidth);
		fprintf( stderr, "\t\tPoolFun: %d, inH: %d, inW: %d, outH: %d, outW: %d \n\n\n", 
									p.type, p.height, p.width, p.outHeight, p.outWidth ); 
	}

	for (int i = 0; i < model->lLayers; i ++){
		FC_LAYER f = model->fcLayer[ i ]; 
		fprintf( stderr, "Layer: %d, Activation: %d, in: %d, out: %d \n\n", 
								model->cLayers + i, f.actFun, f.in, f.out ); 
	}
	fprintf( stderr, " ********************* \n\n\n"); 
}


// 'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
void readVGG11( CNN_MODEL *model, int batchSize, int height, int width, int numClasses, int inChannels, int bias, int enableBatchNorm, DATASET_TYPE datasetType )
{
	int l = 0; 

	int t_h, t_w; 
	CNN_ACTIVATION_FUNCTIONS vggAct = CNN_ACT_SWISH; 
	BATCH_NORM_TYPES bn = PERFORM_NO_BATCH_NORM; 

	if (enableBatchNorm != 0) bn = PERFORM_BATCH_NORM; 
	model->enableBatchNorm = enableBatchNorm; 

	l = 0; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			inChannels, 64, height, width, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			64, 128, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			128, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			256, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			256, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->cLayers = l; 

	l = 0; 
	if (datasetType == IMAGENET)
		model->fcLayer[ l ].in = 512 * 4;
	else 
		model->fcLayer[ l ].in = 512 ;
	model->fcLayer[ l ].out = numClasses; 
	model->fcLayer[ l ].actFun = CNN_ACT_NONE; 
	model->fcLayer[ l ].offset = batchSize * numClasses; 

	l ++; 
	model->lLayers = l; 

	//Bias
	model->bias = bias; 
	model->name = CNN_VGG11NET; 

   //BATCH SIZE
   model->batchSize = batchSize; 

   //compute pSize; 
   computeParamSize( model );  

   //compute Weights/Bias offsets here. 
	if (model->bias != 0)
   	computeWeightBiasOffsets( model );  
	else
   	computeWeightOffsets( model );  

   //compute zOffsets here. 
   computeZOffsets( model, height, width, model->batchSize ); 

	//Report Here. 
   fprintf( stderr, "======================\n"); 
   fprintf( stderr, " W and B Offsets... \n"); 
   for (int i = 0; i < model->cLayers; i ++) 
      fprintf( stderr, "%8d\t\t%8d \n", model->wOffsets[ i ], model->bOffsets[ i ] );  
   fprintf( stderr, "\n"); 
   for (int i = 0; i < model->lLayers; i ++) 
      fprintf( stderr, "%8d\t\t%8d \n", model->wOffsets[ i + model->cLayers ], model->bOffsets[ i + model->cLayers ] );  
   fprintf( stderr, "\n"); 
   fprintf( stderr, "\n"); 

   fprintf( stderr, "Z Offsets ... \n" );  
   for (int i = 0; i <= model->cLayers + model->lLayers + 1; i ++) 
      fprintf( stderr, "%8d \n", model->zOffsets[ i ] );  
   fprintf( stderr, "\n"); 
   fprintf( stderr, "MaxDeltaSize: %d \n\n", model->maxDeltaSize );  
   fprintf( stderr, "\n"); 

   fprintf( stderr, "Params size: %d \n", model->pSize );  
   fprintf( stderr, "Z size: %d \n", model->zSize );  
   fprintf( stderr, "ZZT size: %d \n", model->zztSize );  
   fprintf( stderr, "======================\n"); 

	fprintf( stderr, " *** MODEL SUMMARY *** \n"); 
	fprintf( stderr, " ********************* \n"); 
	for (int i = 0; i < model->cLayers; i ++) {
		CONV_LAYER c = model->convLayer[ i ]; 
		POOL_LAYER p = model->poolLayer[ i ]; 
		fprintf( stderr, "Layer: %d, Activation: %d\n", i, model->actFuns[ i ] ); 
		fprintf( stderr, "\t\tinC: %d, outC: %d, H: %d, W: %d, OH: %d, OW: %d \n", 
									c.inChannels, c.outChannels, c.height, c.width, c.outHeight, c.outWidth);
		fprintf( stderr, "\t\tPoolFun: %d, inH: %d, inW: %d, outH: %d, outW: %d \n\n\n", 
									p.type, p.height, p.width, p.outHeight, p.outWidth ); 
	}

	for (int i = 0; i < model->lLayers; i ++){
		FC_LAYER f = model->fcLayer[ i ]; 
		fprintf( stderr, "Layer: %d, Activation: %d, in: %d, out: %d \n\n", 
								model->cLayers + i, f.actFun, f.in, f.out ); 
	}
	fprintf( stderr, " ********************* \n\n\n"); 
}

//    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
void readVGG13( CNN_MODEL *model, int batchSize, int height, int width, int numClasses, int inChannels, int bias, int enableBatchNorm, DATASET_TYPE datasetType)
{
	int l = 0; 

	int t_h, t_w; 
	CNN_ACTIVATION_FUNCTIONS vggAct = CNN_ACT_SWISH;
	BATCH_NORM_TYPES bn = PERFORM_NO_BATCH_NORM; 

	if (enableBatchNorm != 0) bn = PERFORM_BATCH_NORM; 
	model->enableBatchNorm = enableBatchNorm; 

	l = 0; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			inChannels, 64, height, width, &t_h, &t_w, batchSize, bn ); 

	fprintf( stderr, "OutHeight: %d, OutWidth: %d \n", t_h, t_w ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			64, 64, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			64, 128, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			128, 128, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			128, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			256, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			256, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 


	l ++; 
	model->cLayers = l; 

	l = 0; 
	if (datasetType == IMAGENET)
		model->fcLayer[ l ].in = 512 * 4;
	else 
		model->fcLayer[ l ].in = 512 ;
	model->fcLayer[ l ].out = numClasses; 
	model->fcLayer[ l ].actFun = CNN_ACT_NONE; 
	model->fcLayer[ l ].offset = batchSize * numClasses; 

	l ++; 
	model->lLayers = l; 

	//Bias
	model->bias = bias; 
	model->name = CNN_VGG13NET; 

   //BATCH SIZE
   model->batchSize = batchSize; 

   //compute pSize; 
   computeParamSize( model );  

   //compute Weights/Bias offsets here. 
	if (model->bias != 0)
   	computeWeightBiasOffsets( model );  
	else
   	computeWeightOffsets( model );  

   //compute zOffsets here. 
   computeZOffsets( model, height, width, model->batchSize ); 

	//Report Here. 
   fprintf( stderr, "======================\n"); 
   fprintf( stderr, " W and B Offsets... \n"); 
   for (int i = 0; i < model->cLayers; i ++) 
      fprintf( stderr, "%8d\t\t%8d \n", model->wOffsets[ i ], model->bOffsets[ i ] );  
   fprintf( stderr, "\n"); 
   for (int i = 0; i < model->lLayers; i ++) 
      fprintf( stderr, "%8d\t\t%8d \n", model->wOffsets[ i + model->cLayers ], model->bOffsets[ i + model->cLayers ] );  
   fprintf( stderr, "\n"); 
   fprintf( stderr, "\n"); 

   fprintf( stderr, "Z Offsets ... \n" );  
   for (int i = 0; i <= model->cLayers + model->lLayers + 1; i ++) 
      fprintf( stderr, "%8d \n", model->zOffsets[ i ] );  
   fprintf( stderr, "\n"); 
   fprintf( stderr, "MaxDeltaSize: %d \n\n", model->maxDeltaSize );  
   fprintf( stderr, "\n"); 

   fprintf( stderr, "Params size: %d \n", model->pSize );  
   fprintf( stderr, "Z size: %d \n", model->zSize );  
   fprintf( stderr, "======================\n"); 

	fprintf( stderr, " *** MODEL SUMMARY *** \n"); 
	fprintf( stderr, " ********************* \n"); 
	for (int i = 0; i < model->cLayers; i ++) {
		CONV_LAYER c = model->convLayer[ i ]; 
		POOL_LAYER p = model->poolLayer[ i ]; 
		fprintf( stderr, "Layer: %d, Activation: %d\n", i, model->actFuns[ i ] ); 
		fprintf( stderr, "\t\tinC: %d, outC: %d, H: %d, W: %d, OH: %d, OW: %d \n", 
									c.inChannels, c.outChannels, c.height, c.width, c.outHeight, c.outWidth);
		fprintf( stderr, "\t\tPoolFun: %d, inH: %d, inW: %d, outH: %d, outW: %d \n\n\n", 
									p.type, p.height, p.width, p.outHeight, p.outWidth ); 
	}

	for (int i = 0; i < model->lLayers; i ++){
		FC_LAYER f = model->fcLayer[ i ]; 
		fprintf( stderr, "Layer: %d, Activation: %d, in: %d, out: %d \n\n", 
								model->cLayers + i, f.actFun, f.in, f.out ); 
	}
	fprintf( stderr, " ********************* \n\n\n"); 
}

//    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
void readVGG16( CNN_MODEL *model, int batchSize, int height, int width, int numClasses, int inChannels, int bias, int enableBatchNorm, DATASET_TYPE datasetType )
{
	int l = 0; 

	int t_h, t_w; 
	CNN_ACTIVATION_FUNCTIONS vggAct = CNN_ACT_SWISH;
	BATCH_NORM_TYPES bn = PERFORM_NO_BATCH_NORM; 

	if (enableBatchNorm != 0) bn = PERFORM_BATCH_NORM; 
	model->enableBatchNorm = enableBatchNorm; 

	l = 0; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			inChannels, 64, height, width, &t_h, &t_w, batchSize, bn ); 

	fprintf( stderr, "OutHeight: %d, OutWidth: %d \n", t_h, t_w ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			64, 64, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			64, 128, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			128, 128, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			128, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			256, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			256, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			256, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->cLayers = l; 

	l = 0; 
	if (datasetType == IMAGENET)
		model->fcLayer[ l ].in = 512 * 4;
	else 
		model->fcLayer[ l ].in = 512 ;
	model->fcLayer[ l ].out = numClasses; 
	model->fcLayer[ l ].actFun = CNN_ACT_NONE; 
	model->fcLayer[ l ].offset = batchSize * numClasses; 

	l ++; 
	model->lLayers = l; 

	//Bias
	model->bias = bias; 
	model->name = CNN_VGG16NET; 

   //BATCH SIZE
   model->batchSize = batchSize; 

   //compute pSize; 
   computeParamSize( model );  

   //compute Weights/Bias offsets here. 
	if (model->bias != 0)
   	computeWeightBiasOffsets( model );  
	else
   	computeWeightOffsets( model );  

   //compute zOffsets here. 
   computeZOffsets( model, height, width, model->batchSize ); 

	//Report Here. 
   fprintf( stderr, "======================\n"); 
   fprintf( stderr, " W and B Offsets... \n"); 
   for (int i = 0; i < model->cLayers; i ++) 
      fprintf( stderr, "%8d\t\t%8d \n", model->wOffsets[ i ], model->bOffsets[ i ] );  
   fprintf( stderr, "\n"); 
   for (int i = 0; i < model->lLayers; i ++) 
      fprintf( stderr, "%8d\t\t%8d \n", model->wOffsets[ i + model->cLayers ], model->bOffsets[ i + model->cLayers ] );  
   fprintf( stderr, "\n"); 
   fprintf( stderr, "\n"); 

   fprintf( stderr, "Z Offsets ... \n" );  
   for (int i = 0; i <= model->cLayers + model->lLayers + 1; i ++) 
      fprintf( stderr, "%8d \n", model->zOffsets[ i ] );  
   fprintf( stderr, "\n"); 
   fprintf( stderr, "MaxDeltaSize: %d \n\n", model->maxDeltaSize );  
   fprintf( stderr, "\n"); 

   fprintf( stderr, "Params size: %d \n", model->pSize );  
   fprintf( stderr, "Z size: %d \n", model->zSize );  
   fprintf( stderr, "======================\n"); 

	fprintf( stderr, " *** MODEL SUMMARY *** \n"); 
	fprintf( stderr, " ********************* \n"); 
	for (int i = 0; i < model->cLayers; i ++) {
		CONV_LAYER c = model->convLayer[ i ]; 
		POOL_LAYER p = model->poolLayer[ i ]; 
		fprintf( stderr, "Layer: %d, Activation: %d\n", i, model->actFuns[ i ] ); 
		fprintf( stderr, "\t\tinC: %d, outC: %d, H: %d, W: %d, OH: %d, OW: %d \n", 
									c.inChannels, c.outChannels, c.height, c.width, c.outHeight, c.outWidth);
		fprintf( stderr, "\t\tPoolFun: %d, inH: %d, inW: %d, outH: %d, outW: %d \n\n\n", 
									p.type, p.height, p.width, p.outHeight, p.outWidth ); 
	}

	for (int i = 0; i < model->lLayers; i ++){
		FC_LAYER f = model->fcLayer[ i ]; 
		fprintf( stderr, "Layer: %d, Activation: %d, in: %d, out: %d \n\n", 
								model->cLayers + i, f.actFun, f.in, f.out ); 
	}
	fprintf( stderr, " ********************* \n\n\n"); 
}


//    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
void readVGG19( CNN_MODEL *model, int batchSize, int height, int width, int numClasses, int inChannels, int bias, int enableBatchNorm, DATASET_TYPE datasetType )
{
	int l = 0; 

	int t_h, t_w; 
	CNN_ACTIVATION_FUNCTIONS vggAct = CNN_ACT_SWISH;
	BATCH_NORM_TYPES bn = PERFORM_NO_BATCH_NORM; 

	if (enableBatchNorm != 0) bn = PERFORM_BATCH_NORM; 
	model->enableBatchNorm = enableBatchNorm; 

	l = 0; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			inChannels, 64, height, width, &t_h, &t_w, batchSize, bn ); 

	fprintf( stderr, "OutHeight: %d, OutWidth: %d \n", t_h, t_w ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			64, 64, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			64, 128, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			128, 128, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			128, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			256, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			256, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			256, 256, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			256, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], NO_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->actFuns[ l ] = vggAct; 
	makeLayer( &model->convLayer[ l ], &model->poolLayer[ l ], MAX_POOL, 
			512, 512, t_h, t_w, &t_h, &t_w, batchSize, bn ); 

	l ++; 
	model->cLayers = l; 

	l = 0; 
	if (datasetType == IMAGENET)
		model->fcLayer[ l ].in = 512 * 4;
	else 
		model->fcLayer[ l ].in = 512 ;
	model->fcLayer[ l ].out = numClasses; 
	model->fcLayer[ l ].actFun = CNN_ACT_NONE; 
	model->fcLayer[ l ].offset = batchSize * numClasses; 

	l ++; 
	model->lLayers = l; 

	//Bias
	model->bias = bias;
	model->name = CNN_VGG19NET; 

   //BATCH SIZE
   model->batchSize = batchSize; 

   //compute pSize; 
   computeParamSize( model );  

   //compute Weights/Bias offsets here. 
	if (model->bias != 0)
   	computeWeightBiasOffsets( model );  
	else
   	computeWeightOffsets( model );  

   //compute zOffsets here. 
   computeZOffsets( model, height, width, model->batchSize ); 

	//Report Here. 
   fprintf( stderr, "======================\n"); 
   fprintf( stderr, " W and B Offsets... \n"); 
   for (int i = 0; i < model->cLayers; i ++) 
      fprintf( stderr, "%8d\t\t%8d \n", model->wOffsets[ i ], model->bOffsets[ i ] );  
   fprintf( stderr, "\n"); 
   for (int i = 0; i < model->lLayers; i ++) 
      fprintf( stderr, "%8d\t\t%8d \n", model->wOffsets[ i + model->cLayers ], model->bOffsets[ i + model->cLayers ] );  
   fprintf( stderr, "\n"); 
   fprintf( stderr, "\n"); 

   fprintf( stderr, "Z Offsets ... \n" );  
   for (int i = 0; i <= model->cLayers + model->lLayers + 1; i ++) 
      fprintf( stderr, "%8d \n", model->zOffsets[ i ] );  
   fprintf( stderr, "\n"); 
   fprintf( stderr, "MaxDeltaSize: %d \n\n", model->maxDeltaSize );  
   fprintf( stderr, "\n"); 

   fprintf( stderr, "Params size: %d \n", model->pSize );  
   fprintf( stderr, "Z size: %d \n", model->zSize );  
   fprintf( stderr, "======================\n"); 

	fprintf( stderr, " *** MODEL SUMMARY *** \n"); 
	fprintf( stderr, " ********************* \n"); 
	for (int i = 0; i < model->cLayers; i ++) {
		CONV_LAYER c = model->convLayer[ i ]; 
		POOL_LAYER p = model->poolLayer[ i ]; 
		fprintf( stderr, "Layer: %d, Activation: %d\n", i, model->actFuns[ i ] ); 
		fprintf( stderr, "\t\tinC: %d, outC: %d, H: %d, W: %d, OH: %d, OW: %d \n", 
									c.inChannels, c.outChannels, c.height, c.width, c.outHeight, c.outWidth);
		fprintf( stderr, "\t\tPoolFun: %d, inH: %d, inW: %d, outH: %d, outW: %d \n\n\n", 
									p.type, p.height, p.width, p.outHeight, p.outWidth ); 
	}

	for (int i = 0; i < model->lLayers; i ++){
		FC_LAYER f = model->fcLayer[ i ]; 
		fprintf( stderr, "Layer: %d, Activation: %d, in: %d, out: %d \n\n", 
								model->cLayers + i, f.actFun, f.in, f.out ); 
	}
	fprintf( stderr, " ********************* \n\n\n"); 
}
