#include <nn/read_alexnet.h>
#include <nn/utils.h>
#include <nn/read_nn.h>
#include <nn/nn_decl.h>

#include <core/errors.h>
#include <core/structdefs.h>
#include <device/cuda_utils.h>


#include <stdlib.h>
#include <stdio.h>

void readAlexNetCNN( CNN_MODEL *model, int batchSize, int height, int width, 
		int numClasses, int enableBias, int enableBatchNorm ) {

	int l = 0; 
	int h, w; 
	int t_h, t_w;
	BATCH_NORM_TYPES bnType; 
	fprintf( stderr, "readAlexNet ... begin\n" ); 

	model->bias = enableBias;
	model->enableBatchNorm = enableBatchNorm ; 

	if (enableBatchNorm != 0)
		bnType = PERFORM_BATCH_NORM; 
	else 
		bnType = PERFORM_NO_BATCH_NORM; 

	

	//init the model here. 
	//Conv2d( 3, 64, 5, stride=1, padding=2)
	model->convLayer[ l ].inChannels = 3; 
	model->convLayer[ l ].outChannels = 64; 
	model->convLayer[ l ].kSize= 5; 
	model->convLayer[ l ].height= height; 
	model->convLayer[ l ].width= width; 
	model->convLayer[ l ].stride = 1; 
	model->convLayer[ l ].padding = 2; //New

	model->convLayer[ l ].batchNorm = bnType;
	/*
	model->convLayer[ l ].running_mean = 0; 
	model->convLayer[ l ].running_variance = 0; 
	model->convLayer[ l ].batch_mean = 0; 
	model->convLayer[ l ].batch_variance = 0; 
	*/

	model->actFuns[ l ] = CNN_ACT_SWISH;

	getDimensions( height, width, model->convLayer[ l ].padding, 
			model->convLayer[ l ].stride, model->convLayer[ l ].kSize, &h, &w ); 

	model->convLayer[ l ].outHeight = h; 
	model->convLayer[ l ].outWidth = w; 

   model->convLayer[ l ].activationOffset = h * w * model->convLayer[ l ].outChannels * batchSize;  // Img2Cols Weights

	model->poolLayer[ l ].type = MAX_POOL;  //CNN_MAX_POOL -- NEW
	model->poolLayer[ l ].pSize= 3; 
	model->poolLayer[ l ].stride = 2;	//New
	model->poolLayer[ l ].padding = 1;	//New
	model->poolLayer[ l ].height = h; 
	model->poolLayer[ l ].width = w; 

   getDimensions( h, w, model->poolLayer[ l ].padding,
                  model->poolLayer[ l ].stride, model->poolLayer[ l ].pSize, &t_h, &t_w );
	model->poolLayer[ l ].outHeight = t_h; 
	model->poolLayer[ l ].outWidth = t_w; 

	model->convLayer[ l ].poolOffset = model->convLayer[ l ].activationOffset + 
		model->poolLayer[ l ].height * model->poolLayer[ l ].width  * model->convLayer[ l ].outChannels * batchSize;  // Img2Cols Weights

   if (model->convLayer[ l ].batchNorm == PERFORM_NO_BATCH_NORM){

      model->convLayer[ l ].outputOffset = model->convLayer[ l ].poolOffset;
		model->convLayer[ l ].meansOffset = model->convLayer[ l ].variancesOffset = 0; 
   } else {
      model->convLayer[ l ].batchNormOffset =
         model->convLayer[ l ].poolOffset + t_h * t_w * model->convLayer[ l ].outChannels * batchSize;

      model->convLayer[ l ].outputOffset = model->convLayer[ l ].batchNormOffset;

      model->convLayer[ l ].meansOffset = t_h * t_w * model->convLayer[ l ].outChannels * batchSize; 
      model->convLayer[ l ].variancesOffset = model->convLayer[ l ].meansOffset + 
                                             model->convLayer[ l ].outChannels; 

      model->convLayer[ l ].runningMeansOffset = model->convLayer[ l ].variancesOffset + 
                                             model->convLayer[ l ].outChannels; 
      model->convLayer[ l ].runningVariancesOffset = model->convLayer[ l ].runningMeansOffset + 
                                             model->convLayer[ l ].outChannels; 
   }

   //volumn Terms here. 
   model->convLayer[ l ].convVolumn = model->convLayer[ l ].outHeight * model->convLayer[ l ].outWidth * 
														model->convLayer[ l ].outChannels; 
   model->convLayer[ l ].activationVolumn = model->convLayer[ l ].convVolumn; 
   model->convLayer[ l ].poolVolumn = model->poolLayer[ l ].outHeight * model->poolLayer[ l ].outWidth * 
														model->convLayer[ l ].outChannels; 
   model->convLayer[ l ].batchNormVolumn = model->convLayer[ l ].poolVolumn;



	//Conv2d( 64, 64, 5, stride=1, padding=1 )
	l ++; 
	model->convLayer[ l ].inChannels = 64; 
	model->convLayer[ l ].kSize = 5; 
	model->convLayer[ l ].outChannels = 64; 
	model->convLayer[ l ].height= t_h; 
	model->convLayer[ l ].width= t_w; 
	model->convLayer[ l ].stride = 1; 
	model->convLayer[ l ].padding = 2; 

	model->convLayer[ l ].batchNorm = bnType;
	/*
	model->convLayer[ l ].running_mean = 0; 
	model->convLayer[ l ].running_variance = 0; 
	model->convLayer[ l ].batch_mean = 0; 
	model->convLayer[ l ].batch_variance = 0; 
	*/

	model->actFuns[ l ] = CNN_ACT_SWISH; 		//CNN-SWISH

	getDimensions( t_h, t_w, model->convLayer[ l ].padding, 
			model->convLayer[ l ].stride, model->convLayer[ l ].kSize, &h, &w ); 
	model->convLayer[ l ].outHeight = h; 
	model->convLayer[ l ].outWidth = w; 

   model->convLayer[ l ].activationOffset = h * w * model->convLayer[ l ].outChannels * batchSize;  // Img2Cols Weights

	model->poolLayer[ l ].type = MAX_POOL;			//CNN_MAX_POOL 
	model->poolLayer[ l ].height= h; 
	model->poolLayer[ l ].width= w; 
	model->poolLayer[ l ].pSize = 3; 
	model->poolLayer[ l ].stride = 2; 
	model->poolLayer[ l ].padding = 1; 

   getDimensions( h, w, model->poolLayer[ l ].padding,
                  model->poolLayer[ l ].stride, model->poolLayer[ l ].pSize, &t_h, &t_w );
	model->poolLayer[ l ].outHeight = t_h; 
	model->poolLayer[ l ].outWidth = t_w; 

	model->convLayer[ l ].poolOffset = model->convLayer[ l ].activationOffset + 
		model->poolLayer[ l ].height * model->poolLayer[ l ].width  * model->convLayer[ l ].outChannels * batchSize;  // Img2Cols Weights

   if (model->convLayer[ l ].batchNorm == PERFORM_NO_BATCH_NORM){

      model->convLayer[ l ].outputOffset = model->convLayer[ l ].poolOffset;
      model->convLayer[ l ].meansOffset = model->convLayer[ l ].variancesOffset = 0;
   } else {
      model->convLayer[ l ].batchNormOffset =
         model->convLayer[ l ].poolOffset + t_h * t_w * model->convLayer[ l ].outChannels * batchSize;

      model->convLayer[ l ].outputOffset = model->convLayer[ l ].batchNormOffset;

      model->convLayer[ l ].meansOffset = t_h * t_w * model->convLayer[ l ].outChannels * batchSize; 
      model->convLayer[ l ].variancesOffset = model->convLayer[ l ].meansOffset + 
                                             model->convLayer[ l ].outChannels; 

      model->convLayer[ l ].runningMeansOffset = model->convLayer[ l ].variancesOffset + 
                                             model->convLayer[ l ].outChannels; 
      model->convLayer[ l ].runningVariancesOffset = model->convLayer[ l ].runningMeansOffset + 
                                             model->convLayer[ l ].outChannels; 
   }

   //volumn Terms here. 
   model->convLayer[ l ].convVolumn = model->convLayer[ l ].outHeight * model->convLayer[ l ].outWidth * 
														model->convLayer[ l ].outChannels; 
   model->convLayer[ l ].activationVolumn = model->convLayer[ l ].convVolumn; 
   model->convLayer[ l ].poolVolumn = model->poolLayer[ l ].outHeight * model->poolLayer[ l ].outWidth * 
														model->convLayer[ l ].outChannels; 
   model->convLayer[ l ].batchNormVolumn = model->convLayer[ l ].poolVolumn;


	// increment the layers. 
	l++;
	model->cLayers = l;

	//Fully Connected Layers here. 
	//nn.Linear(64 * 8 * 8, 384)

	l = 0; 
	model->fcLayer[ l ].in = 64 * 8 * 8; 
	model->fcLayer[ l ].out = 384; 
	model->fcLayer[ l ].actFun = CNN_ACT_SWISH; 
	model->fcLayer[ l ].offset = (model->fcLayer[ l ].out * batchSize );

   //nn.Linear(120, 84)
	l ++; 
	model->fcLayer[ l ].in = 384;
	model->fcLayer[ l ].out = 192; 
	model->fcLayer[ l ].actFun = CNN_ACT_SWISH; 
	model->fcLayer[ l ].offset = (model->fcLayer[ l ].out * batchSize );

   //nn.Linear(84, 10)
	l ++; 
	model->fcLayer[ l ].in = 192;
	model->fcLayer[ l ].out = numClasses; 
	model->fcLayer[ l ].actFun = CNN_ACT_NONE; 
	model->fcLayer[ l ].offset = (model->fcLayer[ l ].out * batchSize );

	l ++; 
	model->lLayers = l; 

	//BATCH SIZE
	model->batchSize = batchSize; 
	model->name = CNN_ALEXNET; 

	//compute pSize; 
	computeParamSize( model ); 

	//compute Weights/Bias offsets here. 
	if( model->bias != 0 )
		computeWeightBiasOffsets( model ); 
	else
		computeWeightOffsets( model ); 

	//compute zOffsets here. 
	computeZOffsets( model, height, width, model->batchSize ); 

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
}
