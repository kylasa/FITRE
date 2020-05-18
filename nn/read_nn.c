#include <nn/read_nn.h>
#include <nn/nn_decl.h>

#include <core/errors.h>
#include <core/structdefs.h>
#include <device/cuda_utils.h>

#include <nn/utils.h>


#include <stdlib.h>
#include <stdio.h>

void readFCCNN( CNN_MODEL *model, int batchSize ) {

	int l = 0; 
	int h, w; 
	fprintf( stderr, "readFC ... begin\n" ); 

   l = 0;
   model->fcLayer[ l ].in = 8;
   model->fcLayer[ l ].out = 6; 
   model->fcLayer[ l ].actFun = CNN_ACT_SOFTPLUS;
	model->fcLayer[ l ].offset = (6 * batchSize);  //TODO temp storage

   l ++;
   model->fcLayer[ l ].in = 6;
   model->fcLayer[ l ].out = 4; 
   model->fcLayer[ l ].actFun = CNN_ACT_SOFTPLUS;
	model->fcLayer[ l ].offset = (4 * batchSize);  //TODO temp storage

   l ++;
   model->lLayers = l;
	model->cLayers = 0; 


   //BATCH SIZE
   model->batchSize = batchSize;
	model->bias = 1; 

	computeParamSize( model ); 
	computeWeightBiasOffsets( model ); 
	computeZOffsets( model, 0, 0, model->batchSize ); 

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
   fprintf( stderr, "\n");
	fprintf( stderr, "MaxDeltaSize: %d \n\n", model->maxDeltaSize ); 

   fprintf( stderr, "Params size: %d \n", model->pSize );
   fprintf( stderr, "Z size: %d \n", model->zSize );
   fprintf( stderr, "======================\n");

}

void readConv2CNN( CNN_MODEL *model, int in_channels, int out_channels, int numclasses, 
							int width, int height, int batchSize ) {

	int l = 0; 
	int h, w; 
	int t_h, t_w;
	fprintf( stderr, "readConvolution... begin\n" ); 

   model->convLayer[ l ].inChannels =in_channels; 
   model->convLayer[ l ].outChannels = out_channels;  
   model->convLayer[ l ].kSize= 5; 
   model->convLayer[ l ].height= height; 
   model->convLayer[ l ].width= width; 
   model->convLayer[ l ].stride = 1;  
   model->convLayer[ l ].padding = 2;  

	model->convLayer[ l ].batchNorm = PERFORM_BATCH_NORM; 
	/*
   model->convLayer[ l ].running_mean = 0;  
   model->convLayer[ l ].running_variance = 0;  
   model->convLayer[ l ].batch_mean = 0;  
   model->convLayer[ l ].batch_variance = 0; 
	*/

   model->actFuns[ l ] = CNN_ACT_SOFTPLUS; 

   getDimensions( height, width, model->convLayer[ l ].padding, 
         model->convLayer[ l ].stride, model->convLayer[ l ].kSize, &h, &w );  
   model->convLayer[ l ].outHeight = h; 
   model->convLayer[ l ].outWidth = w; 

   model->poolLayer[ l ].type = MAX_POOL; 
   model->poolLayer[ l ].pSize= 3;  
   model->poolLayer[ l ].height = h;
   model->poolLayer[ l ].width = w;
	model->poolLayer[ l ].stride = 2; 
	model->poolLayer[ l ].padding = 1; 

	// Img2Cols Weights
	model->convLayer[ l ].activationOffset = h * w * model->convLayer[ l ].outChannels * batchSize;  
	model->convLayer[ l ].poolOffset = 
		model->convLayer[ l ].activationOffset +  h * w * model->convLayer[ l ].outChannels * batchSize;  

	getDimensions( h, w, model->poolLayer[ l ].padding, 
						model->poolLayer[ l ].stride, model->poolLayer[ l ].pSize, &t_h, &t_w ); 
	model->poolLayer[ l ].outHeight = t_h; 
	model->poolLayer[ l ].outWidth = t_w; 

	if (model->convLayer[ l ].batchNorm != PERFORM_BATCH_NORM){

		model->convLayer[ l ].outputOffset = 
			model->convLayer[ l ].poolOffset;
		
		model->convLayer [ l ].variancesOffset = model->convLayer[ l ].meansOffset = 0; 
	} else {
		model->convLayer[ l ].batchNormOffset = 
			model->convLayer[ l ].poolOffset + t_h * t_w * model->convLayer[ l ].outChannels * batchSize; 

		model->convLayer[ l ].outputOffset = 
			model->convLayer[ l ].batchNormOffset;
		model->convLayer[ l ].meansOffset = t_h * t_w * model->convLayer[ l ].outChannels * batchSize; 
		model->convLayer[ l ].variancesOffset = model->convLayer[ l ].meansOffset + 
																model->convLayer[ l ].outChannels;

		model->convLayer[ l ].runningMeansOffset = model->convLayer[ l ].variancesOffset + 
															model->convLayer[ l ].outChannels; 
		model->convLayer[ l ].runningVariancesOffset = model->convLayer[ l ].runningMeansOffset + 
															model->convLayer[ l ].outChannels; 
	}
	
	//volumn terms here. 
	model->convLayer[ l ].convVolumn			 = model->convLayer[ l ].outHeight * model->convLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].activationVolumn  = model->convLayer[ l ].convVolumn;
	model->convLayer[ l ].poolVolumn			 = model->poolLayer[ l ].outHeight * model->poolLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].batchNormVolumn	 = model->convLayer[ l ].poolVolumn; 

	getDimensions( h, w, model->poolLayer[ l ].padding, 
						model->poolLayer[ l ].stride, model->poolLayer[ l ].pSize, &h, &w ); 

   // increment the layers. 
	// Layer 2 here
   l++;
   model->convLayer[ l ].inChannels = out_channels; 
   model->convLayer[ l ].outChannels = out_channels;  
   model->convLayer[ l ].kSize= 5;  
   model->convLayer[ l ].height= h; 
   model->convLayer[ l ].width= w; 
   model->convLayer[ l ].stride = 1;  
   model->convLayer[ l ].padding = 2;  

	model->convLayer[ l ].batchNorm = PERFORM_BATCH_NORM; 
	/*
   model->convLayer[ l ].running_mean = 0;  
   model->convLayer[ l ].running_variance = 0;  
   model->convLayer[ l ].batch_mean = 0;  
   model->convLayer[ l ].batch_variance = 0; 
	*/

   model->actFuns[ l ] = CNN_ACT_SOFTPLUS; 

   getDimensions( h, w, model->convLayer[ l ].padding, 
         model->convLayer[ l ].stride, model->convLayer[ l ].kSize, &h, &w );  

   model->convLayer[ l ].outHeight = h; 
   model->convLayer[ l ].outWidth = w; 

	model->convLayer[ l ].activationOffset = 
		h * w * model->convLayer [l].outChannels * batchSize; 
	model->convLayer[ l ].poolOffset = model->convLayer[ l ].activationOffset +  
		h * w * model->convLayer[ l ].outChannels * batchSize;  

   model->poolLayer[ l ].type = MAX_POOL; 
   model->poolLayer[ l ].pSize= 3;  
   model->poolLayer[ l ].height = h;
   model->poolLayer[ l ].width = w;
	model->poolLayer[ l ].stride = 2; 
	model->poolLayer[ l ].padding = 1; 

   getDimensions( h, w, model->poolLayer[ l ].padding, 
                  model->poolLayer[ l ].stride, model->poolLayer[ l ].pSize, &t_h, &t_w );  
   model->poolLayer[ l ].outHeight = t_h; 
   model->poolLayer[ l ].outWidth = t_w; 

   if (model->convLayer[ l ].batchNorm != PERFORM_BATCH_NORM){

      model->convLayer[ l ].outputOffset = 
         model->convLayer[ l ].poolOffset;

		model->convLayer [ l ].variancesOffset = model->convLayer[ l ].meansOffset = 0; 
   } else {
      model->convLayer[ l ].batchNormOffset = 
         model->convLayer[ l ].poolOffset + t_h * t_w * model->convLayer[ l ].outChannels * batchSize; 

      model->convLayer[ l ].outputOffset = 
         model->convLayer[ l ].batchNormOffset;

		model->convLayer[ l ].meansOffset = t_h * t_w * model->convLayer[ l ].outChannels * batchSize; 
		model->convLayer[ l ].variancesOffset = model->convLayer[ l ].meansOffset + 
																model->convLayer[ l ].outChannels;

		model->convLayer[ l ].runningMeansOffset = model->convLayer[ l ].variancesOffset + 
															model->convLayer[ l ].outChannels; 
		model->convLayer[ l ].runningVariancesOffset = model->convLayer[ l ].runningMeansOffset + 
															model->convLayer[ l ].outChannels; 
   }  

	//volumn terms here. 
	model->convLayer[ l ].convVolumn			 = model->convLayer[ l ].outHeight * model->convLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].activationVolumn  = model->convLayer[ l ].convVolumn;
	model->convLayer[ l ].poolVolumn			 = model->poolLayer[ l ].outHeight * model->poolLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].batchNormVolumn	 = model->convLayer[ l ].poolVolumn; 


	// increment final time here
	l ++;

   model->cLayers = l;
   model->lLayers = 0;

	//bias
	model->bias = 1; 

   //BATCH SIZE
   model->batchSize = batchSize;

	computeParamSize( model ); 
	if (model->bias >= 1)
		computeWeightBiasOffsets( model ); 
	else
		computeWeightOffsets( model ); 

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
   fprintf( stderr, "\n");
	fprintf( stderr, "MaxDeltaSize: %d \n\n", model->maxDeltaSize ); 

   fprintf( stderr, "Params size: %d \n", model->pSize );
   fprintf( stderr, "Z size: %d \n", model->zSize );
   fprintf( stderr, "======================\n");

}

void readConvCNN( CNN_MODEL *model, int in_channels, int out_channels, int numclasses, 
							int width, int height, int batchSize ) {

	int l = 0; 
	int h, w; 
	int t_h, t_w;
	fprintf( stderr, "readConvolution... begin\n" ); 

   model->convLayer[ l ].inChannels = in_channels; 
   model->convLayer[ l ].outChannels = out_channels;  
   model->convLayer[ l ].kSize= 3;  
   model->convLayer[ l ].height= height; 
   model->convLayer[ l ].width= width; 
   model->convLayer[ l ].stride = 1;  
   model->convLayer[ l ].padding = 0; 

	model->convLayer[ l ].batchNorm = PERFORM_BATCH_NORM; 
	/*
   model->convLayer[ l ].running_mean = 0;  
   model->convLayer[ l ].running_variance = 0;  
   model->convLayer[ l ].batch_mean = 0;  
   model->convLayer[ l ].batch_variance = 0; 
	*/

   //model->actFuns[ l ] = CNN_ACT_SWISH;
   model->actFuns[ l ] = CNN_ACT_SOFTPLUS;

   getDimensions( height, width, model->convLayer[ l ].padding, 
         model->convLayer[ l ].stride, model->convLayer[ l ].kSize, &h, &w );  
   model->convLayer[ l ].outHeight = h; 
   model->convLayer[ l ].outWidth = w; 

	model->convLayer[ l ].activationOffset = h * w * out_channels * batchSize;  // Img2Cols Weights

   model->poolLayer[ l ].type = MAX_POOL; 
   model->poolLayer[ l ].pSize= 2;  
   model->poolLayer[ l ].stride = 2; //2; //model->poolLayer[ l ].pSize;  
   model->poolLayer[ l ].padding = 0;  
   model->poolLayer[ l ].height = h;
   model->poolLayer[ l ].width = w;

   getDimensions( h, w, model->poolLayer[ l ].padding, 
                  model->poolLayer[ l ].stride, model->poolLayer[ l ].pSize, &t_h, &t_w );  
   model->poolLayer[ l ].outHeight = t_h; 
   model->poolLayer[ l ].outWidth = t_w; 

	model->convLayer[ l ].poolOffset = model->convLayer[ l ].activationOffset + 
		model->poolLayer[ l ].height * model->poolLayer[ l ].width  * out_channels * batchSize;  // Img2Cols Weights

   if (model->convLayer[ l ].batchNorm == PERFORM_NO_BATCH_NORM){

      model->convLayer[ l ].outputOffset = 
         model->convLayer[ l ].poolOffset;

		model->convLayer[ l ].meansOffset = model->convLayer[ l ].variancesOffset = 0; 
   } else {
      model->convLayer[ l ].batchNormOffset = 
         model->convLayer[ l ].poolOffset + t_h * t_w * out_channels * batchSize; 

      model->convLayer[ l ].outputOffset = 
         model->convLayer[ l ].batchNormOffset;

		model->convLayer[ l ].meansOffset = t_h * t_w * out_channels * batchSize; 
		model->convLayer[ l ].variancesOffset = model->convLayer[ l ].meansOffset + 
																out_channels;

		model->convLayer[ l ].runningMeansOffset = model->convLayer[ l ].variancesOffset + 
															model->convLayer[ l ].outChannels; 
		model->convLayer[ l ].runningVariancesOffset = model->convLayer[ l ].runningMeansOffset + 
															model->convLayer[ l ].outChannels; 
   }  

	//volumn terms here. 
	model->convLayer[ l ].convVolumn			 = model->convLayer[ l ].outHeight * model->convLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].activationVolumn  = model->convLayer[ l ].convVolumn;
	model->convLayer[ l ].poolVolumn			 = model->poolLayer[ l ].outHeight * model->poolLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].batchNormVolumn	 = model->convLayer[ l ].poolVolumn; 

   // increment the layers. 
   l++;
   model->cLayers = l;

   model->lLayers = 0;

   //BATCH SIZE
   model->batchSize = batchSize;

	//bias
	model->bias = 1; 

	computeParamSize( model ); 
	computeWeightBiasOffsets( model ); 
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
   fprintf( stderr, "\n");
	fprintf( stderr, "MaxDeltaSize: %d \n\n", model->maxDeltaSize ); 

   fprintf( stderr, "Params size: %d \n", model->pSize );
   fprintf( stderr, "Z size: %d \n", model->zSize );
   fprintf( stderr, "======================\n");

}

void readTestCNN( CNN_MODEL *model, int in_channels, int out_channels, int numclasses, 
							int width, int height, int batchSize ) {

	int l = 0; 
	int h, w, t_h, t_w; 
	fprintf( stderr, "readConvolution... begin\n" ); 

   model->convLayer[ l ].inChannels = in_channels; 
   model->convLayer[ l ].outChannels = out_channels;  
   model->convLayer[ l ].kSize= 3;  
   model->convLayer[ l ].height= height; 
   model->convLayer[ l ].width= width; 
   model->convLayer[ l ].stride = 1;  
   model->convLayer[ l ].padding = 1;  

	model->convLayer[ l ].batchNorm = PERFORM_NO_BATCH_NORM; 
	/*
   model->convLayer[ l ].running_mean = 0;  
   model->convLayer[ l ].running_variance = 0;  
   model->convLayer[ l ].batch_mean = 0;  
   model->convLayer[ l ].batch_variance = 0; 
	*/

   model->actFuns[ l ] = CNN_ACT_SWISH;
   //model->actFuns[ l ] = CNN_ACT_SOFTPLUS;

   getDimensions( height, width, model->convLayer[ l ].padding, 
         model->convLayer[ l ].stride, model->convLayer[ l ].kSize, &h, &w );  
   model->convLayer[ l ].outHeight = h; 
   model->convLayer[ l ].outWidth = w; 

	model->convLayer[ l ].activationOffset = h * w * out_channels * batchSize;  // Img2Cols Weights

   model->poolLayer[ l ].type = MAX_POOL; 
   model->poolLayer[ l ].pSize= 2;  
   model->poolLayer[ l ].stride = model->poolLayer[ l ].pSize;  
   model->poolLayer[ l ].padding = 0;  
   model->poolLayer[ l ].height = h;
   model->poolLayer[ l ].width = w;

   getDimensions( h, w, model->poolLayer[ l ].padding, 
                  model->poolLayer[ l ].stride, model->poolLayer[ l ].pSize, &t_h, &t_w );  
   model->poolLayer[ l ].outHeight = t_h; 
   model->poolLayer[ l ].outWidth = t_w; 

	model->convLayer[ l ].poolOffset = model->convLayer[ l ].activationOffset + 
		model->poolLayer[ l ].height * model->poolLayer[ l ].width  * out_channels * batchSize;  // Img2Cols Weights

   if (model->convLayer[ l ].batchNorm == PERFORM_NO_BATCH_NORM){

      model->convLayer[ l ].outputOffset = 
         model->convLayer[ l ].poolOffset;

		model->convLayer[ l ].meansOffset = model->convLayer[ l ].variancesOffset = 0; 
   } else {
      model->convLayer[ l ].batchNormOffset = 
         model->convLayer[ l ].poolOffset + t_h * t_w * out_channels * batchSize; 

      model->convLayer[ l ].outputOffset = 
         model->convLayer[ l ].batchNormOffset;

		model->convLayer[ l ].meansOffset = t_h * t_w * out_channels * batchSize; 
		model->convLayer[ l ].variancesOffset = model->convLayer[ l ].meansOffset + 
																out_channels;
   }  

	//volumn terms here. 
	model->convLayer[ l ].convVolumn			 = model->convLayer[ l ].outHeight * model->convLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].activationVolumn  = model->convLayer[ l ].convVolumn;
	model->convLayer[ l ].poolVolumn			 = model->poolLayer[ l ].outHeight * model->poolLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].batchNormVolumn	 = model->convLayer[ l ].poolVolumn; 

   // increment the layers. 
   l++;
   model->cLayers = l;

   l = 0;
   model->fcLayer[ l ].in = out_channels * 8 * 8; 
   model->fcLayer[ l ].out = numclasses; 
   model->fcLayer[ l ].actFun = CNN_ACT_SWISH;
	model->fcLayer[ l ].offset = (numclasses * batchSize);  //TODO temp storage

   l ++;
   model->lLayers = l;

   //BATCH SIZE
	model->bias = 0; 
   model->batchSize = batchSize;

   //compute pSize; 
	computeParamSize( model ); 

   //compute Weights/Bias offsets here. 
	computeWeightOffsets( model ); 

   //compute zOffsets here. 
   //output of each layer here. 
	computeZOffsets( model, height, width, model->batchSize ); 


   fprintf( stderr, "======================\n");
   fprintf( stderr, " W and B Offsets... \n");
   for (int i = 0; i <= (model->cLayers + model->lLayers); i ++)
      fprintf( stderr, "%8d \n", model->wOffsets[ i ] ); 
   fprintf( stderr, "\n");

   fprintf( stderr, "Z Offsets ... \n" );
   for (int i = 0; i <= model->cLayers + model->lLayers + 1; i ++)
      fprintf( stderr, "%8d \n", model->zOffsets[ i ] );
   fprintf( stderr, "\n");
   fprintf( stderr, "\n");
	fprintf( stderr, "MaxDeltaSize: %d \n\n", model->maxDeltaSize ); 

   fprintf( stderr, "Params size: %d \n", model->pSize );
   fprintf( stderr, "Z size: %d \n", model->zSize );
   fprintf( stderr, "======================\n");

}

void readLenetCNN( CNN_MODEL *model, int channels, 
						int width, int height, int batchSize, int enableBias, int enableBatchNorm ) {

	int l = 0; 
	int h, w; 
	int t_h, t_w; 
	int out_channels;
	fprintf( stderr, "readConvolution... begin\n" ); 

	//init the model here. 
	//Conv2d( 3, 6, 5)
	model->convLayer[ l ].inChannels = channels; 
	model->convLayer[ l ].outChannels = 6; 
	model->convLayer[ l ].kSize= 5; 
	model->convLayer[ l ].height= height; 
	model->convLayer[ l ].width= width; 
	model->convLayer[ l ].stride = 1; 
	model->convLayer[ l ].padding = 0; 

	model->convLayer[ l ].batchNorm = PERFORM_NO_BATCH_NORM; 
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

	model->poolLayer[ l ].type = AVG_POOL; 
	model->poolLayer[ l ].pSize= 2; 
	model->poolLayer[ l ].height = h; 
	model->poolLayer[ l ].width = w; 
	model->poolLayer[ l ].stride = model->poolLayer[ l ].pSize; 
	model->poolLayer[ l ].padding = 0; 

	//Imge2Col Weights offsets here. 
	model->convLayer[ l ].activationOffset = h * w * model->convLayer[ l ].outChannels * batchSize; 
	model->convLayer[ l ].poolOffset = model->convLayer[ l ].activationOffset + 
										h * w * model->convLayer[ l ].outChannels * batchSize; 

   getDimensions( h, w, model->poolLayer[ l ].padding,
                  model->poolLayer[ l ].stride, model->poolLayer[ l ].pSize, &t_h, &t_w );
   model->poolLayer[ l ].outHeight = t_h; 
   model->poolLayer[ l ].outWidth = t_w; 

   if (model->convLayer[ l ].batchNorm == PERFORM_NO_BATCH_NORM){

      model->convLayer[ l ].outputOffset =
         model->convLayer[ l ].poolOffset;
	
		model->convLayer[ l ].variancesOffset = model->convLayer[ l ].meansOffset = 0; 
   } else {
      model->convLayer[ l ].batchNormOffset =
         model->convLayer[ l ].poolOffset + t_h * t_w * model->convLayer[l].outChannels * batchSize;

      model->convLayer[ l ].outputOffset =
         model->convLayer[ l ].batchNormOffset;

		model->convLayer[ l ].meansOffset = t_h * t_w * model->convLayer[ l ].outChannels * batchSize; 
		model->convLayer[ l ].variancesOffset = model->convLayer[ l ].meansOffset + 
															model->convLayer[ l ].outChannels; 

		model->convLayer[ l ].runningMeansOffset = model->convLayer[ l ].variancesOffset + 
															model->convLayer[ l ].outChannels; 
		model->convLayer[ l ].runningVariancesOffset = model->convLayer[ l ].runningMeansOffset + 
															model->convLayer[ l ].outChannels; 
   }

	//volumn Terms here. 
	model->convLayer[ l ].convVolumn = model->convLayer[ l ].outHeight * model->convLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].activationVolumn = model->convLayer[ l ].convVolumn; 
	model->convLayer[ l ].poolVolumn = model->poolLayer[ l ].outHeight * model->poolLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].batchNormVolumn = model->convLayer[ l ].poolVolumn; 
														
	//Layer 2 BEGIN

	getDimensions( h, w, model->poolLayer[ l ].padding, 
			model->poolLayer[ l ].stride, model->poolLayer[ l ].pSize, &h, &w ); 

	//Conv2d( 6, 16, 5 )
	l ++; 
	model->convLayer[ l ].inChannels = 6; 
	model->convLayer[ l ].kSize = 5; 
	model->convLayer[ l ].outChannels = 16; 
	model->convLayer[ l ].height= h; 
	model->convLayer[ l ].width= w; 
	model->convLayer[ l ].stride = 1; 
	model->convLayer[ l ].padding = 0; 
	model->actFuns[ l ] = CNN_ACT_SWISH; 

	model->convLayer[ l ].batchNorm = PERFORM_NO_BATCH_NORM; 
	/*
   model->convLayer[ l ].running_mean = 0;  
   model->convLayer[ l ].running_variance = 0;  
   model->convLayer[ l ].batch_mean = 0;  
   model->convLayer[ l ].batch_variance = 0; 
	*/

	getDimensions( h, w, model->convLayer[ l ].padding, 
			model->convLayer[ l ].stride, model->convLayer[ l ].kSize, &h, &w ); 
   model->convLayer[ l ].outHeight = h; 
   model->convLayer[ l ].outWidth = w; 

	model->poolLayer[ l ].type = AVG_POOL; 
	model->poolLayer[ l ].pSize = 2; 
	model->poolLayer[ l ].height= h; 
	model->poolLayer[ l ].width= w; 
	model->poolLayer[ l ].stride = model->poolLayer[ l ].pSize; 
	model->poolLayer[ l ].padding = 0; 

	//Img2Col Offset here. 
	model->convLayer[ l ].activationOffset = h * w * model->convLayer[ l ].outChannels * batchSize; 
	model->convLayer[ l ].poolOffset = model->convLayer[ l ].activationOffset + 
										h * w * model->convLayer[ l ].outChannels * batchSize; 

   getDimensions( h, w, model->poolLayer[ l ].padding,
                  model->poolLayer[ l ].stride, model->poolLayer[ l ].pSize, &t_h, &t_w );
   model->poolLayer[ l ].outHeight = t_h; 
   model->poolLayer[ l ].outWidth = t_w; 

   if (model->convLayer[ l ].batchNorm == PERFORM_NO_BATCH_NORM){

      model->convLayer[ l ].outputOffset =
         model->convLayer[ l ].poolOffset;
	
		model->convLayer[ l ].variancesOffset = model->convLayer[ l ].meansOffset = 0; 
   } else {
      model->convLayer[ l ].batchNormOffset =
         model->convLayer[ l ].poolOffset + t_h * t_w * model->convLayer[l].outChannels * batchSize;

      model->convLayer[ l ].outputOffset =
         model->convLayer[ l ].batchNormOffset;

		model->convLayer[ l ].meansOffset = t_h * t_w * model->convLayer[ l ].outChannels * batchSize; 
		model->convLayer[ l ].variancesOffset = model->convLayer[ l ].meansOffset + 
															model->convLayer[ l ].outChannels; 

		model->convLayer[ l ].runningMeansOffset = model->convLayer[ l ].variancesOffset + 	
															model->convLayer[ l ].outChannels; 
		model->convLayer[ l ].runningVariancesOffset = model->convLayer[ l ].runningMeansOffset + 
															model->convLayer[ l ].outChannels; 
   }

	//volumn Terms here. 
	model->convLayer[ l ].convVolumn = model->convLayer[ l ].outHeight * model->convLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].activationVolumn = model->convLayer[ l ].convVolumn; 
	model->convLayer[ l ].poolVolumn = model->poolLayer[ l ].outHeight * model->poolLayer[ l ].outWidth * model->convLayer[ l ].outChannels; 
	model->convLayer[ l ].batchNormVolumn = model->convLayer[ l ].poolVolumn; 

	// increment the layers. 
	l++;
	model->cLayers = l;

	//Fully Connected Layers here. 
	//nn.Linear(16*5*5, 120)

	l = 0; 
	model->fcLayer[ l ].in = 16 * 5 * 5; 
	model->fcLayer[ l ].out = 120; 
	model->fcLayer[ l ].actFun = CNN_ACT_SWISH; 
	model->fcLayer[ l ].offset = (120 * batchSize );

   //nn.Linear(120, 84)
	l ++; 
	model->fcLayer[ l ].in = 120;
	model->fcLayer[ l ].out = 84; 
	model->fcLayer[ l ].actFun = CNN_ACT_SWISH; 
	model->fcLayer[ l ].offset = (84 * batchSize );

   //nn.Linear(84, 10)
	l ++; 
	model->fcLayer[ l ].in = 84;
	model->fcLayer[ l ].out = 10; 
	//model->fcLayer[ l ].actFun = CNN_ACT_SOFTPLUS; 
	model->fcLayer[ l ].actFun = CNN_ACT_NONE;
	model->fcLayer[ l ].offset = (10 * batchSize );

	l ++; 
	model->lLayers = l; 

	//BATCH SIZE
	model->name = CNN_LENET; 
	model->bias = enableBias; 
	model->batchSize = batchSize; 

	//compute pSize; 
	computeParamSize( model ); 

	//compute Weights/Bias offsets here. 
	if (model->bias != 0)
		computeWeightBiasOffsets( model ); 
	else 
		computeWeightOffsets( model ); 

	//compute zOffsets here. 
	//output of each layer here. 
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

void readNeuralNet( NN_MODEL *model, int inputSize, int outputSize, int numPoints ){ 

	int l = 0;
	fprintf( stderr, "readNeuralNet... begin\n");

	//Init the model here. 
	model->layerSizes[ l ] = inputSize; // input layer - outputs the input
	model->actFuns[ l ] = ACT_LOGISTIC; 

	model->layerSizes [ ++l ] = 400;		// Hidden layer --1 
	model->actFuns[ l ] = ACT_LOGISTIC; 

	model->layerSizes [ ++l ] = 200; // Hidden layer --2 
	model->actFuns[ l ] = ACT_LOGISTIC; 

	model->layerSizes [ ++l ] = 100; // Hidden layer --3 
	model->actFuns[ l ] = ACT_LOGISTIC; 

	model->layerSizes [ ++l ] = 50; // Hidden layer --4 
	model->actFuns[ l ] = ACT_LOGISTIC; 

	model->layerSizes [ ++l ] = 25; // Hidden layer --5 
	model->actFuns[ l ] = ACT_LINEAR; 

	model->layerSizes [ ++l ] = 6; // Hidden layer --6 
	model->actFuns[ l ] = ACT_LOGISTIC; 

	model->layerSizes [ ++l ] = 25; // Hidden layer --7 
	model->actFuns[ l ] = ACT_LOGISTIC; 
	
	model->layerSizes [ ++l ] = 50; // Hidden layer --8 
	model->actFuns[ l ] = ACT_LOGISTIC; 

	model->layerSizes [ ++l ] = 100; // Hidden layer --9 
	model->actFuns[ l ] = ACT_LOGISTIC; 

	model->layerSizes [ ++l ] = 200; // Hidden layer --10 
	model->actFuns[ l ] = ACT_LOGISTIC; 

	model->layerSizes [ ++l ] = 400; // Hidden layer --11 
	model->actFuns[ l ] = ACT_LOGISTIC; 

	model->layerSizes [ ++l ] = outputSize; // Output layer --12 
	model->actFuns[ l ] = ACT_NONE; 
	
	model->numLayers = l;
	model->type = MODEL_TYPE_MSE; 

	for (int i = 0; i < MAX_LAYERS ; i ++)
		model->wOffsets[ i ] = model->bOffsets[ i ] = model->zOffsets[ i] = 0; 

	//compute the memory size required for weights

	int w = 0; 

	model->wOffsets[ 0 ] = 0;
	model->bOffsets[ 0 ] = model->layerSizes[ 1 ] * model->layerSizes[ 0 ]; 
	for (int i = 1; i < l; i ++) {
		w += model->layerSizes[ i ] * model->layerSizes[i-1] + model->layerSizes[ i ];
		model->wOffsets[ i ] = w; 
		model->bOffsets [ i ] = w + model->layerSizes[ i ] * model->layerSizes[ i + 1];
	}
	w += model->layerSizes[ l ] * model->layerSizes[ l - 1 ] + model->layerSizes[ l ]; 

	model->pSize= w;
	fprintf( stderr, "readNeuralNet: size of (W + b) = %d \n", w); 

	//zOffsets Here. 
	int weightsSize = 0; 
	model->zOffsets [ 0 ] = weightsSize ; 
	for (int i = 1; i <= l; i ++){
		weightsSize += model->layerSizes[ i-1 ] * numPoints; 
		model->zOffsets[ i ] = weightsSize; 
	}
	weightsSize += model->layerSizes[ l ] * numPoints;
	model->zSize = weightsSize; 
	fprintf( stderr, "readNeuralNet: size of (z) = %d \n", weightsSize ); 

}


void initSampledZOffsets (NN_MODEL *model, int numPoints)
{
	int zSize = 0; 
	model->sZOffsets [ 0 ] = zSize; 
	for (int i = 1; i <= model->numLayers; i ++){
		zSize += model->layerSizes[ i-1 ] * numPoints; 
		model->sZOffsets[ i ] = zSize; 
	}
	zSize += model->layerSizes[ model->numLayers ] * numPoints;
	model->sampledZSize = zSize; 
	fprintf( stderr, "readNeuralNet: size of (sampledZ) = %d \n", zSize ); 

}

void initRZOffsets( NN_MODEL *model, int datasetSize)
{
fprintf (stderr, "Dataset Size: %d \n", datasetSize ); 
	int rZFullSize = 0; 
	model->rZOffsets[ 0 ] = rZFullSize;
	for (int i = 1; i <= model->numLayers ; i ++){
		rZFullSize += model->layerSizes[ i ] * datasetSize; 
		model->rZOffsets[ i ] = rZFullSize; 
	}
	model->rFullSize = rZFullSize; 
	fprintf( stderr, "readNeuralNet: size of (rZFull) = %d \n", rZFullSize ); 

}



void initSampledROffsets( NN_MODEL *model, int sampleSize )
{
	int rZSize = 0; 
	model->sRZOffsets[ 0 ] = rZSize;
	for (int i = 1; i <= model->numLayers ; i ++){
		rZSize += model->layerSizes[ i ] * sampleSize; 
		model->sRZOffsets[ i ] = rZSize; 
	}
	model->sampledRSize = rZSize; 
	fprintf( stderr, "readNeuralNet: size of (sampled rFull) = %d \n", rZSize ); 

}


void autoencoderInitializations ( NN_MODEL *model, DEVICE_DATASET *data )
{
	//test and train sets are the same
	data->trainSetY = data->trainSetX; 
	data->trainSizeY = data->trainSizeX;

	data->testSetY = data->testSetX; 
	data->testSizeY = data->testSizeX; 

	fprintf( stderr, "Weights Size: %d \n", model->pSize ); 

	//allocate space for Weight. 
	cuda_malloc( (void **)&data->weights, model->pSize * sizeof(real), 0, ERROR_MEM_ALLOC );
}

void cnnInitializations( CNN_MODEL *model, DEVICE_DATASET *data )
{
	//test and train sets are the same
	//data->trainSetY = data->trainSetX; 
	//data->trainSizeY = data->trainSizeX;

	//data->testSetY = data->testSetX; 
	//data->testSizeY = data->testSizeX; 

	fprintf( stderr, "Weights Size: %d \n", model->pSize ); 

	//allocate space for Weight. 
	cuda_malloc( (void **)&data->weights, model->pSize * sizeof(real), 0, ERROR_MEM_ALLOC );
}
