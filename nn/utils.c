#include <nn/utils.h>
#include <nn/nn_decl.h>

#include <core/errors.h>
#include <core/structdefs.h>
#include <device/cuda_utils.h>


#include <stdlib.h>
#include <stdio.h>

void getDimensions( int height, int width, int padding, int stride, int kernel, 
	int *h, int *w)
{
	*h = (height + 2 * padding - kernel) / stride + 1; 
	*w = (width + 2 * padding - kernel) / stride + 1; 
}

void computeParamSize( CNN_MODEL *model ) 
{
   model->pSize = 0;
   for (int i = 0; i < model->cLayers; i ++){
      CONV_LAYER t = model->convLayer[ i ];
      model->pSize += t.outChannels * t.kSize * t.kSize * t.inChannels;
		if (model->bias >= 1)
      	model->pSize += t.outChannels;

		/*
		if (t->batchNorm == PERFORM_BATCH_NORM_TRAINABLE ){
			model->pSize += t.outChannels * 2; // Gamma (scaling) + beta (shift)
		}
		*/
   }

   for (int i = 0; i < model->lLayers; i ++) {

      FC_LAYER t = model->fcLayer[ i ];
      model->pSize += t.out * t.in;
		if (model->bias >= 1)
      	model->pSize += t.out;
   }
}

void computeWeightOffsets( CNN_MODEL *model )
{
   for (int i = 0; i < MAX_LAYERS ; i ++)
      model->wOffsets[ i ] = model->bOffsets[ i ] = model->zOffsets[ i ] = 0;

	if (model->bias == 0) for (int i = 0; i < MAX_LAYERS; i ++ ) model->bOffsets[ i ] = INT_MAX; 

	CONV_LAYER t; 
	if (model->cLayers != 0){ 
   	t = model->convLayer[ 0 ];
   	model->wOffsets[ 0 ] = 0;
   	for (int i = 1; i < model->cLayers; i ++){
      	CONV_LAYER p = model->convLayer[ i-1 ];
      	model->wOffsets[ i ] = p.outChannels * p.kSize * p.kSize * p.inChannels + model->wOffsets[ i-1 ];
   	}
   	t = model->convLayer[ model->cLayers-1 ];
   	model->wOffsets[ model->cLayers ] = model->wOffsets[ model->cLayers - 1 ] + t.outChannels * t.kSize * t.kSize * t.inChannels; 
	}

   for (int i = 1; i < model->lLayers; i ++) {
      FC_LAYER p = model->fcLayer[ i-1 ];
      FC_LAYER t = model->fcLayer[ i ];

      model->wOffsets[ i + model->cLayers ] = p.out * p.in + model->wOffsets[ i + model->cLayers - 1] ;
   }
	FC_LAYER f = model->fcLayer[ model->lLayers - 1 ];
	model->wOffsets[ model->cLayers + model->lLayers ] = model->wOffsets[ (model->lLayers - 1) + (model->cLayers) ] + f.out * f.in; 
}


void computeWeightBiasOffsets( CNN_MODEL *model )
{
   for (int i = 0; i < MAX_LAYERS ; i ++)
      model->wOffsets[ i ] = model->bOffsets[ i ] = model->zOffsets[ i ] = 0;

	CONV_LAYER t; 
	if (model->cLayers != 0){ 
   	t = model->convLayer[ 0 ];
   	model->bOffsets[ 0 ] = t.outChannels * t.kSize * t.kSize * t.inChannels ;
   	for (int i = 1; i < model->cLayers; i ++){
      	CONV_LAYER t = model->convLayer[ i-1 ];
      	CONV_LAYER p = model->convLayer[ i ];
      	model->wOffsets[ i ] = t.outChannels + model->bOffsets[ i-1 ];
      	model->bOffsets[ i ] = p.outChannels * p.kSize * p.kSize * p.inChannels + model->wOffsets[ i ];
   	}
   	t = model->convLayer[ model->cLayers-1 ];
   	model->wOffsets[ model->cLayers ] = model->bOffsets[ model->cLayers - 1 ] + t.outChannels;
	}

   FC_LAYER t1 = model->fcLayer[ 0 ];
	if (model->cLayers != 0)
   	model->bOffsets[ model->cLayers ] = model->wOffsets[ model->cLayers ] + t1.out * t1.in;
	else
   	model->bOffsets[ 0 ] = t1.out * t1.in;
   for (int i = 1; i < model->lLayers; i ++) {
      FC_LAYER p = model->fcLayer[ i-1 ];
      FC_LAYER t = model->fcLayer[ i ];

      model->wOffsets[ i + model->cLayers ] = p.out + model->bOffsets[ i + model->cLayers - 1] ;
      model->bOffsets[ i + model->cLayers ] = t.out * t.in  + model->wOffsets[ i + model->cLayers ];
   }
}

void computeZOffsets( CNN_MODEL *model, int height, int width, int batchSize )
{
   for (int i = 0; i < MAX_LAYERS ; i ++) model->zOffsets[ i ] = 0;

   for (int i = 0; i < MAX_LAYERS ; i ++) {
       model->zOffsets[ i ] = 0;
       model->zztOffsets[ i ] = 0;  
   }

   model->zOffsets[ 0 ] = 0;  
   model->zOffsets[ 1 ] = 0;  
   model->zztOffsets[ 0 ] = 0;  
   model->maxDeltaSize = 0;
	model->zztSize = 0; 

   for (int i = 1; i <= model->cLayers; i ++){
      CONV_LAYER t = model->convLayer[ i-1 ];
      POOL_LAYER p = model->poolLayer[ i-1 ];

		//conv + activation
      model->zOffsets[ i+1 ] = model->zOffsets[ i ] + 
				batchSize * (t.outHeight * t.outWidth * t.outChannels * 2); //Conv output + activation output. 

		if ((t.batchNorm == PERFORM_BATCH_NORM) || 
			(t.batchNorm == PERFORM_BATCH_NORM_TRAINABLE) ){
				model->zOffsets[ i + 1 ] += batchSize * t.outHeight * t.outWidth * t.outChannels; 
				model->zOffsets[ i + 1 ] += 4 * t.outChannels;
		}

		model->zztOffsets[ i + 1 ] = model->zztOffsets[ i ] + 
												batchSize * t.outHeight * t.outWidth * t.outChannels; 
		model->zztSize += batchSize * t.outHeight * t.outWidth * t.outChannels; 

		//Pool 
		if (p.type != NO_POOL ) {
      	model->zOffsets[ i+1 ] += 
				batchSize * ( p.outHeight * p.outWidth * t.outChannels ) ;
		}


		if (p.type != NO_POOL) {
      	fprintf( stderr, "%d (%d, %d, %d, %d)\n", model->zOffsets[ i ], batchSize, t.outHeight, t.outWidth, t.outChannels) ;
		} else {
      	fprintf( stderr, "%d (%d, %d, %d, %d)\n", model->zOffsets[ i ], batchSize, p.outHeight, p.outWidth, t.outChannels) ;
		}

		if ((model->zOffsets[ i + 1 ] - model->zOffsets[ i ]) > model->maxDeltaSize )
			model->maxDeltaSize = model->zOffsets[ i+1 ] - model->zOffsets[ i ];
   }

   for (int i = 1; i <= model->lLayers; i ++) {
      FC_LAYER t = model->fcLayer[ i-1 ];
      model->zOffsets[ i + model->cLayers + 1 ] = model->zOffsets[ i + model->cLayers ] +
                  t.out * batchSize * 2; //TODO - This is to store the output (Wz + b), used for hv product
      fprintf( stderr, "%d (%d, %d, %d )\n", model->zOffsets[ i + model->cLayers ],
                  batchSize, t.out, t.in ) ;

		model->zztOffsets[ i + 1 + model->cLayers ] = model->zztOffsets[ i + model->cLayers ] + 	
																		t.out * batchSize; 
		model->zztSize += t.out * batchSize; 

		if ((model->zOffsets[ model->cLayers + i + 1 ] - model->zOffsets[ model->cLayers + i ]) > model->maxDeltaSize )
			model->maxDeltaSize = model->zOffsets[ model->cLayers + i + 1 ] - model->zOffsets[ model->cLayers + i ];
   }
   model->zSize = model->zOffsets[ model->cLayers + model->lLayers + 1 ];
}
