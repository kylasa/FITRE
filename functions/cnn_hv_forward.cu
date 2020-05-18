
#include <functions/cnn_hv_forward.h>
#include <functions/eval_gradient.h>
#include <functions/eval_convolution.h>
#include <functions/dev_backprop_convolution.h>
#include <functions/rop_fc_layer.h>
#include <functions/dev_batch_norm.h>
#include <functions/dev_transpose.h>

#include <core/errors.h>

#include <device/cuda_utils.h>
#include <utilities/print_utils.h>

long cnnROpForwardMemRequired( CNN_MODEL *model ){

	long imgColSize = 0, wRz = 0, tl = 0, tc = 0; 

	for (int l = 0; l < model->cLayers; l ++ ){
      CONV_LAYER *c = &( model->convLayer[l] );  
      POOL_LAYER *p = &( model->poolLayer[l] );  

		//ROp Convolution Layer
		//ImgCols of the input + rzWt
		imgColSize = (p->height * p->width * model->batchSize * c->kSize * c->kSize * c->inChannels) + 
							(p->height * p->width * model->batchSize * c->outChannels)	;

		if (tc < imgColSize) tc = imgColSize; 
	}

	//reshaping of Rz
	// insignificant compared to these two terms....

	//ROp Linear Layer
   for (int l = 0; l < model->lLayers; l ++){
   	FC_LAYER f = model->fcLayer[ l ];  
		
		//W * Rz
		wRz = f.out * model->batchSize;
		if (tl < wRz)  tl = wRz; 
	}

	return (tl > tc) ? tl : tc; 
}

//
// Forward Pass for the hessian vector product. 
//
// Outputs ==> Rz ( ImgColWt, actOut, poolOut ) for convolution
//					Rz ( R{ f( Wz + b ) } } for linear layers... 
//
//
void cnnROpForward(CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch, real *z, real *vector, 
	real *rz, int s, int curBatchSize, 
	real *devPtr, real *hostPtr ) {

	/*
   real *nextDevPtr = scratch->nextDevPtr;
   real *nextPagePtr = scratch->nextPageLckPtr;
   real *nextHostPtr = scratch->nextHostPtr;
	*/

	real *nextDevPtr = devPtr; 
	real *nextHostPtr = hostPtr; 

   real *weights = data->weights;
   int *wOffsets = model->wOffsets;
   int *bOffsets = model->bOffsets;
   int *zOffsets = model->zOffsets;

	if (model->bias == 0)
		bOffsets = NULL; 


   //very first layer here. 
   CONV_LAYER *convLayer = &( model->convLayer[0] );
   POOL_LAYER *poolLayer = &( model->poolLayer[0] );
   //applyROpConvolutionLayer( data->trainSetX + s * data->features, NULL, 0,  z + zOffsets[ 1 ], NULL, curBatchSize,
   applyROpConvolutionLayer( data->currentBatch, NULL, 0,  z + zOffsets[ 1 ], NULL, curBatchSize,
            convLayer->inChannels, convLayer->height, convLayer->width,
            convLayer->kSize, convLayer->padding, convLayer->stride,
				convLayer->outHeight, convLayer->outWidth, 
            weights, ((model->bias == 0) ? NULL : (weights + bOffsets[ 0 ])), 
				vector, ((model->bias == 0) ? NULL : (vector + bOffsets[ 0 ])), 
				NULL, rz + zOffsets[ 1 ], convLayer->outChannels, model->actFuns[ 0 ],
            poolLayer->pSize, poolLayer->stride, poolLayer->padding, poolLayer->type,
				convLayer->activationOffset, convLayer->poolOffset, convLayer->batchNormOffset, convLayer->outputOffset,
				convLayer->convVolumn, convLayer->activationVolumn, convLayer->poolVolumn, convLayer->batchNormVolumn,
            nextDevPtr, nextHostPtr, convLayer->batchNorm, BATCH_NORM_EPSILON, model->batchSize );


	for (int l = 1; l < model->cLayers; l ++ ){
		CONV_LAYER *prevLayer = &( model->convLayer[ l-1 ] ); 
      convLayer = &( model->convLayer[l] );  
      poolLayer = &( model->poolLayer[l] );  

#ifdef DEBUG_ROP
fprintf( stderr, "... Beginning with ROp Forward Layer: %d\n", l); 

if (convLayer->batchNorm != PERFORM_NO_BATCH_NORM) {
	copy_host_device( hostPtr, z + zOffsets[ l + 1 ] + convLayer->batchNormOffset + convLayer->meansOffset, 
		sizeof(real) * convLayer->outChannels, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
	fprintf( stderr, "Means for this layer: \n"); 
	print2DMatrix( hostPtr, convLayer->outChannels, 1 ); 

	copy_host_device( hostPtr, z + zOffsets[ l + 1 ] + convLayer->batchNormOffset + convLayer->variancesOffset, 
		sizeof(real) * convLayer->outChannels, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
	fprintf( stderr, "Variances for this layer: \n"); 
	print2DMatrix( hostPtr, convLayer->outChannels, 1 ); 
}

#endif

		//SUDHIR-DOUBLE-CHECK-CHANGE
		//applyROpConvolutionLayer( NULL, z + zOffsets[ l ], (2 * prevLayer->convOffset), 
		//SUDHIR-DOUBLE-CHECK-CHANGE
		applyROpConvolutionLayer( NULL, z + zOffsets[ l ], prevLayer->outputOffset,
			z + zOffsets[ l+1 ], rz + zOffsets[ l ], curBatchSize, 
			convLayer->inChannels, convLayer->height, convLayer->width, 
			convLayer->kSize, convLayer->padding, convLayer->stride, 
			convLayer->outHeight, convLayer->outWidth, 
			weights + wOffsets[ l ], ((model->bias == 0) ? NULL : (weights + bOffsets[ l ])), 
			vector + wOffsets[ l ], ((model->bias == 0) ? NULL : (vector + bOffsets[ l ])), 
			NULL, rz + zOffsets[ l + 1 ], convLayer->outChannels, model->actFuns[ 0 ], 
			poolLayer->pSize, poolLayer->stride, poolLayer->padding, poolLayer->type,  /// STRIDE was incorrect in this call... BATCH NORM NOT WORKING...
			convLayer->activationOffset, convLayer->poolOffset, convLayer->batchNormOffset, convLayer->outputOffset,
			convLayer->convVolumn, convLayer->activationVolumn, convLayer->poolVolumn, convLayer->batchNormVolumn,
			nextDevPtr, nextHostPtr, convLayer->batchNorm, BATCH_NORM_EPSILON, model->batchSize ); 
	}

	//convert the output of the convolution to Linear Layers input.. 
	// h * w * n X channels --> channels * h * w X n

   POOL_LAYER *p = &( model->poolLayer[ model->cLayers - 1 ] );  
   CONV_LAYER *c = &( model->convLayer[ model->cLayers - 1 ] );  		

	/*
   int p_height = ( p->height - p->pSize ) / p->pSize+ 1;  
   int p_width = ( p->width - p->pSize) / p->pSize + 1;  

   int col_height = (c->height + 2 * c->padding - c->kSize ) / c->stride + 1;  
   int col_width = (c->width + 2 * c->padding - c->kSize ) / c->stride + 1;  

	int poolOffset = 2 * col_height * col_width * c->outChannels * curBatchSize; 
	*/
	int outputOffset = c->outputOffset; 

/*
   real *imgColWtOut = output;
   real *actOut = imgColWtOut + col_height * col_width * out_channels * samples; 
   real *poolOut = actOut + col_height * col_width * out_channels * samples; 
*/

/*
	output of pooling layer is NOT (p->height)
*/
#ifdef DEBUG_ROP
fprintf( stderr, "... BEFORE  Reshaping of Matrix for linear layers... \n"); 
copy_host_device( hostPtr, rz + zOffsets[ model->cLayers ] + outputOffset, 
			sizeof(real) * c->poolVolumn * curBatchSize , 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST );  
print2DMatrix( hostPtr, curBatchSize * p->outHeight * p->outWidth, c->outChannels ); 
#endif

	//SUDHIR TESTING ERRORS SK-1
   //reshapeMatrix( rz + zOffsets[  model->cLayers ] + outputOffset,  
   //           curBatchSize, c->outChannels, p->outHeight * p->outWidth, nextDevPtr );  
	//SUDHIR TESTING ERRORS SK-1
   reshapeMatrix( rz + zOffsets[  model->cLayers ] + outputOffset,  
              c->outChannels, curBatchSize, p->outHeight * p->outWidth, nextDevPtr );  
               //curBatchSize, c->outChannels, p_height * p_width, nextDevPtr );  

/*
   copy_device( rz + zOffsets[ model->cLayers ] + outputOffset, nextDevPtr, 
                     sizeof(real) * c->poolVolumn * curBatchSize,
                     ERROR_MEMCPY_DEVICE_DEVICE );  
*/

	//SK-2 Commented out the above because of transpose below... 
	int transElements = c->outChannels * p->outHeight * p->outWidth * curBatchSize; 
	int transBlocks = (BLOCK_SIZE - 1 + transElements) / BLOCK_SIZE; 
	ker_transpose <<< transBlocks, BLOCK_SIZE >>> 
		( nextDevPtr, transElements, c->outChannels, p->outHeight, p->outWidth, curBatchSize, rz + zOffsets[ model->cLayers ] + outputOffset ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 

#ifdef DEBUG_ROP
fprintf( stderr, "... Done with Reshaping of Matrix for linear layers... \n"); 
copy_host_device( hostPtr, rz + zOffsets[ model->cLayers ] + outputOffset, 
			sizeof(real) * c->poolVolumn * curBatchSize , 
			cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST );  
print2DMatrix( hostPtr, curBatchSize * p->outHeight * p->outWidth, c->outChannels ); 
#endif


   for (int l = 0; l < model->lLayers; l ++){

		if (l > 0) outputOffset = 0; 

   	FC_LAYER f = model->fcLayer[ l ];  
      applyROpLayerActivation ( f.actFun, 
            weights+ wOffsets[ l + model->cLayers ], f.out, f.in, 
            ((model->bias == 0) ? NULL : (weights+ bOffsets[ l + model->cLayers ])), f.out, 
            z + zOffsets[ l + model->cLayers ] + outputOffset, f.in, curBatchSize, 
				z + zOffsets[ l + 1 + model->cLayers ], f.out, curBatchSize, 
				vector+ wOffsets[ l + model->cLayers ], 
				((model->bias == 0) ? NULL : (vector+ bOffsets[ l + model->cLayers ])), 
				rz + zOffsets[ l + model->cLayers ] + outputOffset,
				rz + zOffsets[ l + 1 + model->cLayers ], f.offset, nextDevPtr, nextHostPtr );  

#ifdef DEBUG_ROP
fprintf( stderr, "... Done with ROp Forward (Linear Layer)... %d\n", l ); 
#endif

#ifdef DEBUG_ROP
copy_host_device( hostPtr, rz + zOffsets[ l + 1 + model->cLayers ], 
	sizeof(real) * f.out * curBatchSize, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, f.out , curBatchSize );
#endif
	}
}
