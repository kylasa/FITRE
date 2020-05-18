
#include <functions/cnn_backward.h>
#include <functions/dev_layer_error.h>
#include <functions/dev_initializations.h>
#include <functions/dev_pool.h>
#include <functions/dev_image.h>
#include <functions/dev_layer_error.h>
#include <functions/dev_backprop_convolution.h>
#include <functions/swish.h>
#include <functions/dev_batch_norm.h>
#include <functions/dev_transpose.h>

#include <nn/nn_decl.h> 

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

#include <utilities/reduce.h>
#include <utilities/print_utils.h>

#include <core/errors.h>


GLOBAL void ker_reduce_filters( real *input, int size,
		real *output )
{
	extern __shared__ real myresults[] ; 
	int myChannel = blockIdx.x; 
	int myIdx = threadIdx.x + blockIdx.x * blockDim.x; 

	int lane = threadIdx.x >> 5;

	real sdata = 0; 
	
	if (threadIdx.x < size * size) 
		sdata = input[ myChannel * size * size + threadIdx.x ]; 
	__syncthreads (); 
	
	sdata = warpSum( sdata ); 
	if (threadIdx.x % WARP_SIZE == 0) myresults[ lane ] = sdata; 	
	__syncthreads (); 

	sdata = (lane < 1) ? myresults[ threadIdx.x ] : 0; 
	__syncthreads (); 

   if (lane == 0) sdata = warpSum( sdata );
   if(threadIdx.x == 0) output [ blockIdx.x  ] =  sdata;
}

long cnnBackwardMemRequired( CNN_MODEL *model ){

	long imgColSize = 0, dImgColSize = 0; 
	
   for (int i = 0; i < model->cLayers; i ++) {
      CONV_LAYER *c = & (model->convLayer[ i ]);  
      POOL_LAYER *p = & (model->poolLayer[ i ]); 

		//Img2Col conversion of Z
      if (imgColSize < (p->height * p->width * model->batchSize) * (c->inChannels * c->kSize * c->kSize) )
         imgColSize = (p->height * p->width * model->batchSize) * (c->inChannels * c->kSize * c->kSize) ;

		//One Vector
		//Reshape of deltas... 
		//These two are always less then the below value

		//Img2Col Conversion of delta;  or Weights Reordering
		if (dImgColSize < ((c->height * c->width * model->batchSize) * (c->inChannels * c->kSize * c->kSize) + 
										c->outChannels * c->inChannels * c->kSize * c->kSize) )
			dImgColSize = (c->height * c->width * model->batchSize) * (c->inChannels * c->kSize * c->kSize) + 
										c->outChannels * c->inChannels * c->kSize * c->kSize;
   }   
	return ((imgColSize < dImgColSize) ? dImgColSize : imgColSize);
}

void cnnBackward(CNN_MODEL *model, DEVICE_DATASET *data, 
		real *devPtr, real *z, real *gradient,
		real *dx, real *delta, real *delta_new, 
		int offset, int batchSize, real *hostPtr ) {

	int *wOffsets = model->wOffsets; 
	int *bOffsets = model->bOffsets; 
	int *zOffsets = model->zOffsets; 
	real *weights = data->weights;

	if (model->bias == 0)
		bOffsets = NULL; 

	real *dataset; 
	real *temp;

	real alpha, beta; 
	real *nextDevPtr = devPtr; 

	int n = batchSize;
	int cLayers = model->cLayers; 
	int lLayers = model->lLayers; 
	int count, blocks; 


	// dx  = delta -- needs to be copied somewhere for hessian vector product.
	//copy_device( dx + zOffsets[ lLayers + cLayers ], delta, 
	//					sizeof(real) * data->numClasses * n, ERROR_MEMCPY_DEVICE_DEVICE ); 

	//Used for conversion of the delta to pass to the 
	//convolution layers... 
	CONV_LAYER *cLayer = & (model->convLayer[ cLayers - 1 ] ); 
	POOL_LAYER *pLayer = & (model->poolLayer[ cLayers - 1 ] ); 

	alpha = 1.0; beta = 0; 
	for (int f = model->lLayers - 1; f >= 0; f --) {
		FC_LAYER l = model->fcLayer[ f ]; 		

#ifdef DEBUG_CNN
fprintf( stderr, "BackwardPass Linear Layer: %d \n", f ); 
#endif

		//delta = delta * f'(z)
		//switch( model->actFuns[ f ] )
		switch( l.actFun ) { 
			case ACT_LOGISTIC: 
				count = (l.out * n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
         	kerNNBackPropLogisticErrors <<<count, BLOCK_SIZE >>> 
            	( delta, z + zOffsets[ f + cLayers + 1 ], count );  
         	cudaThreadSynchronize (); 
         	cudaCheckError (); 
				break;

			case ACT_TANH: 
				count = (l.out * n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
         	kerNNBackPropTanHErrors <<< count, BLOCK_SIZE >>>
            	( delta, z + zOffsets[ f + cLayers + 1 ], count);
         	cudaThreadSynchronize ();
         	cudaCheckError ();
				break; 
		
			case ACT_LINEAR:
				break;

			case CNN_ACT_SOFTPLUS: 

#ifdef DEBUG_DETAILED
fprintf( stderr, "Input to the Softmax layer.... \n"); 
copy_host_device( hostPtr, z + zOffsets[ f + cLayers + 1 ] + l.offset, sizeof(real) * l.out * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, n); 
#endif

				count = (l.out * n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
				//use input to this linear layer... 
				kerNNBackPropSOFTPLUS <<< count, BLOCK_SIZE >>> 
					( delta, z + zOffsets[ f + cLayers + 1 ] + l.offset , l.out * n);
				cudaThreadSynchronize (); 
				cudaCheckError (); 
				break;

			case CNN_ACT_SWISH: 

				count = (l.out * n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
				kerNNBackPropSwish <<<count, BLOCK_SIZE>>> 
						( 	z + zOffsets[ f + cLayers + 1] + l.offset, 
							z + zOffsets[ f + cLayers + 1 ],
							delta, l.out * n ); 
				cudaThreadSynchronize (); 
				cudaCheckError (); 
				break;	

			case CNN_ACT_NONE: 
				break;

			default: 
				fprintf( stderr, "Unknown activation function in Linear Layers... %d\n", 
							model->actFuns[ f ]); 
				exit ( -1 ); 
		}

#ifdef DEBUG_CNN
fprintf( stderr, "BackwardPass: Done with activation function\n"); 
#endif

#ifdef DEBUG_DETAILED
fprintf( stderr, "Updated error values are .... \n"); 
copy_host_device( hostPtr, delta, sizeof(real) * l.out * n, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix(hostPtr, l.out, n); 
#endif

		//dW = delta * z'
		alpha = 1.0; beta = 0; 
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
									l.out, l.in, n, 
									&alpha, delta, l.out, 
									//SUDHIR-DOUBLE-CHECK-CHANGES
									//z + zOffsets[ f + cLayers ] + ((f == 0) ? (2 * cLayer->convOffset) : (0) ), l.in, 
									//SUDHIR-DOUBLE-CHECK-CHANGES
									//z + zOffsets[ f + cLayers ] + ((f == 0) ? (cLayer->poolOffset) : (0) ), l.in, 
									z + zOffsets[ f + cLayers ] + ((f == 0) ? (cLayer->outputOffset) : (0) ), l.in, 
									&beta, gradient + wOffsets [ f + cLayers ], l.out ) ); 

/*
cublasCheckError( cublasDnrm2( cublasHandle, l.in * l.out, gradient + wOffsets[ f + cLayers ], 1, hostPtr ) ); 
fprintf( stderr, "dx Norm of level: %d is %e\n", f, *hostPtr ); 
*/

#ifdef DEBUG_DETAILED
fprintf( stderr, "DW Linear Layer ....\n"); 
copy_host_device( hostPtr, gradient + wOffsets [ f + cLayers ], 
			sizeof(real) * l.out * l.in, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, l.in ); 
fprintf( stderr, "BackwardPass: LinearLayer ..... donw dW\n"); 
#endif

		if (model->bias != 0) {

			//db = sum( delta, 2 ) - out terms
			blocks = ( n + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 
			kerInitOneVector <<<blocks, BLOCK_SIZE >>> 
				( nextDevPtr, n ); 
			cudaThreadSynchronize (); 
			cudaCheckError (); 			

			cublasCheckError( 
				cublasDgemv( cublasHandle, CUBLAS_OP_N, 
							l.out, n, &alpha, delta, l.out, 
							nextDevPtr, 1, &beta, gradient + bOffsets[ f + cLayers ], 1) ); 

#ifdef DEBUG_DETAILED
fprintf( stderr, "db----\n"); 
copy_host_device( hostPtr, gradient + bOffsets[ f + cLayers ], 
		sizeof(real) * l.out, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, 1, l.out ); 
fprintf( stderr, "BackwardPass: LinearLayer ..... donw db\n"); 
#endif
		}


		// dx  = delta -- needs to be copied somewhere for hessian vector product.
		copy_device( dx + zOffsets[ f + cLayers + 1 ], delta, 
							sizeof(real) * l.out * n, ERROR_MEMCPY_DEVICE_DEVICE ); 

#ifdef DEBUG_DETAILED
fprintf( stderr, "Storing the dx value for Rdx evaluation... \n"); 
copy_host_device( hostPtr, dx + zOffsets[ f + cLayers + 1 ], sizeof(real) * l.out * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, n ); 
#endif


		//udpate the error term here. 
		// delta = w' * delta
		cublasCheckError ( 
			cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
							l.in, n, l.out, 
							&alpha, weights + wOffsets[ f + cLayers ], l.out, 
							delta, l.out, &beta, delta_new, l.in ) ); 
							//TODO Check this carefully... 
							//dx + zOffsets[ f + 1 + cLayers ], l.out, &beta, delta_new, l.in ) ); 


#ifdef DEBUG_CNN
fprintf( stderr, "BackwardPass: LinearLayer.... done with Delta\n"); 
#endif

/*
		copy_device( dx + zOffsets[ f + cLayers ], delta_new, 
							sizeof(real) * l.in * n, ERROR_MEMCPY_DEVICE_DEVICE ); 
*/

#ifdef DEBUG_DETAILED
fprintf( stderr, "BackwardPass: updated delta is ..... \n"); 
copy_host_device( hostPtr, delta_new, sizeof(real) * l.in * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.in, n ); 
#endif


		//swap the pointers. 
		temp = delta_new; 
		delta_new = delta; 
		delta = temp; 
	}

	// CONVERSION OF ERRORS from 
	// h * w * c X n to h * w * n X c
   //CONV_LAYER* c = &(model->convLayer[ cLayers - 1 ]);  
   //POOL_LAYER* p = &(model->poolLayer[ cLayers - 1 ]);  
	/*
   int p_height = ( pLayer->height - pLayer->pSize ) / pLayer->pSize+ 1;  
   int p_width = ( pLayer->width - pLayer->pSize) / pLayer->pSize + 1;  
   int col_height = (cLayer->height + 2 * cLayer->padding - cLayer->kSize ) / cLayer->stride + 1;  
   int col_width = (cLayer->width + 2 * cLayer->padding - cLayer->kSize ) / cLayer->stride + 1;  
   int poolOffset = 2 * col_height * col_width * cLayer->outChannels * batchSize; 
	*/

	//SUDHIR TESTING ERRORS SK-1
	//reshapeMatrix( delta, cLayer->outChannels, n, pLayer->outHeight * pLayer->outWidth, nextDevPtr ); 
	//SUDHIR TESTING ERRORS SK-1

/*
	reshapeMatrix( delta, n, cLayer->outChannels, pLayer->outHeight * pLayer->outWidth, nextDevPtr ); 

	//SK-2 COMMENTED OUT BECAUSE OF RESHAPE
	int numElements = cLayer->outChannels * pLayer->outHeight * pLayer->outWidth * n; 
	int transposeBlocks = (BLOCK_SIZE - 1 + numElements) / BLOCK_SIZE; 
	ker_transpose <<< transposeBlocks, BLOCK_SIZE >>> 
		(nextDevPtr, numElements, cLayer->outChannels , pLayer->outHeight , pLayer->outWidth , n, delta ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 	
*/
	//delta -- reshape and transpose here. 
	if (pLayer->type != NO_POOL ) {
		reshapeMatrix( delta, n, cLayer->outChannels, pLayer->outHeight * pLayer->outWidth, nextDevPtr ); 
		int numElements = cLayer->outChannels * pLayer->outHeight * pLayer->outWidth * n; 
		int transposeBlocks = (BLOCK_SIZE - 1 + numElements) / BLOCK_SIZE; 
		ker_transpose <<< transposeBlocks, BLOCK_SIZE >>> 
			(nextDevPtr, numElements, cLayer->outChannels , pLayer->outHeight , pLayer->outWidth , n, delta ); 
		cudaDeviceSynchronize (); 
		cudaCheckError (); 	
	} else {
		reshapeMatrix( delta, n, cLayer->outChannels, cLayer->outHeight * cLayer->outWidth, nextDevPtr ); 
		int numElements = cLayer->outChannels * cLayer->outHeight * cLayer->outWidth * n; 
		int transposeBlocks = (BLOCK_SIZE - 1 + numElements) / BLOCK_SIZE; 
		ker_transpose <<< transposeBlocks, BLOCK_SIZE >>> 
			(nextDevPtr, numElements, cLayer->outChannels , cLayer->outHeight , cLayer->outWidth , n, delta ); 
		cudaDeviceSynchronize (); 
		cudaCheckError (); 	
	}



/*
	copy_device( delta, nextDevPtr, 
			sizeof(real) * n * cLayer->poolVolumn, ERROR_MEMCPY_DEVICE_DEVICE ); 	
*/

/* TODO -- NOT Needed here. 
	Reshape is only for the updaded error... not for the snapshot error value... 
	copy_device( dx + zOffsets[ cLayers ], delta, 
					sizeof(real) * cLayer->outChannels * n * p_height * p_width, 
					ERROR_MEMCPY_DEVICE_DEVICE ); 
*/

#ifdef DEBUG_DETAILED
fprintf( stderr, "Reshaped delta is as follows.... \n\n"); 
copy_host_device( hostPtr, delta, sizeof(real) * pLayer->outHeight * pLayer->outWidth * n * cLayer->outChannels, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, pLayer->outHeight * pLayer->outWidth * n, cLayer->outChannels ); 
fprintf( stderr, "Done with reshape... \n\n"); 
#endif


	//Convolution Layers... 
	//
	// Z for Convolution layer is 
	//			(imgColWt + actOut + poolOut)
	//
	for (int c = model->cLayers - 1; c >= 0; c -- ){
		CONV_LAYER l = model->convLayer[ c ]; 
		POOL_LAYER p = model->poolLayer[ c ];

#ifdef DEBUG_CNN
fprintf( stderr, "BackwardPass: Convolution Layer : %d \n", c ); 
#endif

/*
		if (l.batchNorm == PERFORM_BATCH_NORM ) {

			real *lastZ = z + zOffsets[ c + 1 ] + l.batchNormOffset; 

			real *modifiedZ, *batchScratch; 

			if (c == (model->cLayers - 1)){

				modifiedZ = nextDevPtr; 
				batchScratch = modifiedZ + p.outHeight * p.outWidth * l.outChannels * n; 

				reshapeMatrix( lastZ, l.outChannels, n, p.outHeight * p.outWidth, batchScratch); 

				int numElements = l.outChannels * p.outHeight * p.outWidth * n; 
				int transposeBlocks = (BLOCK_SIZE - 1 + numElements) / BLOCK_SIZE; 
				ker_transpose <<< transposeBlocks, BLOCK_SIZE >>> 
					(batchScratch, numElements, l.outChannels , p.outHeight , p.outWidth , n, modifiedZ); 
				cudaDeviceSynchronize (); 
				cudaCheckError (); 	

				lastZ = modifiedZ; 
			} else {
				batchScratch = nextDevPtr; 
			}

			// Store the error terms here... which can be used later during
			// Hessian vector product.
			copy_device( dx + zOffsets[ c + 1 ] + l.batchNormOffset, 
					delta, sizeof(real) * p.outHeight * p.outWidth * l.outChannels * n, 
					ERROR_MEMCPY_DEVICE_DEVICE ); 

			computeBatchNormDerivative( delta, p.outHeight, p.outWidth, l.outChannels, n, 
				z + zOffsets[ c + 1 ] + l.batchNormOffset + l.meansOffset, 
				z + zOffsets[ c + 1 ] + l.batchNormOffset + l.variancesOffset, 
				delta_new, 
				//z + zOffsets[ c + 1 ] + l.batchNormOffset, 
				lastZ,
				//BATCH_NORM_EPSILON, nextDevPtr, dx + zOffsets[ c + 1 ] + l.batchNormOffset, 
				BATCH_NORM_EPSILON, batchScratch, dx + zOffsets[ c + 1 ] + l.batchNormOffset, 
				hostPtr ); 

#ifdef DEBUG_BATCH_NORM
fprintf( stderr, "Derivative of Batch Norm .... \n"); 
copy_host_device( hostPtr, delta_new, sizeof(real) * l.batchNormVolumn * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, p.outHeight * p.outWidth * n, l.outChannels ); 
#endif

				//TODO for now... copy back the updated results to delta_new from delta. 
				copy_device( delta, delta_new, sizeof(real) * l.batchNormVolumn * n, 
									ERROR_MEMCPY_DEVICE_DEVICE ); 
		}

*/

/*
cublasCheckError( cublasDnrm2( cublasHandle, p.outHeight * p.outWidth * n * l.outChannels, 
											delta, 1, hostPtr ) ); 
fprintf( stderr, "Incoming dx norm for layer: %d --> %e \n", c, *hostPtr ); 

cublasCheckError( cublasDnrm2( cublasHandle, p.outHeight * p.outWidth, delta, 1, hostPtr ) ); 
fprintf( stderr, "Norm of first image before Pooling.... is: %.10f \n", *hostPtr ); 

if( c == 0) {
	copy_host_device( hostPtr, delta, sizeof(real) * p.outHeight * p.outWidth, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
	print2DMatrix(hostPtr, p.outHeight, p.outWidth ); 
}
*/

		// Pool Layer Processing
		switch( p.type ) {
			case MAX_POOL: 
				//source --> img ( outChannels x height x width )
				//dest   --> img( inChannels x height x width )	
				// max pool derivative is source location with max value gets 1
				// others get zero. 
				// for this we need the max indices here. 
				//fprintf( stderr, " MAX_POOL is NOT IMPLEMENTED YET... \n" ); 
				//exit ( -1 ); 
#ifdef DEBUG_DETAILED
fprintf( stderr, "Convolution Pooling Backward Preparation (z_in).... \n"); 
copy_host_device( hostPtr, z + zOffsets[ c + 1 ] + l.activationOffset, sizeof(real) * n * l.outChannels * p.height * p.width, 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, p.height * p.width * n, l.outChannels ); 
#endif

				/*
				computeMaxPoolDerivative (delta, z + zOffsets[ c + 1 ] + l.activationOffset, p.outHeight, p.height, l.outChannels, 
					delta_new, p.pSize, p.stride, p.padding, n ); 
				*/
				computeMaxPoolDerivative( delta, z + zOffsets[ c + 1 ] + l.activationOffset, l.outChannels, 
					p.height, p.width, p.pSize, p.stride, p.padding, 
					p.outHeight, p.outWidth, n, delta_new ); 
				break; 
			
			case AVG_POOL: 
				/*
					p.pSize  -- Kernel Size
					p.height -- inHeight
					p.width  -- inWidth
				*/
				// avg pool: error is multiplied with 1 / (psizexpsize). 
				/*
   			p_height = ( p.height - p.pSize ) / p.pSize + 1;  
   			p_width = ( p.width - p.pSize ) / p.pSize + 1;  
				count = ( p_height * p_width ) * n * l.outChannels; 
				*/
				count = n * l.poolVolumn; 

				alpha = 1./( p.pSize * p.pSize );
				cublasCheckError( cublasDscal ( cublasHandle, count, &alpha, delta, 1 ) ); 

				//map from destination to source images here
				// dest ( outChannels * l.height * l.width ), 
				//source( l.height * l.width * outChannels )
				// This increases the size by the reduction in the forward pass. 
				computePoolDerivative( delta, p.outHeight , l.outChannels, 
					delta_new, p.pSize, n ); 
	
				break;

			case NO_POOL: 
				copy_device( delta_new, delta, sizeof(real) * n * l.activationVolumn, 
									ERROR_MEMCPY_DEVICE_DEVICE ); 
				break;

			default: 
				fprintf( stderr, "Unknown Pool Function... \n"); 
				exit ( -1 ); 
		} 

#ifdef DEBUG_DETAILED
fprintf( stderr, "Convolution Pooling Backward.... \n"); 
if (p.type != NO_POOL) {
copy_host_device( hostPtr, delta_new, sizeof(real) * n * l.activationVolumn, 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, p.height * p.width * n, l.outChannels ); 
} else {
copy_host_device( hostPtr, delta_new, sizeof(real) * n * l.activationVolumn, 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.outHeight* l.outWidth * n, l.outChannels ); 
}
fprintf( stderr, "BackwardPass... ConvLayer Done with Pooling\n"); 
#endif

#ifdef DEBUG_CNN
fprintf( stderr, "Convolution Pooling storing the dx_P .... \n"); 
#endif

copy_device( dx + zOffsets[ c + 1 ], delta_new, sizeof(real) * n * l.activationVolumn, 
	ERROR_MEMCPY_DEVICE_DEVICE ); 

/*
cublasCheckError( cublasDnrm2( cublasHandle, p.height * p.width, delta_new, 1, hostPtr ) ); 
fprintf( stderr, "Norm of first image before activation.... is: %.10f \n", *hostPtr ); 
*/

		// the activation layer... 
		// hadamard product with the deltas... 
		switch (model->actFuns[ c ] ) {
			case CNN_ACT_SOFTPLUS: 

/*
if( c == 0) {
	fprintf( stderr, "inputs to activation layer... \n"); 

copy_host_device( hostPtr, delta_new, sizeof(real) * 10, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
fprintf( stderr, "delta input is ... \n"); 
print2DMatrix( hostPtr, 1, 10 ); 

copy_host_device( hostPtr, z + zOffsets[ c + 1], sizeof(real) * 10, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
fprintf( stderr, "z input is ... \n"); 
print2DMatrix( hostPtr, 1, 10 ); 
}
*/

				// derivative is Sigmoid 1/ 1 + exp(-x)
				count = n * l.activationVolumn; 
				blocks = ( count + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 

				// takes in the result of img2col * wt . 
/*
				kerNNBackPropSOFTPLUS <<< blocks, BLOCK_SIZE >>> 
					( delta_new, z + zOffsets[ c + 1 ], count );
*/
				if (l.batchNorm != PERFORM_NO_BATCH_NORM) {
					kerNNBackPropSOFTPLUS <<< blocks, BLOCK_SIZE >>> 
						( delta_new, z + zOffsets[ c + 1 ] + l.batchNormOffset, count );
				} else { 
					kerNNBackPropSOFTPLUS <<< blocks, BLOCK_SIZE >>> 
						( delta_new, z + zOffsets[ c + 1 ], count );
				}

				cudaThreadSynchronize (); 
				cudaCheckError (); 
				break; 

			case CNN_ACT_RELU: 
				break; 

			case CNN_ACT_ELU: 
				break; 

			case CNN_ACT_SWISH: 
				count = n * l.activationVolumn;
				blocks = ( count + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 

/*
				kerNNBackPropSwish <<< blocks, BLOCK_SIZE >>> 
					( 	z + zOffsets[ c + 1 ], 
						z + zOffsets[ c + 1 ] + l.activationOffset, 
						delta_new, count ); 
*/
				kerNNBackPropSwish <<< blocks, BLOCK_SIZE >>> 
					( 	((l.batchNorm != PERFORM_NO_BATCH_NORM) ? 
							(z + zOffsets[ c + 1 ] + l.batchNormOffset) : (z + zOffsets[ c + 1])), 
						z + zOffsets[ c + 1 ] + l.activationOffset, 
						delta_new, count ); 
				cudaThreadSynchronize (); 
				cudaCheckError (); 
				break;

			case CNN_ACT_NONE: 
				break;
		
			default: 
				fprintf( stderr, "Unknown actication function... while storing dx_p \n"); 
				exit( -1 ); 
		}

#ifdef DEBUG_DETAILED
fprintf( stderr, "Convolution Activation Backward.... \n"); 
copy_host_device( hostPtr, delta_new, sizeof(real) * n * l.outChannels * p.height * p.width, 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, p.height * p.width * n, l.outChannels ); 
#endif

#ifdef DEBUG_CNN
fprintf( stderr, "Convolution Activation... storing dx_A ... \n"); 
#endif 

copy_device( dx + zOffsets[ c + 1 ] + ( l.activationOffset ), delta_new, 
	sizeof(real) * n * l.activationVolumn, ERROR_MEMCPY_DEVICE_DEVICE ); 


/*
	if (CNN_ACT_NONE == model->actFuns[ c ]){
		fprintf( stderr, "Skipping the convolution part for this laery... \n"); 
		copy_device( delta, delta_new, sizeof(real) * n * l.outChannels * p.height * p.width, 
							ERROR_MEMCPY_DEVICE_DEVICE ); 
		continue; 
	}
*/


/*

cublasCheckError( cublasDnrm2( cublasHandle, p.height * p.width, delta_new, 1, hostPtr ) ); 
fprintf( stderr, "Norm of first image after activation.... is: %.10f \n", *hostPtr ); 

if( c == 0) {
	copy_host_device( hostPtr, delta_new, sizeof(real) * p.height * p.width, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
	print2DMatrix(hostPtr, p.height, p.width ); 
}

*/



/*
fprintf( stderr, "Storing the dx_a back prop results ... \n"); 
copy_host_device( hostPtr, delta_new, sizeof(real) * l.outHeight * l.outWidth * l.outChannels * n, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.outHeight * l.outWidth * n, l.outChannels );
*/

#ifdef DEBUG_CNN
fprintf( stderr, "BackwardPass... ConvLayer done with Activation\n"); 
fprintf( stderr, "BackwardPass... ConvLayer starting batch normalization...\n" ); 
#endif
		if (l.batchNorm != PERFORM_NO_BATCH_NORM ) {

			computeBatchNormDerivative( delta_new, l.outHeight, l.outWidth, l.outChannels, n, 
				z + zOffsets[ c + 1 ] + l.batchNormOffset + l.meansOffset, 
				z + zOffsets[ c + 1 ] + l.batchNormOffset + l.variancesOffset, 
				delta, 
				z + zOffsets[ c + 1 ] + l.batchNormOffset, 
				BATCH_NORM_EPSILON, nextDevPtr, dx + zOffsets[ c + 1 ] + l.batchNormOffset, 
				hostPtr ); 

			copy_device( dx + zOffsets[ c + 1 ] + l.batchNormOffset, 
					delta, sizeof(real) * l.outHeight * l.outWidth * l.outChannels * n, 
					ERROR_MEMCPY_DEVICE_DEVICE ); 

			copy_device( delta_new, delta, sizeof(real) * l.activationVolumn * n, 
							ERROR_MEMCPY_DEVICE_DEVICE ); 	
		}


		if( model->bias != 0) {

			//db -- BEGIN
			// Now compute dW and db for Convolution function here. 
			// db - sum( delta, 2) - sum the weights for each channel... 
			// sum along all points... and the sum all in each filter. 

			// samplepoints wise and channel-wise as the input. 
			// one_vector ( p.height * p.width * n X 1) * errors( p.height * p.width * n X channels ); 
			blocks = ( p.height * p.width * n + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 
			kerInitOneVector <<< blocks, BLOCK_SIZE >>> 	
				(nextDevPtr, p.height * p.width * n); 
			cudaThreadSynchronize (); 
			cudaCheckError (); 

/*
fprintf( stderr, "One Vector and height/width of pool layer input (%d, %d, %d) \n", p.height, p.width, p.height * p.width * n ); 
copy_host_device (hostPtr, nextDevPtr, sizeof(real) * p.height * p.width * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, 1, p.height * p.width * n ); 
*/

			alpha = 1.0; 
			cublasCheckError ( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
										1, l.outChannels, p.height * p.width * n, 
										&alpha, nextDevPtr, 1, 
										delta_new, p.height * p.width * n, 
										&beta, gradient + bOffsets[ c ], 1 ) ); 
		/*
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
										l.outChannels, 1, p.height * p.width * n, 
										&alpha, delta_new, p.height * p.width * n, 
										nextDevPtr, p.height * p.width * n, 
										&beta, gradient + bOffsets[ c ], l.outChannels ) ); 
		*/
		/*
		cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, 
										//p.height * p.width * n, l.outChannels, 
										l.outChannels, p.height * p.width * n, 
										&alpha, delta_new, l.outChannels, 
										nextDevPtr, 1, &beta, gradient + bOffsets[ c ], 1 ) ); 
		*/


#ifdef DEBUG_DETAILED
fprintf( stderr, "Convolution db.... \n"); 
copy_host_device( hostPtr, gradient + bOffsets[ c ], sizeof(real) * l.outChannels, 
		cudaMemcpyDeviceToHost, 	ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, 1, l.outChannels ); 
#endif

/*

fprintf( stderr, "Manual computation here... \n"); 
copy_host_device( hostPtr, delta_new, sizeof(real) * p.height * p.width * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 

real sum = 0; 
for (int x = 0; x < p.height * p.width * n; x ++)
	sum += hostPtr[ x ]; 	

fprintf( stderr, "db using manual is ... %3.4e \n", sum); 

fprintf( stderr, "Testing asum here... \n"); 
		cublasCheckError( cublasDasum( cublasHandle, 1,
												delta_new, 1, hostPtr) ); 
fprintf( stderr, "Testing asum here... \n"); 
		//copy_host_device( hostPtr, nextDevPtr, sizeof(real), 
		//	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
fprintf( stderr, "Testing asum here... \n"); 
fprintf( stderr, "Summing result: %3.4e \n", hostPtr ); 

*/

#ifdef DEBUG_CNN
fprintf (stderr, "BackwardPass... Done with db\n"); 
#endif


/*
cublasCheckError( cublasDnrm2( cublasHandle, l.outChannels, gradient + bOffsets[ c ], 1, hostPtr ) ); 
fprintf( stderr, "dx-b norm is : %e \n", *hostPtr ); 

fprintf( stderr, "Bias output for this layer: %d is : \n", c ); 
copy_host_device( hostPtr, gradient + bOffsets[ c ], sizeof(real) * l.outChannels, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, 1, l.outChannels ); 
*/



		}
		//db - END
		
		//
		// dW - z_left^T * delta
		//
		//if (c == 0) dataset = data->trainSetX + offset * data->features; 
		if (c == 0) dataset = data->currentBatch;
		else {
			// pool Output from the prev. layer
			CONV_LAYER prevLayer = model->convLayer[ c-1 ]; 
			dataset = z + zOffsets[ c ] + prevLayer.outputOffset;  
		}
	
		// Img2Col conversion of z
		//TODO
		//TODO
		//getBatchImageCols( dataset, n, 
		//		l.outChannels, l.height, l.width, l.kSize, l.padding, l.stride, nextDevPtr ); 
		getBatchImageCols( dataset, n, 
				l.inChannels, l.height, l.width, l.kSize, l.padding, l.stride, nextDevPtr ); 
		//TODO
		//TODO

	
		// dW = img2col( z )' * delta
		alpha = 1.; 
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
				l.kSize * l.kSize * l.inChannels, l.outChannels, p.height * p.width * n, 
				&alpha, nextDevPtr, p.height * p.width * n, 
				delta_new, p.height * p.width * n, 
				&beta, gradient + wOffsets[ c ], l.kSize * l.kSize * l.inChannels ) ); 

/*
cublasCheckError( cublasDnrm2( cublasHandle, l.kSize * l.kSize * l.outChannels * l.inChannels, gradient + wOffsets[ c ], 1, hostPtr ) ); 
fprintf( stderr, "dx norm is : %e \n", *hostPtr ); 
*/

#ifdef DEBUG_DETAILED
fprintf( stderr, "convolution dW.... \n"); 
copy_host_device( hostPtr, gradient + wOffsets[ c ], sizeof(real) * l.kSize * l.kSize  * l.inChannels * l.outChannels, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
//print4DMatrix( hostPtr, l.outChannels, l.inChannels, l.kSize, l.kSize ); 
print2DMatrix( hostPtr, l.kSize * l.kSize * l.inChannels, l.outChannels ); 
#endif

#ifdef DEBUG_CNN
fprintf (stderr, "BackwardPass... Done with dW\n"); 
#endif

		// update delta here. 
		// delta_new = Img2Col( delta ) * (Weights, reordered) here. 
		if (c != 0) {

			// store the delta_new to some place... for the hessian vector product later. 
			//TODO
			//TODO SUDHIR-DOUBLE-CHECK
			//TODO
			/*
			copy_device( dx + zOffsets[ c ], delta_new, 
							sizeof(real) * l.outChannels * p.height * p.width * n, 
							ERROR_MEMCPY_DEVICE_DEVICE ); 
			*/
			//

			backpropConvolution( delta_new, p.height, p.width, l.outChannels,
					weights + wOffsets[ c ], l.kSize, l.kSize, 
					l.height, l.width, l.padding, l.inChannels, 
					n, delta, nextDevPtr, hostPtr); 


#ifdef DEBUG_CNN
fprintf (stderr, "BackwardPass... Done with delta\n"); 
copy_host_device( hostPtr, delta, sizeof(real) * l.inChannels * l.height * l.width * n, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.height * l.width * n, l.inChannels ); 
#endif

/*
cublasCheckError( cublasDnrm2( cublasHandle, l.height * l.width * n * l.inChannels , delta, 1, 
											hostPtr ) ); 
fprintf( stderr, "outgoing dx is : %e\n", *hostPtr ); 
*/



			// store the delta_new to some place... for the hessian vector product later. 
			/*
			copy_device( dx + zOffsets[ c ], delta_new, 
								sizeof(real) * l.inChannels * l.height * l.width * n, 
								ERROR_MEMCPY_DEVICE_DEVICE ); 
			*/
		}


		//TODO -- NOT NEEDED ANYMORE for swapping
	
		//swap the pointers... 
		// delta_new, delta. 
		//temp = delta_new; 
		//delta_new = delta; 
		//delta = temp; 
	}

	// scale appropriately here. 
	//alpha = 1./(real)batchSize; 
	//cublasCheckError( cublasDscal( cublasHandle, model->pSize, &alpha, gradient, 1 )); 
}
