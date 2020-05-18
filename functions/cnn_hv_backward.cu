
#include <functions/cnn_hv_backward.h>

#include <functions/dev_layer_error.h>
#include <functions/dev_layer_r_error.h>
#include <functions/dev_initializations.h>
#include <functions/dev_pool.h>
#include <functions/dev_image.h>
#include <functions/dev_backprop_convolution.h>
#include <functions/dev_mat_mat_scale.h>
#include <functions/swish.h>
#include <functions/dev_batch_norm.h>
#include <functions/dev_transpose.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

#include <utilities/reduce.h>
#include <utilities/print_utils.h>

#include <core/errors.h>

long cnnROpBackwardMemRequired( CNN_MODEL *model ){

	long lRequired = 0, cRequired = 0;
	long t1, t2, t3; 

	for (int f = model->lLayers - 1; f >= 0; f -- ){
		FC_LAYER l = model->fcLayer[ f ]; 

		t1 = l.in * model->batchSize;
		if (lRequired < t1) lRequired = t1;

		// delta * R{ z }'
		t2 = l.out * l.in;
		if (lRequired < t2) lRequired = t2; 

		//delta = W * delta
		t3 = l.in * model->batchSize;
		if (lRequired < t3) lRequired = t3; 
	}

	//Reshape is insignificant compared to these two terms...

	for (int c = model->cLayers - 1; c >= 0; c -- ) {
      CONV_LAYER l = model->convLayer[ c ];
      POOL_LAYER p = model->poolLayer[ c ];	

		//pool derivative
		t1 = p.height * p.width * l.outChannels * model->batchSize;
		if (cRequired < t1) cRequired = t1;

		//RdW
		//imgCol of Rz
		t2 = (l.kSize * l.kSize * l.inChannels * p.height * p.width * model->batchSize) + 
					l.inChannels * l.outChannels * l.kSize * l.kSize;
		if (cRequired < t2) cRequired = t2;

		//R{ error }
		t3 = (l.height * l.width * l.inChannels * model->batchSize) + 
			(l.height * l.width * model->batchSize * l.outChannels * l.kSize * l.kSize);
		if (cRequired < t3) cRequired = t3;
	}

	return (( lRequired < cRequired ) ? cRequired : lRequired );

}

void cnnROpBackward( CNN_MODEL *model, DEVICE_DATASET *data, SCRATCH_AREA *scratch, 
		real *z, real *dx, real *lossFuncErrors, real *rError, real *rz,
		real *vector, real *hv, 
		int s, int curBatchSize, 
		real *devPtr, real *hostPtr){

	/*
   real *nextDevPtr = scratch->nextDevPtr;
   real *nextPagePtr = scratch->nextPageLckPtr;
   real *nextHostPtr = scratch->nextHostPtr;
	*/
	real *nextDevPtr = devPtr; 
	real *nextHostPtr = hostPtr; 
	real *nextDevPtr2; 
	real *temp;

   real *weights = data->weights;
   int *wOffsets = model->wOffsets;
   int *bOffsets = model->bOffsets;
   int *zOffsets = model->zOffsets;

	if (model->bias == 0)
		bOffsets = NULL; 

	int cLayers = model->cLayers; 
	int lLayers = model->lLayers; 
	int n = curBatchSize; 
	int count, blocks; 

	int outputOffset = 0; 
	int p_height, p_width, col_height, col_width; 
	POOL_LAYER *pLayer; 
	CONV_LAYER *cLayer; 

	real alpha, beta; 

	
	// moving backwards... 
	// The very last linear layer here. 

	/*
		dW = delta * z^T
		rdW = R{ delta * z^T }
			 = R{delta} * z^T + delta * R{z}^T

		db = sum( delta, 2 )
		Rdb = R{ sum( delta, 2) }
			 = sum( R{ delta }, 2 )

		delta = W * delta
		Rdelta = R{ W * delta }
				 = R{ W } * delta + W * R{ delta }
				 = VW * delta + W * R{ delta }
	*/

#ifdef DEBUG_ROP
fprintf( stderr, "... Beginging with ROp Backward Pass.... \n\n"); 
copy_host_device( hostPtr, rError, sizeof(real) * n * data->numClasses, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, data->numClasses, n );
#endif

/*
	if (f == 0) {
   	pLayer = &( model->poolLayer[ model->cLayers - 1 ] );  
   	cLayer = &( model->convLayer[ model->cLayers - 1 ] );    
   	p_height = ( pLayer->height - pLayer->pSize ) / pLayer->pSize+ 1;  
   	p_width = ( pLayer->width - pLayer->pSize) / pLayer->pSize + 1;  
   	col_height = (cLayer->height + 2 * cLayer->padding - cLayer->kSize ) / cLayer->stride + 1;  
   	col_width = (cLayer->width + 2 * cLayer->padding - cLayer->kSize ) / cLayer->stride + 1;  

   	poolOffset = 2 * col_height * col_width * cLayer->outChannels * curBatchSize; 
	}
*/


   cLayer = &( model->convLayer[ model->cLayers - 1 ] );    

	// SUDHIR-CHANGES-DOUBLE-CHECK
	//poolOffset = 2 * cLayer->convOffset; 
	// SUDHIR-CHANGES-DOUBLE-CHECK
	outputOffset = cLayer->outputOffset; 

	alpha = 1; beta = 0; 
	for (int f = lLayers - 1; f >= 0; f -- ){
		FC_LAYER l = model->fcLayer[ f ]; 

		if (f == 0) outputOffset = cLayer->outputOffset; 
		else outputOffset = 0; 

		//update Rdx here. 
		switch( l.actFun ){
			// delta = h'(z) .* delta
			// R{ delta } = R{ h'(z) .* delta }
			// 			  = R{ h'(z) } .* delta + h'(z) .* R{ delta }
			//				  = h''(z) .* R{ z } .* delta + h'(z) .* R{ delta }
			case CNN_ACT_SOFTPLUS: 
				count = ( l.out * n	+ BLOCK_SIZE - 1 ) / BLOCK_SIZE; 

#ifdef DEBUG_ROP
fprintf( stderr, "... dx(%d)... \n", f+1 ); 
copy_host_device( hostPtr, 
			((f == (lLayers - 1)) ? (lossFuncErrors) : (dx + zOffsets[ cLayers + f + 1 ] )) , 
			sizeof(real) * n * l.out, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, n ); 

fprintf( stderr, "... Z(%d)... \n", f+1); 
copy_host_device( hostPtr, z + zOffsets[ cLayers + f + 1 ], 
			sizeof(real) * n * l.out, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, n ); 

fprintf( stderr, "... RZ(%d)... \n", f+1); 
copy_host_device( hostPtr, rz + zOffsets[ cLayers + f + 1 ] + l.offset, 
			sizeof(real) * n * l.out, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, n ); 
#endif

				// h'(Z) .* R{ delta }
				//kerNNROpSOFTPLUS <<< count, BLOCK_SIZE >>> 
				//	( rError, z + zOffsets[ cLayers + f + 1 ], l.out * n); 
				
				// Now with modification, it takes x, instead of f(x)
				kerNNROpSOFTPLUS <<< count, BLOCK_SIZE >>> 
					( rError, z + zOffsets[ cLayers + f + 1 ] + l.offset, l.out * n ); 
				cudaThreadSynchronize ();
				cudaCheckError (); 

				//use the correct dx term here. if it is not the very last layer...
				// dx = w' * dx_A
				if (f != (lLayers - 1)) {
					FC_LAYER ll = model->fcLayer[ f + 1 ]; 
					alpha = 1.; beta = 0.; 
					cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
											ll.in, n, ll.out, 
											&alpha, weights + wOffsets[ cLayers + f + 1 ], ll.out, 
											dx + zOffsets[ cLayers + f + 1 + 1 ], ll.out, 
											&beta, nextDevPtr, ll.in ) ); 
				}

				// h''(z) .* R{ z } .* delta
				kerNNROpSOFTPLUSWithZ <<< count, BLOCK_SIZE >>>
					( rError, 
					((f == (lLayers - 1)) ? (lossFuncErrors) : (nextDevPtr)), 
					rz + zOffsets[ cLayers + f + 1] + l.offset , 
					z + zOffsets[ cLayers + f + 1] + l.offset , l.out * n); 
				cudaThreadSynchronize ();
				cudaCheckError (); 

				break;

			case CNN_ACT_SWISH: 
				count = (l.out * n + BLOCK_SIZE - 1) / BLOCK_SIZE; 

				//h'(Z) .* R{ delta }
				kerNNBackPropSwish <<< count, BLOCK_SIZE >>> 
					( z + zOffsets[ cLayers + f + 1 ] + l.offset, z + zOffsets[ cLayers + f + 1 ], 
						rError, l.out * n); 
				cudaThreadSynchronize ();
				cudaCheckError (); 

				//h''(z) .* R{ z } .* delta
				//h''(z)
				kerNNSecondDerivSwish <<< count, BLOCK_SIZE >>> 
					( z + zOffsets[ cLayers + f + 1 ] + l.offset, z + zOffsets[ cLayers + f + 1], 
						dx + zOffsets[ cLayers + f + 1 ], nextDevPtr, l.out * n); 	
				cudaThreadSynchronize ();
				cudaCheckError (); 

				kerUtilsMatMatScale <<< count, BLOCK_SIZE >>> 
					( nextDevPtr, rz + zOffsets[ cLayers + f + 1 ] + l.offset, 
						l.out * n, nextDevPtr ); 
				cudaThreadSynchronize ();
				cudaCheckError (); 

				nextDevPtr2 = NULL;
				if (f != (lLayers - 1)) {
					nextDevPtr2 = nextDevPtr + l.out * n; 
					/*
					alpha = 1.; 
					cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
											l.in, n, l.out, 
											&alpha, weights + wOffsets[ cLayers + f ], l.out, 
											dx + zOffsets[ cLayers + f + 1 ], l.out, 
											&beta, nextDevPtr2, l.in ) ); 
					*/

					FC_LAYER ll = model->fcLayer[ f + 1 ]; 
					alpha = 1.; beta = 0.; 
					cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
											ll.in, n, ll.out, 
											&alpha, weights + wOffsets[ cLayers + f + 1 ], ll.out, 
											dx + zOffsets[ cLayers + f + 1 + 1 ], ll.out, 
											&beta, nextDevPtr2, ll.in ) ); 
				}

				temp = (f == (lLayers - 1)) ? (lossFuncErrors) : (nextDevPtr2);	
				kerUtilsMatMatScale <<< count, BLOCK_SIZE >>> 
					( nextDevPtr, temp, l.out * n, nextDevPtr ); 
				cudaThreadSynchronize ();
				cudaCheckError (); 

				//Add the two terms
				// h'(z) .* R{ delta } + h''(z) .* R{ z } .* delta
				alpha = 1.; 
				cublasCheckError( cublasDaxpy( cublasHandle, l.out * n, 
											&alpha, nextDevPtr, 1, 
											rError, 1 ) ); 

				break;

			case CNN_ACT_NONE: 
				break;

			default: 
				fprintf( stderr, "Undefined Activation function... HV(backward)... \n"); 
				exit( -1 ); 
		}
#ifdef DEBUG_ROP
fprintf( stderr, "...Done with ROp Backward Linear Layer Activation (rError): %d \n", f ); 
copy_host_device( hostPtr, rError, sizeof(real) * l.out * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, n ); 
#endif


		//dW = delta * z'
		//RdW = R{ delta * z' } = R{ delta } * z' + delta * R{ z }'
		// remember 
		// delta from the next layer
		// z from from the previous layer... 

#ifdef DEBUG_ROP
fprintf( stderr, " ... Rdx ... \n");
copy_host_device( hostPtr, rError, sizeof(real) * l.out * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, n ); 

fprintf( stderr, "... RZ... \n" ); 
copy_host_device( hostPtr, rz + zOffsets[ cLayers + f ] + outputOffset, sizeof(real) * l.in * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.in, n ); 

fprintf( stderr, "... dx'... \n" ); 
copy_host_device( hostPtr, dx + zOffsets[ cLayers + f + 1 ], sizeof(real) * l.out * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, n ); 
#endif

		// R{ delta } * z'
		alpha = 1.; beta = 0.; 
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
									l.out, l.in, n, 
									&alpha,  rError, l.out, 
									z + zOffsets[ cLayers + f ] + outputOffset, l.in, 
									&beta, hv + wOffsets[ cLayers + f ], l.out ) );

		// delta * R{ z }'
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
									l.out, l.in, n, 
									&alpha, dx + zOffsets[ cLayers + f + 1], l.out, 
									rz + zOffsets[ cLayers + f ] + outputOffset, l.in, 
									&beta, nextDevPtr, l.out ) ); 

		//Add two matrices. 
		cublasCheckError( cublasDaxpy( cublasHandle, l.out * l.in, 
									&alpha, nextDevPtr, 1, 
									hv + wOffsets[ cLayers + f ], 1 ) ); 
						
#ifdef DEBUG_ROP
fprintf( stderr, "...Done with ROp Backward Linear Layer *dW*: %d\n", f ); 
copy_host_device( hostPtr, hv + wOffsets[ cLayers + f ], 
	sizeof(real) * l.out * l.in, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, l.in ); 
#endif
		
		//db = sum( delta, 2 )
		//Rdb = sum( R{ delta }, 2 )

		if (model->bias != 0) {
			count = ( n + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 
			kerInitOneVector <<< count, BLOCK_SIZE >>> 
				( nextDevPtr, n ); 
			cudaThreadSynchronize (); 	
			cudaCheckError (); 

			alpha = 1.; 
			cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, 	
								l.out, n, &alpha, rError, l.out, 
								nextDevPtr, 1, &beta, hv + bOffsets[ cLayers + f ], 1 ) ); 

#ifdef DEBUG_ROP
fprintf( stderr, "...Done with ROp Backward Linear Layer *db*: %d\n", f ); 
copy_host_device( hostPtr, hv + bOffsets[ cLayers + f ], 
	sizeof(real) * l.out, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.out, 1 ); 
#endif
		}

		/*
		delta = W * delta
		Rdelta = R{ W' * delta }
				 = R{ W }' * delta + W' * R{ delta }
				 = VW' * delta + W' * R{ delta }
		*/
		alpha = 1.; 
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
								l.in, n, l.out, 
								&alpha, weights + wOffsets[ cLayers + f ], l.out, 
								rError, l.out, &beta, nextDevPtr, l.in ) ); 

		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
								l.in, n, l.out, 
								&alpha, vector + wOffsets[ cLayers + f ], l.out, 
								dx + zOffsets[ cLayers + f + 1 ], l.out, 
								&beta, rError, l.in ) ); 

		cublasCheckError( cublasDaxpy( cublasHandle, l.in * n, 
								&alpha, nextDevPtr, 1, rError, 1 ) ); 

#ifdef DEBUG_ROP
fprintf( stderr, "...Done with ROp Backward Linear Layer *dError*: %d\n", f ); 
copy_host_device( hostPtr, rError, sizeof(real) * l.in * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.in, n ); 
#endif

	}

	//Convolution Layer starts here. 
	//Complete the backprop (ROp) to get the HessianVec... Hay !!!

	//re-shape R{delta} 
	// from h * w * c X n to h * w * n X c
   pLayer = &( model->poolLayer[ model->cLayers - 1 ] );  
   cLayer = &( model->convLayer[ model->cLayers - 1 ] );    
/*
   p_height = ( pLayer->height - pLayer->pSize ) / pLayer->pSize+ 1;  
   p_width = ( pLayer->width - pLayer->pSize) / pLayer->pSize + 1;  
   col_height = (cLayer->height + 2 * cLayer->padding - cLayer->kSize ) / cLayer->stride + 1;  
   col_width = (cLayer->width + 2 * cLayer->padding - cLayer->kSize ) / cLayer->stride + 1;  

   poolOffset = 2 * col_height * col_width * cLayer->outChannels * curBatchSize; 
*/

	//SK-1
	reshapeMatrix( rError, n, cLayer->outChannels, pLayer->outHeight * pLayer->outWidth, nextDevPtr ); 

/*
	copy_device( rError, nextDevPtr, sizeof(real) * n * cLayer->poolVolumn, 
						ERROR_MEMCPY_DEVICE_HOST ); 
*/

   //SK-2 Commented out the above because of transpose below... 
   int transElements = cLayer->outChannels * pLayer->outHeight * pLayer->outWidth * n; 
   int transBlocks = (BLOCK_SIZE - 1 + transElements) / BLOCK_SIZE; 
   ker_transpose <<< transBlocks, BLOCK_SIZE >>> 
      ( nextDevPtr, transElements, cLayer->outChannels, pLayer->outHeight, pLayer->outWidth, n, rError ); 
   cudaDeviceSynchronize (); 
   cudaCheckError (); 

#ifdef DEBUG_ROP
fprintf( stderr, "... Done with Reshaping of the eRrror Matrix... \n\n"); 
copy_host_device( hostPtr, rError, sizeof(real) *  n * cLayer->poolVolumn, 	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, pLayer->outHeight * pLayer->outWidth * n, cLayer->outChannels ); 
fprintf( stderr, "... Now beginning the ROp Backward Pass with the Conv. Layers... \n\n"); 
#endif

	for (int c = cLayers - 1; c >= 0; c -- ) {
      CONV_LAYER l = model->convLayer[ c ];
      POOL_LAYER p = model->poolLayer[ c ];	

/*
		//Batch Normalization here. 
		if (l.batchNorm != PERFORM_NO_BATCH_NORM){

			real *zOut = z + zOffsets[ c + 1 ] + l.batchNormOffset;  
			real *rzOut = rz + zOffsets[ c + 1 ] + l.batchNormOffset;  
			real *devScratch = nextDevPtr; 

			if (c == (cLayers - 1) ){

				zOut = nextDevPtr;
				rzOut = zOut + l.outChannels * p.outHeight * p.outWidth * n; 
				devScratch = rzOut + l.outChannels * p.outHeight * p.outWidth * n; 

				//rzOut processing
				reshapeMatrix( rz + zOffsets[ c + 1] + l.batchNormOffset, 
					l.outChannels, n, p.outHeight * p.outWidth, devScratch); 

   			int transElements = l.outChannels * p.outHeight * p.outWidth * n; 
   			int transBlocks = (BLOCK_SIZE - 1 + transElements) / BLOCK_SIZE; 
   			ker_transpose <<< transBlocks, BLOCK_SIZE >>> 
      			( devScratch, transElements, l.outChannels, p.outHeight, p.outWidth, n, rzOut); 
   			cudaDeviceSynchronize (); 
   			cudaCheckError (); 

				// zOut processing...
				reshapeMatrix( z + zOffsets[ c + 1 ] + l.batchNormOffset, 
					l.outChannels, n, p.outHeight * p.outWidth, devScratch); 

   			ker_transpose <<< transBlocks, BLOCK_SIZE >>> 
      			( devScratch, transElements, l.outChannels, p.outHeight, p.outWidth, n, zOut); 
   			cudaDeviceSynchronize (); 
   			cudaCheckError (); 
			}

			//update rError in the backward direction... 
			computeROpBatchNormBackward( z + zOffsets[ c + 1 ] + l.poolOffset, 
				zOut, 
				rz + zOffsets[ c + 1 ] + l.poolOffset, 
				rzOut, 
				dx + zOffsets[ c + 1 ] + l.batchNormOffset, 
				rError, 
				NULL, BATCH_NORM_EPSILON, 
				z + zOffsets[ c + 1 ] + l.batchNormOffset + l.meansOffset, 
				z + zOffsets[ c + 1 ] + l.batchNormOffset + l.variancesOffset, 
				p.outHeight, p.outWidth, l.outChannels, n, model->batchSize, 
				devScratch, hostPtr ); 
#ifdef DEBUG_ROP
fprintf( stderr, "Done with ROp-Conv-Act-Pool-BN ... \n"); 
copy_host_device( hostPtr, rError, sizeof(real) * l.poolVolumn * n,
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, p.outHeight, p.outWidth ); 
#endif

		}
*/

		//backwards Pool 
		switch( p.type ) { 
			case MAX_POOL: 

				computeMaxPoolDerivative( rError, z + zOffsets[ c + 1 ] + l.activationOffset, l.outChannels, 
						p.height, p.width, p.pSize, p.stride, p.padding, p.outHeight, p.outWidth, n, 
						nextDevPtr ); 

				/*
				computeMaxPoolDerivative( rError, z + zOffsets[ c + 1 ] + l.activationOffset, p.outHeight, p.height, 
						l.outChannels, nextDevPtr, p.pSize, p.stride, p.padding, n ); 
				*/
				copy_device( rError, nextDevPtr, sizeof(real) * l.activationVolumn * n, 
									ERROR_MEMCPY_DEVICE_DEVICE ); 
				break;

			case AVG_POOL: 

				/*
   			p_height = ( p.height - p.pSize ) / p.pSize+ 1;  
   			p_width = ( p.width - p.pSize) / p.pSize + 1;  
				*/

				
				//TODO 
				//TODO 
				/*
						THIS IS DONE TO MATCH PYTORCH'S IMPLEMENTATION...
						WHICH ONE IS RIGHT ????? 
				*/
				//TODO 
				//TODO 
				count = n * l.poolVolumn; 
				alpha = 1./( p.pSize * p.pSize ); 
				cublasCheckError( cublasDscal( cublasHandle, count, &alpha, rError, 1 ) ); 


				//update the rError here. 
				//This will increase the size of rError by the Pool Kernel Size
				computePoolDerivative( rError, p.outHeight, l.outChannels, nextDevPtr, p.pSize, n ); 
				copy_device( rError, nextDevPtr, sizeof(real) * l.activationVolumn * n, 
									ERROR_MEMCPY_DEVICE_DEVICE ); 
				//TODO 
				//TODO 
				//TODO 
				//TODO 
				break; 

			case NO_POOL: 
				break;



			default: 
				fprintf( stderr, "Undefined Pooling function in Convolution ROp Backward.... \n"); 
				exit (-1);
		}
#ifdef DEBUG_ROP
fprintf( stderr, "... Done with ROp Backward (Convolution - Pool): %d\n", c ); 
copy_host_device( hostPtr, rError, sizeof(real) * l.activationVolumn * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
//print2DMatrix( hostPtr, p.height * p.width * n, l.outChannels); 
print4DMatrix( hostPtr, n, l.outChannels, p.height, p.width ); 
#endif

		//backwards Activation. 
		/*
			R{z} = ImgColWt + actOut + PoolOut... 
		*/
		switch( model->actFuns[ c ] ){
			/*
				delta = h'(z) .* delta
				R{ delta } = R{ h'(z) } .* delta + h'(z) .* R{ delta }
				R{ delta } = h''(z) .* R{ z } .* delta + h'(z) .* R{ delta }
			*/
			case CNN_ACT_SOFTPLUS: 
				count = l.activationVolumn * n; 
				blocks = (count + BLOCK_SIZE - 1)/BLOCK_SIZE; 

				// h'(z) .* R{ delta }
				// z = input to the h(.) function here. 
				kerNNBackPropSOFTPLUS <<< blocks, BLOCK_SIZE >>> 
					//( rError, z + zOffsets[ c + 1 ], count ); 
					( rError, ((l.batchNorm != PERFORM_NO_BATCH_NORM) ? (z + zOffsets[ c + 1 ] + l.batchNormOffset) : (z + zOffsets[ c + 1 ])), count ); 
				cudaThreadSynchronize (); 
				cudaCheckError (); 

#ifdef DEBUG_ROP
fprintf( stderr, " h'(z) .* R{ delta } is --->\n "); 
copy_host_device( hostPtr, rError, sizeof(real) * l.outChannels * p.height * p.width * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, p.height * p.width * n, l.outChannels ); 

fprintf( stderr, "dx_P ---> \n" );
copy_host_device( hostPtr, dx + zOffsets[ c + 1], sizeof(real) * l.outChannels * p.height * p.width * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, p.height * p.width * n, l.outChannels ); 

fprintf( stderr, "rz ---- >\n"); 
copy_host_device( hostPtr, rz + zOffsets[ c + 1], sizeof(real) * l.outChannels * p.height * p.width * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, p.height * p.width * n, l.outChannels ); 

fprintf( stderr, "wz + b ---- > \n"); 
copy_host_device( hostPtr, z + zOffsets[ c + 1], sizeof(real) * l.outChannels * p.height * p.width * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, p.height * p.width * n, l.outChannels ); 
#endif

				// h''(z) .* R{ z } .* delta == (dx_P)
				kerNNROpSOFTPLUSWithZ <<< blocks, BLOCK_SIZE >>> 
/*
					( rError, dx + zOffsets[ c + 1 ], rz + zOffsets[ c + 1 ], // using (ImgCol * W)
					//z + zOffsets[ c + 1 ] + l.convOffset, count ); // using f( ImgCol * W + b) to compute derivative. 
					z + zOffsets[ c + 1 ], count ); // using ( ImgCol * W + b) to compute derivative. 
*/
					(rError, dx + zOffsets[ c + 1 ], 
						((l.batchNorm != PERFORM_NO_BATCH_NORM) ? (rz + zOffsets[ c + 1] + l.batchNormOffset) : ( rz + zOffsets[ c + 1] )), 
						((l.batchNorm != PERFORM_NO_BATCH_NORM) ? (z + zOffsets[ c + 1 ] + l.batchNormOffset) : (z + zOffsets[ c + 1] )), count ); 
				cudaThreadSynchronize (); 
				cudaCheckError (); 
				break;

			case CNN_ACT_SWISH: 
				count = l.activationVolumn * n; 
				blocks = (count + BLOCK_SIZE - 1)/BLOCK_SIZE; 

				//h'(z) .* R{ delta }
				kerNNBackPropSwish <<<blocks, BLOCK_SIZE >>> 
					( ((l.batchNorm != PERFORM_NO_BATCH_NORM) ? 
							(z + zOffsets[ c + 1 ] + l.batchNormOffset) : z + zOffsets[ c + 1]), 
						z + zOffsets[ c + 1 ] + l.activationOffset, rError, count ); 
				cudaThreadSynchronize (); 
				cudaCheckError (); 

				// h''(z) .* R{ z } .* delta
				kerNNSecondDerivSwish <<< blocks, BLOCK_SIZE >>> 
					( ((l.batchNorm != PERFORM_NO_BATCH_NORM) ? 
							(z + zOffsets[ c + 1 ] + l.batchNormOffset) : z + zOffsets[ c + 1]), 
						z + zOffsets[ c + 1] + l.activationOffset, 
						dx + zOffsets[ c + 1 ], nextDevPtr , count ); 	
				cudaThreadSynchronize (); 
				cudaCheckError (); 

				kerUtilsMatMatScale <<< blocks, BLOCK_SIZE >>> 
					( nextDevPtr , 
						((l.batchNorm != PERFORM_NO_BATCH_NORM) ? 
							(rz + zOffsets[ c + 1 ] + l.batchNormOffset) : (rz + zOffsets[ c + 1]) ), 
						count, nextDevPtr ); 
				cudaThreadSynchronize (); 
				cudaCheckError (); 

				// BATCH NORM BUG..... 
				//TODO
				//TODO
				//TODO
				//TODO
				/*
					If there is no pool layer... then we need the derivative inputs from the previous layer
						which are not stored at the moment. 
					Change the code, in backward pass to store the incoming derivatives here. 
				*/
				kerUtilsMatMatScale <<< blocks, BLOCK_SIZE >>> 
					( nextDevPtr, dx + zOffsets[ c + 1 ], count, nextDevPtr );
				cudaThreadSynchronize (); 
				cudaCheckError (); 
				//TODO
				//TODO
				//TODO
				//TODO

				//Add the two terms
				alpha = 1.; 
				cublasCheckError( cublasDaxpy( cublasHandle, count, 
											&alpha, nextDevPtr, 1, 
											rError, 1 ) ); 

				break;

			default: 
				fprintf( stderr, "Undefined Activation Function convolution Rop.... \n"); 
				exit ( -1 ); 
		}

#ifdef DEBUG_ROP
fprintf( stderr, "... Done with ROp Backward (Convolution - Activation): %d\n", c ); 
copy_host_device( hostPtr, rError, sizeof(real) * l.activationVolumn * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, p.height * p.width * n, l.outChannels ); 
print4DMatrix( hostPtr, n, l.outChannels, p.height, p.width ); 
#endif

		//BATCH NORM BEGINNING HERE.....
		//BATCH NORM BEGINNING HERE.....
		//BATCH NORM BEGINNING HERE.....
			//update rError in the backward direction... 
		if (l.batchNorm != PERFORM_NO_BATCH_NORM) {
			computeROpBatchNormBackward( z + zOffsets[ c + 1 ], 
				z + zOffsets[ c + 1] + l.batchNormOffset, 
				rz + zOffsets[ c + 1 ], 
				rz + zOffsets[ c + 1 ] + l.batchNormOffset, 
				dx + zOffsets[ c + 1 ] + l.activationOffset, 
				rError, 
				NULL, BATCH_NORM_EPSILON, 
				z + zOffsets[ c + 1 ] + l.batchNormOffset + l.meansOffset, 
				z + zOffsets[ c + 1 ] + l.batchNormOffset + l.variancesOffset, 
				l.outHeight, l.outWidth, l.outChannels, n, model->batchSize, 
				nextDevPtr, hostPtr ); 

#ifdef DEBUG_ROP
fprintf( stderr, "Done with ROp-BN ... \n"); 
copy_host_device( hostPtr, rError, sizeof(real) * l.activationVolumn * n,
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.outHeight, l.outWidth ); 
#endif

		}
		//BATCH NORM BEGINNING HERE.....
		//BATCH NORM BEGINNING HERE.....
		//BATCH NORM BEGINNING HERE.....

		//backwards Convolution.... Meat of the operation here. 
		/* 
			dW = z' * delta 
			RdW = R{ z' O delta}
				 = R{ z }' O delta + z' O  R{ delta }

			RdW --- BEGIN
		*/
		
		if (c != 0) {
      	CONV_LAYER prevLayer = model->convLayer[ c-1 ];
			// Img2Col( z )
			//TODO
			//TODO
			//TODO
			//getBatchImageCols( rz + zOffsets[ c ] + 2 * prevLayer.convOffset, n, 
			//	l.outChannels, l.height, l.width, l.kSize, l.padding, l.stride, nextDevPtr ); 

			//SUDHIR-CHANGES-DOUBLE-CHECK
			//getBatchImageCols( rz + zOffsets[ c ] + 2 * prevLayer.convOffset, n, 
			//SUDHIR-CHANGES-DOUBLE-CHECK


			getBatchImageCols( rz + zOffsets[ c ] + prevLayer.outputOffset, n, 
				l.inChannels, l.height, l.width, l.kSize, l.padding, l.stride, nextDevPtr ); 
			//TODO
			//TODO

			// dW = R{ Img2Col( z ) }' O delta
			alpha = 1.; 
			cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
						l.kSize * l.kSize * l.inChannels, l.outChannels, p.height * p.width * n, 
						&alpha, nextDevPtr, p.height * p.width * n, 
						//TODO SUDHIR-DOUBLE-CHECK
						//dx + zOffsets[ c + 1 ] + n * l.outChannels * p.height * p.width, 
						//dx + zOffsets[ c + 1 ] + l.activationOffset, 
						((l.batchNorm != PERFORM_NO_BATCH_NORM) ? (dx + zOffsets[ c + 1 ] + l.batchNormOffset) : 
						(dx + zOffsets[ c + 1] + l.activationOffset)),
						p.height * p.width * n, 
						&beta, hv + wOffsets[ c ], l.kSize * l.kSize * l.inChannels ) ); 
		} else {
			cuda_memset( hv + wOffsets[ c ], 0, sizeof(real) * l.kSize * l.kSize * l.inChannels * l.outChannels, 
				ERROR_MEMSET ); 
		}

		//  z' O R{ delta }
		if (c != 0){
      	CONV_LAYER prevLayer = model->convLayer[ c-1 ];
			//TODO
			//TODO
			//getBatchImageCols( z + zOffsets[ c ] + 2 * prevLayer.convOffset, n, 
			//	l.outChannels, l.height, l.width, l.kSize, l.padding, l.stride, nextDevPtr ); 
			//SUDHIR-DOUBLE-CHECK-CHANGES
			//getBatchImageCols( z + zOffsets[ c ] + 2 * prevLayer.convOffset, n, 
			//SUDHIR-DOUBLE-CHECK-CHANGES

			getBatchImageCols( z + zOffsets[ c ] + prevLayer.outputOffset, n, 
				l.inChannels, l.height, l.width, l.kSize, l.padding, l.stride, nextDevPtr ); 
			//TODO
			//TODO
		} else {
			//TODO
			//TODO
			//TODO
			//getBatchImageCols( data->trainSetX + s * data->features, n, 
			//	l.outChannels, l.height, l.width, l.kSize, l.padding, l.stride, nextDevPtr ); 
			//getBatchImageCols( data->trainSetX + s * data->features, n, 
			getBatchImageCols( data->currentBatch, n, 
				l.inChannels, l.height, l.width, l.kSize, l.padding, l.stride, nextDevPtr ); 
			//TODO
			//TODO
			//TODO
		}
		alpha = 1.; 
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
						l.kSize * l.kSize * l.inChannels, l.outChannels, p.height * p.width * n, 
						&alpha, nextDevPtr, p.height * p.width * n, 
						rError, p.height * p.width * n, 
						&beta, nextDevPtr + l.kSize * l.kSize * l.inChannels * p.height * p.width * n, 
									l.kSize * l.kSize * l.inChannels ) ); 	

		// add the two terms; 
		alpha = 1.; 
		cublasCheckError( cublasDaxpy( cublasHandle, l.inChannels * l.outChannels * l.kSize * l.kSize, 
									&alpha, nextDevPtr + l.kSize * l.kSize * l.inChannels * p.height * p.width * n, 1, 
									hv + wOffsets [ c ], 1 ) ); 

#ifdef DEBUG_ROP
fprintf( stderr, "... Done with ROp Backward (Convolution - dW): %d\n", c ); 
copy_host_device( hostPtr, hv + wOffsets[ c ], 
	sizeof(real) * l.kSize * l.kSize * l.inChannels * l.outChannels, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.kSize * l.kSize * l.inChannels, l.outChannels ); 
#endif
		/*
			RdW ----- END


			db = sum( delta, 2 )
			Rdb = sum( R{ delta }, 2 )
		
			RdB ------ Begin
		*/
		if (model->bias != 0) {
			blocks = ( p.height * p.width * n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
			kerInitOneVector <<< blocks, BLOCK_SIZE >>> 
				(nextDevPtr, p.height * p.width * n ); 
			cudaThreadSynchronize (); 
			cudaCheckError (); 

			// 1( 1 X p.height * p.width * n ) * errors( p.height * p.width * n * channels )
			alpha = 1.; 
			cublasCheckError( cublasDgemm ( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, l.outChannels, p.height * p.width * n, 
								&alpha, nextDevPtr, 1, 
								rError, p.height * p.width * n, 
								&beta, hv + bOffsets[ c ], 1 ) ); 
#ifdef DEBUG_ROP
fprintf( stderr, "... Done with ROp Backward (Convolution - dB): %d\n", c ); 
copy_host_device( hostPtr, hv + bOffsets[ c ], sizeof(real) * l.outChannels, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.outChannels, 1 ); 
#endif
		}

		/* 
			RdB ------ End

			delta = delta * Weights
			R{ delta } = R{ delta * weights }
						 = R{ delta } * weights + delta * R{ weights }

			R{delta} ----- BEGIN
		*/

		// R{ delta } O weights
		if (c != 0){ 
			nextDevPtr2 = nextDevPtr + l.height * l.width * l.inChannels * n; 
			backpropConvolution( rError, p.height, p.width, l.outChannels, 
					weights + wOffsets[ c ], l.kSize, l.kSize, l.height, l.width, l.padding, l.inChannels, 	
					n, nextDevPtr, nextDevPtr2, hostPtr ); 
#ifdef DEBUG_ROP
fprintf( stderr, "... Done with ROp Backward (Convolution - dError): %d, R{delta} * weights\n", c ); 
copy_host_device( hostPtr, nextDevPtr, sizeof(real) * l.inChannels * l.height * l.width * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.height, l.width ); 
#endif
			//TODO -- Check whether the correct dx is being used or not here. 
			//TODO 
			//TODO 
			//TODO 
			//TODO 
			//backpropConvolution( dx + zOffsets[ c + 1 ] + l.activationOffset, 
			backpropConvolution( ((l.batchNorm != PERFORM_NO_BATCH_NORM) ? 
											(dx + zOffsets[ c + 1 ] + l.batchNormOffset) : 
											(dx + zOffsets[ c + 1 ] + l.activationOffset)), 
					p.height, p.width, l.outChannels, 
					vector + wOffsets[ c ], l.kSize, l.kSize, l.height, l.width, l.padding, l.inChannels, 
					n, rError, nextDevPtr2, hostPtr ); 
			//TODO 
			//TODO 
			//TODO 
			//TODO 
#ifdef DEBUG_ROP
fprintf( stderr, "... Done with ROp Backward (Convolution - dError): %d, delta * R{ weights }\n", c ); 
copy_host_device( hostPtr, rError, sizeof(real) * l.inChannels * l.height * l.width * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.height, l.width ); 
#endif

			//y = ax + y
			alpha = 1.;
			cublasCheckError( cublasDaxpy( cublasHandle, l.inChannels * l.height * l.width * n, 
									&alpha, nextDevPtr, 1, rError, 1 ) ); 
#ifdef DEBUG_ROP
fprintf( stderr, "... Done with ROp Backward (Convolution - dError): %d, result\n", c ); 
copy_host_device( hostPtr, rError, sizeof(real) * l.inChannels * l.height * l.width * n, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, l.height, l.width ); 
#endif
		}

		/*
			R{delta} ----- END
		*/
#ifdef DEBUG_ROP
fprintf( stderr, ".... Done with Convolution Layer (ROP BackwardPass) : %d \n", c ); 
#endif
		
	}

	//scale appropriately
	//alpha = 1./(real)curBatchSize; 
	//cublasCheckError( cublasDscal( cublasHandle, model->pSize, &alpha, hv, 1 ) ); 
}
