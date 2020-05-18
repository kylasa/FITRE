
#include <solvers/kfac_natural_gradient.h>

#include <nn/nn_decl.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>

#include <functions/dev_backprop_convolution.h>
#include <functions/dev_transpose.h>

#include <core/errors.h>

#include <solvers/kfac_inverses.h>
#include <utilities/print_utils.h>

GLOBAL void ker_copy_bias( real *mat, int rows, int cols, real* value )
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x; 

   if (idx < cols) {
      mat[ idx * rows + rows - 1 ] = value[ idx ]; 
   }   
}

/*
	Compute the Natural Gradient here. 
	Natural Gradient 	 = K^{Inv} * vec
							 = Lambda * vec * Omega

	Lambda 	= D^{Inv}
	Omega		= Z^{Inv} 
*/

void computeNaturalGradient (CNN_MODEL *model, 
	KFAC_CURVATURE_INFO *kfacInfo,
	real *devPtr, real *hostPtr )
{
	real *nextDevPtr;
	real *nextHostPtr = hostPtr;
	real *vComponent; 
	real *wb; 

	real *omegaZ; 
	real *lambdaG; 

	int *wOffsets = model->wOffsets; 
	int *bOffsets = model->bOffsets; 
	int *zOffsets = model->zOffsets; 

	int *omegaZOffsets = kfacInfo->OmegaZOffsets; 
	int *lambdaGOffsets = kfacInfo->LambdaGOffsets; 

	real *omegaPtr = kfacInfo->OmegaZInv; 
	real *lambdaPtr = kfacInfo->LambdaGInv; 
	real *vec = kfacInfo->gradient; 
	real *nGradient = kfacInfo->nGradient; 
	real *nGradComponent = NULL; 

	real alpha, beta; 
	int outChannels, inChannels, height, width;
	int blocks; 



	//Convolution Layer here. 
	for (int c = 0; c < model->cLayers; c ++) {

		CONV_LAYER *convLayer = &model->convLayer[ c ]; 
		POOL_LAYER *poolLayer = &model->poolLayer[ c ];

		outChannels = convLayer->outChannels; 
		inChannels = convLayer->inChannels; 
		height = convLayer->kSize; 
		width = convLayer->kSize; 

		omegaZ = omegaPtr + omegaZOffsets[ c ]; 
		lambdaG = lambdaPtr + lambdaGOffsets[ c ]; 

		vComponent = vec + wOffsets[ c ] ; 
		nGradComponent = nGradient + wOffsets[ c ]; 
		wb = devPtr; 
		if (model->bias != 0)
			nextDevPtr = wb + outChannels * (inChannels * height * width + 1); 
		else
			nextDevPtr = wb + outChannels * inChannels * height * width ; 

		//Perform Delta' * vec * Lambda'
		// Delta' --> outChannel * outChannel
		// vec    --> outChannels * inChannels * height * width + outChannels
		//			 --> outChannels X (inChannels * height * width + 1 )
		// Z'     --> (inChannels * height * width + 1) X (inChannels * heightsamples * width + 1)

		// Conversion from Column Major to Row Major of each filter in the outChannel * inChannel
		// combination here. 

/*
		blocks = (outChannels * inChannels * height * width + BLOCK_SIZE - 1) / BLOCK_SIZE; 
		ker_transpose <<< blocks, BLOCK_SIZE >>>
			( vComponent, outChannels * inChannels * height * width, outChannels, height, width, inChannels, nextDevPtr ); 
		cudaDeviceSynchronize (); 
		cudaCheckError (); 
*/
copy_device( nextDevPtr, vComponent, sizeof(real) * outChannels * inChannels * width * height, ERROR_MEMCPY_DEVICE_DEVICE);

		if (model->bias != 0) {

			cudaMemcpy2D( 	wb, sizeof(real) * (inChannels * height * width + 1), 
							nextDevPtr, sizeof(real) * inChannels * height * width,  
							sizeof(real) * (inChannels * height * width), sizeof(real) * outChannels,  
							cudaMemcpyDeviceToDevice ) ;
			cudaCheckError (); 

			blocks = (outChannels + BLOCK_SIZE - 1) / BLOCK_SIZE; 
			ker_copy_bias<<< blocks, BLOCK_SIZE >>> 
				( wb, inChannels * height * width + 1, outChannels, vec + bOffsets[ c ] ); 
			cudaThreadSynchronize (); 
			cudaCheckError (); 
		} else {
			copy_device( wb, nextDevPtr, sizeof(real) * inChannels * height * width * outChannels, 
								ERROR_MEMCPY_DEVICE_DEVICE ); 
		}

//cublasCheckError( cublasDnrm2( cublasHandle, outChannels * (inChannels * height * width + 1), wb, 1, hostPtr )); 
//fprintf( stderr, "Vector Component: %.10f\n", *hostPtr ); 

/*
fprintf( stderr, "vector ... \n"); 
copy_host_device( nextHostPtr, wb, sizeof(real) * outChannels * (inChannels * height * width + 1), 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, inChannels * height * width + 1, outChannels ); 
*/

		// result = Delta * vec'
		alpha = 1; beta = 0; 
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
									outChannels, 
									((model->bias == 0) ? 0 : 1 ) + inChannels * height * width, 
									outChannels, 
									&alpha, lambdaG, outChannels, 
									wb, inChannels * height * width + ((model->bias == 0) ? 0 : 1), 
									&beta, nextDevPtr, outChannels ) ); 

		// result * Z
		alpha = 1; beta = 0; 
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
									outChannels, 
									inChannels * height * width + ((model->bias == 0) ? 0 : 1), 
									inChannels * height * width + ((model->bias == 0) ? 0 : 1), 
									&alpha, nextDevPtr, outChannels, 
									omegaZ, inChannels * height * width + ((model->bias == 0) ? 0 : 1), 
									&beta, nGradComponent, outChannels ) ); 

//cublasCheckError( cublasDnrm2( cublasHandle, outChannels * (inChannels * height * width + 1), nGradComponent, 1, hostPtr )); 
//fprintf( stderr, "NGrad Component: %.10f\n", *hostPtr ); 

		// We need to reshape the nGradComponent here. 
		// From outChannels * (inChannels * kSize * kSize + 1)
		// To inChannels * kSize * kSize * outChannels  + outChannels
		// Basically a Transpose of a matrix here. 
		//reshapeMatrix( nGradComponent, inChannels, outChannels, height * width, nextDevPtr ); 	

		//Transpose the matrix here. 
		alpha = 1.; beta = 0; 
		cublasCheckError( cublasDgeam( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
									inChannels * height * width, outChannels, 
									&alpha, nGradComponent, outChannels, 
									&beta, nextDevPtr, inChannels * height * width, 
									nextDevPtr, inChannels * height * width ) ); 

		copy_device( nGradComponent, nextDevPtr, sizeof(real)*outChannels * inChannels * height * width, 
							ERROR_MEMCPY_DEVICE_DEVICE ); 

/*
		blocks = (outChannels * inChannels * height * width + BLOCK_SIZE - 1) / BLOCK_SIZE; 
		ker_transpose_rc <<< blocks, BLOCK_SIZE >>>
			( nextDevPtr, outChannels * inChannels * height * width, outChannels, height, width, inChannels, nGradComponent); 
		cudaDeviceSynchronize (); 
		cudaCheckError (); 
*/

//cublasCheckError( cublasDnrm2( cublasHandle, outChannels * (inChannels * height * width + 1), nGradComponent, 1, hostPtr )); 
//fprintf( stderr, "NGrad Component: %.10f\n", *hostPtr ); 
			
/*
fprintf( stderr, "Natural Gradient Convolution .... \n"); 
copy_host_device( hostPtr, nGradComponent, sizeof(real) * outChannels * (inChannels * height * width  + 1), 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, inChannels * height * width, outChannels ); 
print2DMatrix( hostPtr + outChannels * inChannels * height * width, 1, outChannels ); 
*/
		
	}

	//Linear Layers here. 
	for (int l = 0; l < model->lLayers; l ++) {
		FC_LAYER *ll = &model->fcLayer[ l ]; 

		omegaZ = omegaPtr + omegaZOffsets[ model->cLayers + l ] ; 
		lambdaG = lambdaPtr + lambdaGOffsets[ model->cLayers + l ]; 

		vComponent = vec + wOffsets[ model->cLayers + l ];
		nGradComponent = nGradient + wOffsets[ model->cLayers + l ] ;
		nextDevPtr = devPtr; 

/*
fprintf( stderr, "vector ... \n"); 
copy_host_device( nextHostPtr, vComponent, sizeof(real) * ll->out* (ll->in + 1), 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( nextHostPtr, ll->out, ll->in + 1); 
*/

		//result = Delta' * vec
		alpha = 1.; beta = 0; 
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								ll->out, ll->in + ((model->bias == 0) ? 0 : 1), ll->out, 	
								&alpha, lambdaG, ll->out, vComponent, ll->out, 
								&beta, nextDevPtr, ll->out ) ); 

		// result * Lambda
		alpha = 1.; beta = 0; 
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								ll->out, ll->in + ((model->bias == 0) ? 0 : 1), ll->in + ((model->bias == 0) ? 0 : 1), 
								&alpha, nextDevPtr, ll->out, omegaZ, ll->in + ((model->bias == 0) ? 0 : 1), 
								&beta, nGradComponent, ll->out ) ); 

//cublasCheckError( cublasDnrm2( cublasHandle, ll->out * (ll->in + 1), nGradComponent, 1, hostPtr )); 
//fprintf( stderr, "NGrad Component: %.10f\n", *hostPtr ); 

/*
fprintf( stderr, "Natural Gradient Linear .... \n"); 
copy_host_device( hostPtr, nGradComponent, sizeof(real) * ll->out* (ll->in + 1), 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, ll->out, ll->in); 
print2DMatrix( hostPtr + ll->out * ll->in, 1, ll->out ); 
*/

	}
	//printNorms( model, kfacInfo ); 
}
