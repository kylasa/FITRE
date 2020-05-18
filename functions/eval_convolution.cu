
#include <functions/eval_convolution.h>
#include <functions/dev_layer_r_error.h>
#include <functions/swish.h>
#include <functions/dev_batch_norm.h>

#include <core/structdefs.h>
#include <core/errors.h>
#include <core/memsizes.h>

#include <nn/nn_decl.h>
#include <nn/utils.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

#include <functions/dev_image.h>
#include <functions/dev_pool.h>
#include <functions/dev_activations.h>

#include <utilities/utils.h>
#include <utilities/print_utils.h>


GLOBAL void ker_add_bias ( real *input, int kernel_size, real *bias, int samples, int n, int channels )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	int cIdx = idx / (kernel_size * samples ); 
	
	if (idx < n){
		input[ idx ] += bias[ cIdx ]; 
	}
}

void convAddBias( real *input, real *bias, int kernel_size, int channels, int samples )
{
	int threads = kernel_size * channels * samples; 
	int blocks = (threads + BLOCK_SIZE - 1) / BLOCK_SIZE; 

	ker_add_bias <<< blocks, BLOCK_SIZE >>>
		(input, kernel_size, bias, samples, threads, channels ); 
	cudaThreadSynchronize (); 
	cudaCheckError ();
}


//apply convolution layer here. 
//1. Convolution. 
//2. Activation Function
//3. Pooling layer here. 
void applyConvolutionLayer(real *input, int samples, 
	int in_channels, int height, int width, 
	int ksize, int pad, int stride, 
	int col_height, int col_width, 
	real *weights, real *bias,
	real *output, int out_channels, 
	int actFun, int pkSize, int pkStride, int pkPad, int poolFun, 
	int poolOutHeight, int poolOutWidth,  
	BATCH_NORM_TYPES performBatchNorm, real epsilon, 
	int activationOffset, int poolOffset, int batchNormOffset, 
	int meansOffset, int variancesOffset, 
	real *devScratch, real *hostPtr, 
	EVAL_TYPE forTesting, int runningMeansOffset, int runningVariancesOffset)
{
	//int col_height = (height + 2 * pad - ksize ) / stride + 1; 
	//int col_width = (width + 2 * pad - ksize ) / stride + 1; 

	real *imgCols = devScratch; 

	/*
	real *imgColWtOut = output;
	real *actOut = imgColWtOut + activationOffset ;
	real *poolOut = imgColWtOut + poolOffset ;
	real *batchOut = imgColWtOut + batchNormOffset ;
	*/

	real *imgColWtOut, *batchOut, *actOut, *poolOut; 

	imgColWtOut = batchOut = actOut = poolOut = NULL; 

	if (performBatchNorm != PERFORM_NO_BATCH_NORM) {
		imgColWtOut = output;
		batchOut = imgColWtOut + batchNormOffset; 
		actOut = imgColWtOut + activationOffset; 
		poolOut = imgColWtOut + poolOffset; 
	} else { 
		imgColWtOut = output;
		actOut = imgColWtOut + activationOffset; 
		poolOut = imgColWtOut + poolOffset; 
	}
	
#ifdef DEBUG_CNN
	size_t memReqd = col_height * col_width * in_channels * ksize * ksize * samples + 
							col_width * col_height * out_channels * samples + 
							col_height * col_width * in_channels * out_channels * ksize * ksize * samples; 	

	fprintf( stderr, "Memory requirement for scratch here is %2.3f GB, or %4.3f MB or %zu .... \n", 
					(memReqd * sizeof(real)) / __GIGA_BYTE_SIZE__ , (memReqd * sizeof(real)/(1024 * 1024)), memReqd ); 
#endif
							

	real alpha, beta; 
	
	//img2col conversion
#ifdef DEBUG_CNN
fprintf( stderr, "Convolution--> samples: %d, Channels: %d, height: %d, width: %d, Kernel: %d, Pad: %d, Stride: %d \n", samples, in_channels, height, width, ksize, pad, stride ); 
#endif

	getBatchImageCols (input, samples, in_channels, height, width, 
			ksize, pad, stride, imgCols); 			

	//ImageCols * Weights... 
	alpha = 1; beta = 0;
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
				samples * col_height * col_width, out_channels, ksize * ksize * in_channels, 
				&alpha, imgCols, col_height * col_width * samples, 
				weights, ksize * ksize * in_channels, &beta, 
				imgColWtOut, col_height * col_width * samples ) );  

	if (bias != NULL )
		convAddBias( imgColWtOut, bias, col_height * col_width, out_channels, samples); 

#ifdef DEBUG_DETAILED
fprintf (stderr, "Convolution ImgCols X Weights output is --> \n"); 
copy_host_device( hostPtr, imgColWtOut, sizeof(real) * samples * out_channels *col_height * col_width, 
							cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, col_height * col_width * samples, out_channels ); 
#endif

	// VGG BATCH Normalization Here. 
	if ( performBatchNorm != PERFORM_NO_BATCH_NORM ){

		computeBatchNormForward( imgColWtOut, col_height, col_width,  out_channels, samples, 
			batchOut, meansOffset, variancesOffset, epsilon, imgCols, hostPtr, 
			forTesting, BATCH_NORM_MOMENTUM, runningMeansOffset, runningVariancesOffset ); 		
		
#ifdef DEBUG_BATCH_NORM
		fprintf( stderr, "Input to BATCH NORMALIZATION... \n"); 
		copy_host_device( hostPtr, imgColWtOut, 
				sizeof(real) * out_channels * samples * col_height * col_width, 
				cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
		print2DMatrix( hostPtr, samples * col_height * col_width, out_channels ); 
			
		fprintf( stderr, "Done with BatchNormalization.... Means... %d\n", meansOffset);
		copy_host_device( hostPtr, batchOut + meansOffset, sizeof(real) * out_channels, 
				cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
		print2DMatrix( hostPtr, out_channels, 1 ); 

		fprintf( stderr, "Done with BatchNormalization.... Variances... %d\n", variancesOffset);
		copy_host_device( hostPtr, batchOut + variancesOffset, sizeof(real) * out_channels, 
				cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
		print2DMatrix( hostPtr, out_channels, 1 ); 

		fprintf( stderr, "Done with BatchNormalization... output... \n\n"); 
		copy_host_device( hostPtr, batchOut, sizeof(real) * out_channels * col_height * col_width * samples, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
		print2DMatrix( hostPtr, col_height * col_width * samples, out_channels ); 
#endif

		copy_device( actOut, batchOut, sizeof(real) * col_height * col_width * out_channels * samples, 
					ERROR_MEMCPY_DEVICE_DEVICE ); 
	} else { 

		copy_device( actOut, imgColWtOut, sizeof(real) * col_height * col_width * out_channels * samples, 
					ERROR_MEMCPY_DEVICE_DEVICE ); 
	}

	int blocks = (samples * col_height * col_width * out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	//dim3 blocks( 1, samples, x_blocks ); 

	//activation function here. 
	switch( actFun ) {
			case CNN_ACT_RELU: 
					kerNNApplyRELU <<< blocks, BLOCK_SIZE >>>
						( actOut, col_height * col_width * out_channels * samples );
					cudaDeviceSynchronize (); 
					cudaCheckError (); 
				break; 

			case CNN_ACT_SOFTPLUS: 
					kerNNApplySOFTPLUS <<< blocks, BLOCK_SIZE >>>
						( actOut, col_height * col_width * out_channels * samples );
					cudaDeviceSynchronize (); 
					cudaCheckError (); 
				break; 

			case CNN_ACT_ELU: 
					kerNNApplyELU <<< blocks, BLOCK_SIZE >>>
						( actOut, col_height * col_width * out_channels * samples, 1.0  );
					cudaDeviceSynchronize (); 
					cudaCheckError (); 
				break; 
			case CNN_ACT_SWISH: 
				kerNNSwish <<< blocks, BLOCK_SIZE >>> 
					( actOut, actOut, col_height * col_width * out_channels * samples ); 
					cudaDeviceSynchronize (); 
					cudaCheckError (); 
				break;

			case CNN_ACT_NONE: 
				break;

			default: 
				exit ( -1 );
	}

#ifdef DEBUG_DETAILED
fprintf( stderr, "Convolution (Activation out) ---> \n" ); 
copy_host_device( hostPtr, actOut, sizeof(real) * samples * out_channels * col_height * col_width, 
						cudaMemcpyDeviceToHost,	ERROR_MEMCPY_DEVICE_HOST ); 
//print4DMatrix( hostPtr, samples, out_channels, col_height, col_width ); 
print2DMatrix( hostPtr, col_height * col_width * samples, out_channels ); 
#endif

	if (poolFun != NO_POOL) { 
		
		// Pooling here. 
		applyPoolLayer( actOut, samples, out_channels, col_height, col_width, 
				pkSize, pkStride, pkPad, poolFun, poolOut, 1.); 

		#ifdef DEBUG_DETAILED
		fprintf( stderr, "Convolution (Pool Out) ----- \n"); 
		copy_host_device( hostPtr, poolOut, sizeof(real) * samples * out_channels * ((col_height - pkSize)/pkStride + 1) * ((col_width - pkSize)/pkStride + 1), cudaMemcpyDeviceToHost,ERROR_MEMCPY_DEVICE_HOST ); 
		//print4DMatrix( hostPtr, samples, out_channels, (col_height - pkSize)/pkStride + 1, (col_width - pkSize)/pkStride + 1 ); 
		print2DMatrix( hostPtr, samples * ((col_height - pkSize)/pkStride + 1) * ((col_width - pkSize)/pkStride + 1), out_channels ); 
		#endif
	}

#ifdef DEBUG_BATCH_NORM
fprintf( stderr, "Convolution (BatchNorm) begin...--- \n"); 
#endif

/*
	if ( performBatchNorm >= 1 ){

		computeBatchNormForward( poolOut, poolOutHeight, poolOutWidth,  out_channels, samples, 
			batchOut, meansOffset, variancesOffset, epsilon, imgCols, hostPtr, 
			forTesting, BATCH_NORM_MOMENTUM, runningMeansOffset, runningVariancesOffset ); 		
		
#ifdef DEBUG_BATCH_NORM
		fprintf( stderr, "Input to BATCH NORMALIZATION... \n"); 
		copy_host_device( hostPtr, poolOut, 
				sizeof(real) * out_channels * samples * poolOutHeight * poolOutWidth, 
				cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
		print2DMatrix( hostPtr, samples * poolOutHeight * poolOutWidth, out_channels ); 
			
		fprintf( stderr, "Done with BatchNormalization.... Means... %d\n", meansOffset);
		copy_host_device( hostPtr, batchOut + meansOffset, sizeof(real) * out_channels, 
				cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
		print2DMatrix( hostPtr, out_channels, 1 ); 

		fprintf( stderr, "Done with BatchNormalization.... Variances... %d\n", variancesOffset);
		copy_host_device( hostPtr, batchOut + variancesOffset, sizeof(real) * out_channels, 
				cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
		print2DMatrix( hostPtr, out_channels, 1 ); 

		fprintf( stderr, "Done with BatchNormalization... output... \n\n"); 
		copy_host_device( hostPtr, batchOut, sizeof(real) * out_channels * poolOutHeight * poolOutWidth * samples, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
		print2DMatrix( hostPtr, poolOutHeight * poolOutWidth * samples, out_channels ); 
#endif
	}
*/

#ifdef DEBUG_BATCH_NORM
fprintf( stderr, "Convolution (BatchNorm) end...--- \n"); 
#endif
}

/*
	Forward pass for the Convolution Layer

	R{ W O Z + b } = 	R{W} O Z + W O R{Z} + R{b}
	VW O Z + W O R{Z} + Vb
	
	with R{ Z_0 } = 0	

	For the moment, we just store rz

	rz_1 = R{ W O Z + b }
	rz_2 = h(rz_1)
	rz_3 = pool( rz_2 )	
	
*/

void applyROpConvolutionLayer(real *input, real *prev_z, int offset, 
	real *z, real *rz_in, int samples, 
	int in_channels, int height, int width, 
	int ksize, int pad, int stride, 
	int col_height, int col_width, 
	real *weights, real *bias,
	real *vweights, real *vbias, 
	real *rx, real *rz, int out_channels, 
	int actFun, int pkSize, int pkStride, int pkPad, int poolFun,  
	int activationOffset, int poolOffset, int batchNormOffset, int outputOffset,
	int convVolumn, int activationVolumn, int poolVolumn, int batchNormVolumn, 
	real *devScratch, real *hostPtr, BATCH_NORM_TYPES performBatchNorm, real epsilon, int batchSize)
{
	real *imgCols = devScratch; 
	real *rzWt = imgCols + col_height * col_width * samples * in_channels * ksize * ksize; 

	/*
	real *imgColWtOut = rz;
	real *actOut = imgColWtOut + activationOffset;
	real *poolOut = imgColWtOut + poolOffset;
	*/

	real *imgColWtOut, *batchOut, *actOut, *poolOut; 
	
	imgColWtOut = batchOut = actOut = poolOut = NULL; 

	if (performBatchNorm != PERFORM_NO_BATCH_NORM) {
		imgColWtOut = rz; 
		batchOut = imgColWtOut + batchNormOffset; 
		actOut = imgColWtOut + activationOffset; 
		poolOut = imgColWtOut + poolOffset;  
	} else {
		imgColWtOut = rz; 
		actOut = imgColWtOut + activationOffset; 
		poolOut = imgColWtOut + poolOffset;  
	}

//#ifdef DEBUG_BATCH
//fprintf( stderr, "ROp Conv-Act-Pooling-Batch ... TESTING z_out for batch normalization... \n"); 
//copy_host_device( hostPtr, z + batchNormOffset, sizeof(real) * poolVolumn * batchSize , cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
//print2DMatrix( hostPtr, poolVolumn, out_channels ); 

//fprintf( stderr, "Before Applying ROP Means are as follows: %d\n", batchNormOffset + 2 * 2); 
//copy_host_device( hostPtr, (z + batchNormOffset + poolVolumn * batchSize), sizeof(real) * out_channels , cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
//print2DMatrix( hostPtr, 1, out_channels ); 
//#endif
	

	real alpha, beta; 

	real *dataset = input; 
	if (input == NULL) dataset = prev_z + offset; 
	
	//img2col conversion
	getBatchImageCols (dataset, samples, in_channels, height, width, 
			ksize, pad, stride, imgCols); 			

	//ImageCols * Weights... 
	// VW O input
	alpha = 1; beta = 0;
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
				samples * col_height * col_width, out_channels, ksize * ksize * in_channels, 
				&alpha, imgCols, col_height * col_width * samples, 
				vweights, ksize * ksize * in_channels, &beta, 
				imgColWtOut, col_height * col_width * samples ) );  

	//  Img2Col( R{ Z } ) * W
	if (rz_in != NULL) {
#ifdef DEBUG_DETAILED
fprintf( stderr, "Rz is used as follows.... \n"); 
copy_host_device( hostPtr, rz_in + offset, sizeof(real) * height * width * in_channels, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height, width ); 
#endif
	getBatchImageCols( rz_in + offset, samples, in_channels, height, width, 
			ksize, pad, stride, imgCols ); 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
				samples * col_height * col_width, out_channels, ksize * ksize * in_channels, 
				&alpha, imgCols, col_height * col_width * samples, 
				weights, ksize * ksize * in_channels, &beta, 
				rzWt, col_height * col_width * samples ) ); 	
	} else { 
		cuda_memset( rzWt, 0, 
						//sizeof(real) * ksize * ksize * out_channels * col_height * col_width * samples, 
						sizeof(real) * out_channels * col_height * col_width * samples, 
						ERROR_MEMSET ); 
	}

	// R{Z} O W + Vb
	if ((bias != NULL) && (vbias != NULL) )
		convAddBias( rzWt, vbias, col_height * col_width, out_channels, samples); 

	// VW O Z + (W O R{Z} + Vb)
	// use Daxpy = Y = ax + y
	cublasCheckError( cublasDaxpy( cublasHandle, 
								activationVolumn * samples , 
								&alpha, rzWt, 1, imgColWtOut, 1 ) ); 

#ifdef DEBUG_DETAILED
fprintf( stderr, "... ROp done with Convlution... \n"); 
copy_host_device( hostPtr, imgColWtOut, sizeof(real) * col_height * col_width * samples * out_channels, cudaMemcpyDeviceToHost, 
							ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, col_height * col_width * samples, out_channels ); 
#endif

	if (performBatchNorm != PERFORM_NO_BATCH_NORM){

#ifdef DEBUG_BATCH
fprintf( stderr, "ROp Conv-Batch ... begin... \n"); 
#endif

		computeROpBatchNormForward (z, z + batchNormOffset, imgColWtOut, batchOut,
			devScratch, hostPtr, epsilon, col_height, col_width, out_channels, samples, batchSize ); 

#ifdef DEBUG_BATCH
fprintf( stderr, "ROp Conv-Batch ... end... \n"); 
copy_host_device( hostPtr, batchOut, sizeof(real) * col_height * col_width * out_channels * samples, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, col_height * col_width * samples, out_channels ); 
#endif

		copy_device( actOut, batchOut, sizeof(real) * convVolumn * samples, 
					ERROR_MEMCPY_DEVICE_DEVICE ); 
	} else {
		copy_device( actOut, imgColWtOut, sizeof(real) * convVolumn * samples, 
					ERROR_MEMCPY_DEVICE_DEVICE ); 
	}

	

	int blocks = (samples * convVolumn + BLOCK_SIZE - 1) / BLOCK_SIZE; 

	//activation function here. 
	switch( actFun ) {
			case CNN_ACT_RELU: 
					kerNNApplyRELU <<< blocks, BLOCK_SIZE >>>
						( actOut, convVolumn * samples );
					cudaDeviceSynchronize (); 
					cudaCheckError (); 
				break; 

			case CNN_ACT_SOFTPLUS: 
					/*
					kerNNApplySOFTPLUS <<< blocks, BLOCK_SIZE >>>
						( actOut, col_height * col_width * out_channels * samples );
					*/
					kerNNROpSOFTPLUS <<< blocks, BLOCK_SIZE >>> 
						//( actOut, actOut, col_height * col_width * out_channels * samples ); 
						( actOut, 
							(performBatchNorm != PERFORM_NO_BATCH_NORM) ? (z + batchNormOffset) : ( z ), 
							convVolumn * samples ); 
					cudaDeviceSynchronize (); 
					cudaCheckError (); 
				break; 

			case CNN_ACT_ELU: 
					kerNNApplyELU <<< blocks, BLOCK_SIZE >>>
						( actOut, convVolumn * samples, 1.0  );
					cudaDeviceSynchronize (); 
					cudaCheckError (); 
				break; 

			case CNN_ACT_SWISH: 
					/*
					kerNNBackPropSwish <<< blocks, BLOCK_SIZE >>> 
						( z , z + activationOffset,
							actOut, convVolumn * samples ); 
					*/
/*
fprintf( stderr, "Z Input to the Rop Swish function... \n"); 
copy_host_device( hostPtr, z, sizeof(real) * convVolumn * samples, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, col_height * col_width * samples, out_channels ); 
*/

					//kerNNROpSwish <<< blocks, BLOCK_SIZE >>> 
					//	( z, actOut, convVolumn * samples ); 

					kerNNBackPropSwish <<< blocks, BLOCK_SIZE >>> 
						//( z, z + activationOffset, actOut, convVolumn * samples );
					( (performBatchNorm != PERFORM_NO_BATCH_NORM) ? ( z + batchNormOffset ) : ( z ), 
						z + activationOffset, actOut, convVolumn * samples ); 
					cudaDeviceSynchronize (); 
					cudaCheckError (); 
				break;


			default: 
				exit ( -1 );
	}

#ifdef DEBUG_DETAILED
fprintf( stderr, "... ROp Convolution-Activation.... done\n\n"); 
copy_host_device( hostPtr, actOut, sizeof(real) * convVolumn * samples, cudaMemcpyDeviceToHost,
						ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, col_height * col_width * samples, out_channels ); 
#endif


	// Pooling here. 
	//applyPoolLayer( actOut, samples, out_channels, col_height, col_width, 
	//			pkSize, pkStride, pkPad, poolFun, poolOut); 

	//Validate this implementation somewhere( pytorch and tensorflow ). 
	//TODO - In R Operation this should be pooling derivative... 
	// In this case, for Average Pooling... we should * with 1/(kernel * kernel)

	//applyPoolLayer( actOut, samples, out_channels, col_height, col_width, 
	//			pkSize, pkStride, pkPad, poolFun, poolOut, 1./(pkSize * pkSize)); 
	//applyRopPoolLayer( actOut, z + activationOffset, samples, out_channels, col_height, col_width, 
	//			pkSize, pkStride, pkPad, poolFun, poolOut, 1.); 
	/*
	applyPoolLayer( actOut, samples, out_channels, col_height, col_width, 
				pkSize, pkStride, pkPad, poolFun, poolOut, 1.); 
	*/

	if (poolFun != NO_POOL) {

		applyROpPoolLayer( actOut, z + activationOffset, samples, out_channels, col_height, col_width, pkSize, pkStride, pkPad, poolFun, poolOut, 1. ); 

		#ifdef DEBUG_DETAILED
		int p_height = ( col_height - pkSize) / pkStride+ 1;  
		fprintf( stderr, "... ROp Conv-Act-Pooling ... done... \n\n"); 
		copy_host_device( hostPtr, poolOut, sizeof(real) * p_height * p_height * out_channels * samples, 
				cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
		print2DMatrix( hostPtr, p_height * p_height * samples, out_channels ); 
		#endif
	}

/*
	if (performBatchNorm != PERFORM_NO_BATCH_NORM){

#ifdef DEBUG_BATCH
fprintf( stderr, "ROp Conv-Act-Pooling-Batch ... begin... \n"); 
#endif

		int pool_height, pool_width;
		pool_height = 0;
		pool_width = 0; 

		getDimensions(col_height, col_width, pkPad, pkStride, pkSize, 
		  	&pool_height, &pool_width); 

		computeROpBatchNormForward (z + poolOffset, z + batchNormOffset, poolOut, rz + batchNormOffset, 
			devScratch, hostPtr, epsilon, pool_height, pool_width, out_channels, samples, batchSize ); 

#ifdef DEBUG_BATCH
fprintf( stderr, "ROp Conv-Act-Pooling-Batch ... end... \n"); 
copy_host_device( hostPtr, rz + batchNormOffset, sizeof(real) * pool_height * pool_width * out_channels * samples, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, pool_height * pool_width * samples, out_channels ); 
#endif

	}
*/


}

void testImgConv(real *input, int in_channels, int height, int width, 
	int ksize, int pad, int stride, int out_channels, int poolFun, 
	real *weights, real *bias, int samples, real *devScratch)
{
	int col_height = (height + 2 * pad - ksize ) / stride + 1; 
	int col_width = (width + 2 * pad - ksize ) / stride + 1; 

	real alpha, beta; 

	real *imgCols = devScratch;
	real *imgColsWeights = imgCols + (col_height * col_width * ksize * ksize * in_channels ) * samples;
	real *convOut= imgColsWeights + (col_height * col_width * out_channels) * samples; 

	real *nextDevPtr = convOut + col_height * col_width * out_channels * samples; 

	//img2col conversion
	if (samples > 1) {
		getBatchImageCols (input, samples, in_channels, height, width, ksize, pad, stride, imgCols); 			
	} else {
		getImageCols( input, in_channels, height, width, ksize, pad, stride, imgCols ); 
	}
	fprintf( stderr, "... ImgCols conversion done..\n"); 

	//ImageCols * Weights... 
	//cuda_memset( imgColsWeights, 0, sizeof(real) * col_height * col_width * out_channels, ERROR_MEMSET ); 
	//copy_host_device( temp, weights, sizeof(real) * ksize * ksize , cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 

	alpha = 1; 
	beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
				samples * col_height * col_width, out_channels, ksize * ksize * in_channels, 
				&alpha, imgCols, col_height * col_width * samples, 
				weights, in_channels * ksize * ksize, &beta, 
				imgColsWeights, col_height * col_width * samples) );  
	fprintf( stderr, ".... Done with ImgCols x Weights... \n"); 

	convAddBias( imgColsWeights, bias, col_height * col_width, out_channels, samples); 

	//perform the merge operation here. 
	//mergeColWeights( imgColsWeights, col_height * col_width, out_channels, in_channels, samples, nextDevPtr ); 

	int x_blocks = (col_height * col_width * out_channels * samples + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	//dim3 blocks( 1, samples, x_blocks ); 

	kerNNApplySOFTPLUS <<< x_blocks, BLOCK_SIZE >>>
						( imgColsWeights, col_height * col_width * out_channels * samples );
	cudaDeviceSynchronize (); 
	cudaCheckError (); 
	fprintf( stderr, ".... Done with Activation... \n"); 

	applyPoolLayer( imgColsWeights, samples, out_channels, col_height, col_width, 
			2, 2, 0, poolFun, convOut, 1.); 
	fprintf( stderr, ".... Done with Pooling... \n"); 
}
