
#include <functions/dev_batch_norm.h>
#include <functions/dev_hadamard.h>
#include <functions/dev_backprop_convolution.h>
#include <functions/dev_initializations.h>

#include <device/cuda_utils.h>
#include <device/handles.h>

#include <utilities/print_utils.h>

#include <core/errors.h>

GLOBAL void ker_compute_batch_variance ( real *input, 
	int height, int width, int channels, int samples, real *mean, real *output ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	real t = 0; 

	if (idx < (height * width * channels * samples) ) {
		int chIdx = idx / (height * width * samples); 
		t = input[ idx ] - (  mean[ chIdx ]);
		output[ idx ] = t * t;
	}
}

GLOBAL void ker_compute_ZHat( real *input, int rows, int channels, 
	real *means, real *variances, real epsilon, real *output ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	real t = 0; 

	if (idx < (rows * channels) ) {
		int chIdx = idx / rows; 
		t = input[ idx ] - means[ chIdx ];
		output[ idx ] = t / sqrt( variances[ chIdx ] + epsilon );
	}
}

GLOBAL void ker_compute_rop_first( real *input, int rows, int channels, 
	real *means, real *output ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	real t = 0; 

	if (idx < (rows * channels) ) {
		int chIdx = idx / rows; 
		output[ idx ]  = input[ idx ] - means[ chIdx ];
	}
}

GLOBAL void ker_compute_rop_first_scale( real *input, 
	int rows, int channels, real *variances, real epsilon ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	real t = 0; 

	if (idx < (rows * channels) ) {
		int chIdx = idx / rows; 
		input[ idx ]  = input[ idx ]/sqrt( epsilon + variances[ chIdx ] ); 
	}
}


GLOBAL void ker_compute_rop_second (real *z_in, real *means, 
	real *rop_in, real *output, 
	int rows, int channels )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	real t = 0;

	if (idx < (rows * channels)){
		int chIdx = (idx / rows); 
	
		output[ idx ]  = ( z_in[ idx ] - means[ chIdx ] ) * rop_in[ idx ]; 
	}
}

GLOBAL void ker_compute_rop_second_scale( real *rop_input, 
	real *zin, real *means, real *output, int rows, int channels, real *variances, real epsilon )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	real t = 0;

	if (idx < (rows * channels)){
		int chIdx = idx / rows; 

		output[ idx ] = rop_input [ chIdx ] * (zin[ idx ] - means [ chIdx ]) / ( rows * pow( epsilon + variances[ chIdx ], (real)1.5 ) ); 
	}
}

GLOBAL void ker_backprop_batch_norm_no_dx( 
	real *sum_across_samples_1, 
	real *sum_across_samples_2, 
	real *sum_across_samples_3, 
	real *z_out, real *rz_out, 
	real *output, real *variances, real epsilon, 
	int samples, int channels, int chSize ){
	
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	if (idx < (channels * chSize * samples)){
		int chIdx = idx / (chSize * samples);  

		output[ idx ] =  output[ idx ] * samples * chSize 
							- sum_across_samples_1[ chIdx ] 
							- sum_across_samples_2[ chIdx ] * rz_out[ idx ] 
							- sum_across_samples_3[ chIdx ] * z_out[ idx ]; 
		output[ idx ] /= (samples * chSize) * sqrt(epsilon + variances[ chIdx ] );
	}
}

GLOBAL void ker_backprop_batch_norm_no_scale( 
	real *sum_across_samples_1, 
	real *sum_across_samples_2, real *z_out, 
	real *output, real *variances, real epsilon, 
	int samples, int channels, int chSize ){
	
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	if (idx < (channels * chSize * samples)){
		int chIdx = idx / (chSize * samples);  

		output[ idx ] =  output[ idx ] 
							- sum_across_samples_1[ chIdx ] 
							- sum_across_samples_2[ chIdx ] * z_out[ idx ]; 
	}
}



GLOBAL void ker_backprop_batch_norm( 
	real *sum_across_samples_1, 
	real *sum_across_samples_2, real *zout, 
	real *output, real *variances, 
	real *dx, real epsilon, 
	int samples, int channels, int chSize ){
	
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	if (idx < (channels * chSize * samples)){
		int chIdx = idx / (chSize * samples);  

		output[ idx ] =  output[ idx ]
							- sum_across_samples_1[ chIdx ] 
							- sum_across_samples_2[ chIdx ] * zout[ idx ]; 
		//dx[ idx ] = output[ idx ]; 
		output[ idx ] /= (((real)(samples * chSize)) * sqrt(epsilon + variances[ chIdx ] ));
	}
}


GLOBAL void ker_compute_RI 
		(real *z_in, real *means, real *rz_in, real *rop_means, real *rI, 
			real *variance, real epsilon,
			int samples, int channels, int height, int width) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	if (idx < (samples * channels * height * width) ){
		int chIdx = idx / (height * width * samples); 

		rI[ idx ] = 
			     (z_in[ idx ] - means[ chIdx ] ) * 
				  (rz_in[ idx ] - rop_means[ chIdx ]);

	}
}

/*
	R(I) = -1/(2 * pow( variance + epsilon, 3/2)) * (1/m)
				Sigma ( 2 ( x_i - mu ) ( Rx_i - (1/m) Sigma (Rx_j) )
*/
GLOBAL void ker_rI_scale ( real *variances, real *output, 
	int height, int width, int samples, int channels, real epsilon )
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	if (idx < (height * width * channels * samples ) ){
		int chIdx = idx / (height * width * samples); 
		output[ idx ] *=  -1. / ( samples * height * width * samples * height * width * pow(epsilon + variances [idx], (real)1.5)); 
	}
}

GLOBAL void ker_I_scale
		(real *variances, real epsilon, int imgSize, int samples, 
		real *scaledVariances)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	if (idx < imgSize){
		scaledVariances[ idx ] = 1. / ( samples * sqrt( variances[ idx ] + epsilon) ); 	
	}
}

GLOBAL void ker_rzout_scale ( real *zout, real *sums, real *output, 
	int height, int width, int channels, int samples, 
	real *variances, real epsilon ) {
		
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx < height * width * channels * samples ){
		int chIdx = idx /( samples * height * width ); 
		output[ idx ] = zout[ idx ] * sums[ chIdx ]; 	
		output[ idx ] /= (samples * height * width) * sqrt(variances[ chIdx ] + epsilon);
	}
}

GLOBAL void ker_rop_helper (
		real *scaledVariances, real *rII, int channels, int height, int width, int samples)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	if (idx < (samples * channels * height * width) ){
		int chIdx = idx / (height * width * samples); 

		rII[ idx ] *= scaledVariances[ chIdx ] ; 

	}
}

GLOBAL void ker_compute_rII ( 
	real *output, real *variances, real epsilon,
	int height, int width, int channels, int samples ){

	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	if (idx < (samples * channels * height * width) ){
		int chIdx = idx / (height * width * samples); 

		output[ idx ] *= (1. / ( sqrt( variances[ chIdx ] + epsilon ) * height * width * samples ));
	}
}

/*
	Init Means and Variances Here. 
*/
GLOBAL void ker_init_mean_variance( real *means, real *variances, int count )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < count) {
		means[ idx ] = 0; 
		variances[ idx ] = 1; 
	}
}

void initMeanVariances( real *means, real *variances, int channels )
{
	int blocks = (channels + BLOCK_SIZE - 1) / BLOCK_SIZE; 

	ker_init_mean_variance <<< blocks, BLOCK_SIZE >>> 
		(means, variances, channels); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 
}



/*
Compute batch mean and variance
This type is called Spatial Batch Normalization... 
BatchSize = N * H * W
Features = Channels
*/
void computeBatchMeanVariance( real *input, int height, int width, int channels, int samples, 
	real *output, real *batchMean, real *batchVariance,  real *devPtr, 
	real batchNormMomentum, real *datasetMean, real *datasetVariance ){

	real alpha, beta; 
	int blocks = (height * width * samples * channels + BLOCK_SIZE - 1) / BLOCK_SIZE ; 

	real *oneVector = devPtr; 
	real *nextDevPtr = oneVector + samples * height * width;


	//begin
	blocks = (height * width * samples + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	kerInitOneVector <<< blocks, BLOCK_SIZE>>> 
   	( oneVector, samples * height * width);  
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	//mean
	// oneVector * input( samples * height * width X channels )
	alpha = 1.; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, channels, samples * height * width,	
								&alpha, oneVector, 1, input, samples * height * width, 
								&beta, batchMean, 1) ); 
	alpha = 1./(real)(samples * height * width); 
	cublasCheckError( cublasDscal( cublasHandle, channels, &alpha, batchMean, 1 ) ); 

	//variance
	//one thread per cell. 
	blocks = (height * width * samples * channels + BLOCK_SIZE - 1) / BLOCK_SIZE ; 
	ker_compute_batch_variance <<<blocks, BLOCK_SIZE >>> 
		(input, height, width, channels, samples, batchMean, nextDevPtr ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	alpha = 1; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, channels, height * width * samples, 	
								&alpha, oneVector, 1, nextDevPtr, samples * height * width, 
								&beta, batchVariance, 1) ); 
	alpha = 1./(real)(samples * height * width); 
	cublasCheckError( cublasDscal( cublasHandle, channels, &alpha, batchVariance, 1 ) ); 

	//Batch Norm Momentum Term here. 
	alpha = batchNormMomentum; 
	cublasCheckError( cublasDscal( cublasHandle, channels, &alpha, datasetMean, 1) );
	cublasCheckError( cublasDscal( cublasHandle, channels, &alpha, datasetVariance, 1) );

	alpha = 1. - batchNormMomentum; 
	cublasCheckError( cublasDaxpy( cublasHandle, channels, &alpha, batchMean, 1, datasetMean, 1 ) );
	cublasCheckError( cublasDaxpy( cublasHandle, channels, &alpha, batchVariance, 1, datasetVariance, 1 ) );
}

/*
	Compute z_hat = z - mu / sqrt( variance + epsilon )
*/
void computeZOut( real *input, int rows, int channels, 
	real *means, real *variances, real epsilon, real *output ){

	int blocks = (rows * channels + BLOCK_SIZE - 1) / BLOCK_SIZE ; 

	ker_compute_ZHat <<<blocks, BLOCK_SIZE>>> 
		(input, rows, channels, means, variances, epsilon, output ); 	
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}

void computeBatchNormForward( real *input, int height, int width, int channels, int samples, 
	real *output, int meansOffset, int variancesOffset, real epsilon, real *devPtr, real *hostPtr, 
	EVAL_TYPE forTesting, real batchNormMomentum, int runningMeansOffset, int runningVariancesOffset ) {

	real *batch_mean = output + meansOffset; 
	real *batch_variance = output + variancesOffset;
	real *dataset_mean = output + runningMeansOffset; 
	real *dataset_variance = output + runningVariancesOffset; 
	real *nextDevPtr = devPtr; 


/*
	fprintf( stderr, "Means: %d, Variance: %d, DMean: %d, DVar: %d \n", 
							meansOffset, variancesOffset, runningMeansOffset, runningVariancesOffset ); 
*/

	if (forTesting == MODEL_TRAIN) {

		computeBatchMeanVariance( input, height, width, channels, samples, 
			output, batch_mean, batch_variance, nextDevPtr, batchNormMomentum, dataset_mean, dataset_variance); 

/*
		copy_host_device( hostPtr, dataset_mean, sizeof(real)*channels, cudaMemcpyDeviceToHost, 
			ERROR_MEMCPY_DEVICE_HOST ); 
		fprintf( stderr, "Means: \n");
		print2DMatrix( hostPtr, 1, channels ); 

		copy_host_device( hostPtr, dataset_variance, sizeof(real)*channels, cudaMemcpyDeviceToHost, 
			ERROR_MEMCPY_DEVICE_HOST ); 
		fprintf( stderr, "Variances: \n");
		print2DMatrix( hostPtr, 1, channels ); 
*/

	} else {

/*
		copy_host_device( hostPtr, dataset_mean, sizeof(real)*channels, cudaMemcpyDeviceToHost, 
			ERROR_MEMCPY_DEVICE_HOST ); 
		fprintf( stderr, "Means (Testing): \n");
		print2DMatrix( hostPtr, 1, channels ); 

		copy_host_device( hostPtr, dataset_variance, sizeof(real)*channels, cudaMemcpyDeviceToHost, 
			ERROR_MEMCPY_DEVICE_HOST ); 
		fprintf( stderr, "Variances (Testing): \n");
		print2DMatrix( hostPtr, 1, channels ); 
*/
	
	}

#ifdef DEBUG_BATCH_NORM
fprintf( stderr, "Batch Normalization Means/Variances ... \n"); 
fprintf( stderr, "Height: %d, Width: %d, Channels: %d, Samples: %d \n", 
					height, width, channels, samples ); 
#endif

	if (forTesting == MODEL_TRAIN) {
		computeZOut( input, height * width * samples, channels, batch_mean, batch_variance, 
			epsilon, output ); 
	} else {
		computeZOut( input, height * width * samples, channels, dataset_mean, dataset_variance, 
			epsilon, output ); 
	}
}

/*
	df/dZin = ( m df/dzHat_i - Sigma df/dzHat_i - Zhat_i Sigma df/dzHat_j * zHat_j )
*/
void batchNormDerivativeHelper (
	real *delta, real *output, 
	real *z_out,
	real *sum_across_samples_1, real *sum_across_samples_2, 
	int height, int width, int channels, int samples,
	real *devPtr, real *hostPtr)
{
	real alpha, beta; 
	int numBlocks;
	
	real *oneVector = devPtr; 
	real *nextDevPtr = oneVector + samples * height * width;
	
	/*
		scale --> m * df / dzHat_i
	*/
	alpha = (real)(samples * height * width); 
	cublasCheckError( cublasDscal( cublasHandle, height * width * channels * samples, &alpha, output, 1 ) ); 

	/*
		Sigma df / dzHat_i = sum_across_samples_1	
	*/
	numBlocks = (samples * height * width + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	kerInitOneVector <<< numBlocks, BLOCK_SIZE>>> 
   	( oneVector, samples * height * width);  
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	alpha = 1.; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, channels, height * width * samples, 
								&alpha, oneVector, 1, delta, height * width * samples, 
								&beta, sum_across_samples_1, 1 ) ); 

	/*
		zHat_i * Sigma_{j = 1 : m} (df / dzHat_j) * zhat_j
	*/
	copy_device( nextDevPtr, z_out, sizeof(real) * height * width * channels * samples, 
			ERROR_MEMCPY_DEVICE_DEVICE ); 

	numBlocks = (height * width * channels * samples + BLOCK_SIZE -1) / BLOCK_SIZE ; 
	ker_hadamard<<<numBlocks, BLOCK_SIZE>>> 
				(delta, height * width * channels * samples, nextDevPtr ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

/*
fprintf( stderr, "err  ..\n"); 
copy_host_device( hostPtr, delta, sizeof(real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 

fprintf( stderr, "zout  ..\n"); 
copy_host_device( hostPtr, z_out, sizeof(real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 

fprintf( stderr, "err * zout ..\n"); 
copy_host_device( hostPtr, nextDevPtr, sizeof(real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 
*/

	alpha = 1.; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, channels,  height * width * samples, 
								&alpha, oneVector, 1, nextDevPtr, height * width * samples, 
								&beta, sum_across_samples_2, 1 ) ); 
}

void batchNormROpHelper (
	real *rdelta, real *delta, real *output, 
	real *z_out, real *rz_out, 
	real *sum_across_samples_1, real *sum_across_samples_2, 
	real *sum_across_samples_3, 
	int height, int width, int channels, int samples,
	real *devPtr)
{
	real alpha, beta; 
	int numBlocks;
	
	real *oneVector = devPtr; 
	real *nextDevPtr = oneVector + samples * height * width;
	
	/*
		Sigma df / dzHat_i = sum_across_samples_1	
	*/
	numBlocks = (samples * height * width + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	kerInitOneVector <<< numBlocks, BLOCK_SIZE>>> 
   	( oneVector, samples * height * width);  
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	alpha = 1.; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, channels, height * width * samples, 
								&alpha, oneVector, 1, rdelta, height * width * samples, 
								&beta, sum_across_samples_1, 1 ) ); 

	/*
		RzHat_i * Sigma_{j = 1 : m} (df / dzHat_j) * zhat_j
	*/
	copy_device( nextDevPtr, z_out, sizeof(real) * height * width * channels * samples, 
			ERROR_MEMCPY_DEVICE_DEVICE ); 

	numBlocks = (height * width * channels * samples + BLOCK_SIZE -1) / BLOCK_SIZE ; 
	ker_hadamard<<<numBlocks, BLOCK_SIZE>>> 
				(delta, height * width * channels * samples, nextDevPtr ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	alpha = 1.; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, channels,  height * width * samples, 
								&alpha, oneVector, 1, nextDevPtr, height * width * samples, 
								&beta, sum_across_samples_2, 1 ) ); 

	/*
		z_out Sigma { z_out * rdelta + delta * rz_out }
	*/
	numBlocks = (height * width * channels * samples + BLOCK_SIZE -1) / BLOCK_SIZE ; 
	ker_hadamard_2<<<numBlocks, BLOCK_SIZE>>> 
				(delta, rz_out, z_out, rdelta, height * width * channels * samples, nextDevPtr ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	alpha = 1.; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, channels,  height * width * samples, 
								&alpha, oneVector, 1, nextDevPtr, height * width * samples, 
								&beta, sum_across_samples_3, 1 ) ); 
}


/*
	Derivative Computation Here. 

	
	df/dZin = [ ( m df/dzHat_i - Sigma df/dzHat_i - Zhat_i Sigma {df/dzHat_j * zHat_j} ) ]
				 --------------------------------------------------------------------------
                                    m * sqrt( epsilon + variance )

	df / dzHat =   df 
					 -------- * gamma
					   dZout

	zHat = 			(Zin - mu)
			----------------------------
			  sqrt( epsilon + variance )

*/

void computeBatchNormDerivative( 
	real *delta, int height, int width, int channels, int samples, 
	real *means, real *variance, real *output, 
	real *zout, real epsilon, real *devPtr, real *dx, real *hostPtr ){

	int numBlocks;
	
	real *sum_across_samples_1 = devPtr;
	real *sum_across_samples_2 = sum_across_samples_1 + channels ; 
	real *nextDevPtr = sum_across_samples_2 + channels ;

/*
fprintf (stderr, "Zout --> \n"); 
copy_host_device( hostPtr, zout, sizeof(real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 

fprintf (stderr, "Delta --> \n"); 
copy_host_device( hostPtr, delta, sizeof(real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 

fprintf( stderr, "Means --> \n"); 
copy_host_device( hostPtr, means, sizeof(real) * channels , 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, channels, 1 ); 

fprintf( stderr, "Variances--> \n"); 
copy_host_device( hostPtr, variance, sizeof(real) * channels , 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, channels, 1 ); 
*/

	//begin
	copy_device( output, delta, sizeof(real) * height * width * channels * samples, 
		ERROR_MEMCPY_DEVICE_DEVICE ); 

	//Helper here. 
	batchNormDerivativeHelper( delta, output, 
		zout, //rzout,
		sum_across_samples_1, sum_across_samples_2, 
		height, width, channels, samples,
		nextDevPtr, hostPtr ); 

/*
fprintf( stderr, "... m * err \n"); 
copy_host_device( hostPtr, output, sizeof(real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 

fprintf( stderr, "... sum_1 \n"); 
copy_host_device( hostPtr, sum_across_samples_1, sizeof(real) * channels, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, 1, channels ); 

fprintf( stderr, "... sum_2 \n"); 
copy_host_device( hostPtr, sum_across_samples_2, sizeof(real) * channels, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, 1, channels ); 
*/


	/* 
		Compute the backpropation of BatchNorm Layer here. 
	*/
	numBlocks = (height * width * channels * samples + BLOCK_SIZE -1) / BLOCK_SIZE ; 
	ker_backprop_batch_norm <<<numBlocks, BLOCK_SIZE>>> 
		( sum_across_samples_1, sum_across_samples_2, zout, 
			output, variance, dx, epsilon, samples, channels, height * width ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}

/*
	ROp Computatio here. 
	assumes that Z/RZ are of 2 * volumn storage size. 
	1. 1 unit of volumn stores the z_hat
	2. 2 unit of volumn is used to store means, variances
*/

/*
	Rz_hat = 
		1 / (sqrt( variance + epsilon) * { Rz_i - Rmu } + 
		(z_i - mu) / pow(variance + epsilon, 1.5) {(m-1)/m Sigma 2 * (z_j - mu) [ Rz_j - Rmu ] }

*/

void computeROpBatchNormForward (real *z, real *zout, real *rz_in, real *rz_out, real *devPtr, real *hostPtr,  
	real epsilon, int height, int width, int channels, int samples, int batchSize )
{
	real *means 	 = zout + height * width * channels * batchSize ; 
	real *variances = means + channels; 

	real *oneVector 			= devPtr; 
	real *rop_means 			= oneVector + height * width * samples;
	real *rop_means_forward = rop_means + channels;
	real *nextDevPtr 			= rop_means_forward + channels; 

	real alpha, beta; 
	int blocks;

/*
fprintf( stderr, "ComputeROPBatchNormForward.... inputs... %d \n", batchSize); 
fprintf( stderr, "Means... \n"); 
copy_host_device( hostPtr, means, sizeof(real) * channels, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, channels, 1 ); 

fprintf( stderr, "Variances... \n"); 
copy_host_device( hostPtr, variances, sizeof(real) * channels, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, channels, 1 ); 

fprintf( stderr, "Z_in ... \n");
copy_host_device( hostPtr, z, sizeof(real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels); 

fprintf( stderr, "Z_out ... \n");
copy_host_device( hostPtr, zout, sizeof(real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels); 

fprintf( stderr, "RZ_in ... \n");
copy_host_device( hostPtr, rz_in, sizeof(real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels); 
*/

	//begin
	blocks = (height * width * samples + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	kerInitOneVector <<< blocks, BLOCK_SIZE>>> 
   	( oneVector, samples * height * width);  
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	//Compute Rmu
	alpha = 1.; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, channels, height * width * samples, 	
								&alpha, oneVector, 1, rz_in, samples * height * width, 
								&beta, rop_means, 1) ); 

	alpha = 1./(real)(samples * height * width); 
	cublasCheckError( cublasDscal( cublasHandle, channels, &alpha, rop_means, 1 ) ); 

/*
	fprintf( stderr, "ROP Means... \n"); 
	copy_host_device( hostPtr, rop_means, sizeof(real) * channels, 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
	print2DMatrix( hostPtr, 1, channels ); 
*/

	//First Half.
	// R(x) - 1/m Sigma R(x)
	blocks = (height * width * channels * samples + BLOCK_SIZE -1) / BLOCK_SIZE ; 
	ker_compute_rop_first <<<blocks, BLOCK_SIZE>>> 
		(rz_in, height * width * samples, channels, rop_means, rz_out); 	
	cudaThreadSynchronize (); 
	cudaCheckError (); 
	
	//second half.
	//z_in, means, and the monsterous term here.
	/*
		(x_i - mu) * (R(x) - 1/m Sigma R(x) ) = SECOND
	*/
	
	ker_compute_rop_second <<< blocks, BLOCK_SIZE >>>
		(z, means, rz_out, nextDevPtr, height * width * samples, channels); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	//summation across images here. 
	/*
		Sigma SECOND
	*/
	alpha = 1.; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 	
							1, channels, height * width * samples, 
							&alpha, oneVector, 1, nextDevPtr, height * width * samples, 
							&beta, rop_means_forward, 1) ); 

	/*
		(x - mu) / pow( variance + epsilon, 1.5 )
	*/
	ker_compute_rop_second_scale <<< blocks, BLOCK_SIZE >>> 
		(rop_means_forward, z, means, nextDevPtr, height * width * samples, channels, variances, epsilon);
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	//Add first-half and second-half together
	//together compute ROp forward Pass. 

	//scale first * height * width here. 
	/*
		1 / sqrt( variacne + epsilon) * FIRST
	*/
	ker_compute_rop_first_scale <<< blocks, BLOCK_SIZE >>> 
		(rz_out, height * width * samples, channels, variances, epsilon); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

/*
	fprintf( stderr, "First term... \n"); 
	copy_host_device( hostPtr, rz_out, sizeof(real) * height * width * channels * samples, 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
	print2DMatrix( hostPtr, height * width * samples, channels ); 

	fprintf( stderr, "Second term... \n"); 
	copy_host_device( hostPtr, nextDevPtr, sizeof(real) * height * width * channels * samples, 
		cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
	print2DMatrix( hostPtr, height * width * samples, channels ); 
*/
	

	//rz_out += rop_means_forward
	alpha = -1.; 
	cublasCheckError( cublasDaxpy( cublasHandle, height * width * channels * samples, 
								&alpha, nextDevPtr, 1, 
								rz_out, 1 ) ); 
}



/*
	Rd_i = (I) * R( II ) + R( I ) * II
	
	I 	 = (1/m) * sqrt( variance + epsilon )
	R(II) = m * Rd_i+1 - Sigma (Rd_j+1) 
						  - Y_i Sigma ( Rd_j+1 Y_j )

	R(I) = -1/(2 * pow( variance + epsilon, 3/2)) * (m-1/m)
				Sigma ( 2 ( x_i - mu ) ( Rx_i - (1/m) Sigma (Rx_j) )
	II  = m d_i+1 - Sigma ( d_i+1 ) - y_i * Sigma (d_j+1 * Y_j )

*/

void computeROpBatchNormBackward ( 
	real *z_in, real *z_out, 
	real *rz_in, real *rz_out, 
	real *delta, real *rdelta, 
	real *dx_temp, real epsilon,
	real *means, real *variances, 
	int height, int width, int channels, int samples, int batchSize, 
	real *devPtr, real *hostPtr )
{
	//real *means 		= z_out + height * width * channels * batchSize ; 
	//real *variances 	= means  + channels; 

	real *output 					= devPtr; 
	real *rIIOutput 				= output + height * width * channels * samples; 
	real *sum_across_samples_1 = rIIOutput + height * width * channels * samples; 
	real *sum_across_samples_2 = sum_across_samples_1 + channels;
	real *sum_across_samples_3 = sum_across_samples_2 + channels;
	real *rop_means 				= sum_across_samples_3 + channels; 
	real *oneVector 				= rop_means + channels; 
	real *nextDevPtr 				= oneVector + height * width * samples; 

	real alpha, beta;
	int blocks;

//Print the inputs here... make sure that they are alright... 
/*
fprintf( stderr, "Beginning the backward Pass ROP Here..... \n\n"); 
fprintf( stderr, "Z_input.... \n");
copy_host_device( hostPtr, z_in, sizeof (real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 

fprintf( stderr, "Means.... \n");
copy_host_device( hostPtr, means, sizeof (real) * channels,
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, 1, channels); 

fprintf( stderr, "Variances.... \n");
copy_host_device( hostPtr, variances, sizeof (real) * channels,
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, 1, channels); 

fprintf( stderr, "Z_output.... \n");
copy_host_device( hostPtr, z_out, sizeof (real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 

fprintf( stderr, "RZ_in .... \n");
copy_host_device( hostPtr, rz_in, sizeof (real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 

fprintf( stderr, "RZ_out .... \n");
copy_host_device( hostPtr, rz_out, sizeof (real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 

fprintf( stderr, "dx...... \n");
copy_host_device( hostPtr, delta, sizeof (real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 

fprintf( stderr, "R{dx}...... \n");
copy_host_device( hostPtr, rdelta, sizeof (real) * height * width * channels * samples, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width * samples, channels ); 
*/

	//begin
	blocks = (height * width * samples + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	kerInitOneVector <<< blocks, BLOCK_SIZE>>> 
   	( oneVector, samples * height * width);  
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	//R( II )	
	//begin
	copy_device( rIIOutput, rdelta, sizeof(real) * height * width * channels * samples, 
			ERROR_MEMCPY_DEVICE_DEVICE ); 	

	batchNormROpHelper( rdelta, delta, rIIOutput, 
		z_out, rz_out, 
		sum_across_samples_1, sum_across_samples_2, 
		sum_across_samples_3,
		height, width, channels, samples, nextDevPtr ); 

	blocks = (height * width * channels * samples + BLOCK_SIZE -1) / BLOCK_SIZE ; 
	ker_backprop_batch_norm_no_dx <<<blocks, BLOCK_SIZE>>> 
		( sum_across_samples_1, sum_across_samples_2, sum_across_samples_3, z_out, rz_out, 
			rIIOutput, variances, epsilon, samples, channels, height * width ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 


	//Compute R( I )
	//rI --> 1 x (channels * height * width)

	//compute rop means
	alpha = 1; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, channels, height * width * samples, 
								&alpha, oneVector, 1, rz_in, height * width * samples, 
								&beta, rop_means, 1 ) ); 

	alpha = 1./((real) height * width * samples); 
	cublasCheckError( cublasDscal( cublasHandle, channels, &alpha, rop_means, 1 ) ); 
	

	ker_compute_RI <<<blocks, BLOCK_SIZE >>>
		(z_in, means, rz_in, rop_means, nextDevPtr,
			variances, epsilon,
			samples, channels, height, width);
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	//now sum across all the images here. 
	alpha = 1.; beta = 0.; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
								1, channels, height * width * samples, 	
								&alpha, oneVector, 1, nextDevPtr, height * width * samples, 
								&beta, sum_across_samples_3, 1) ); 

	blocks = (height * width * channels * samples + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	ker_rI_scale <<<blocks, BLOCK_SIZE >>> 
		(variances, sum_across_samples_3, height, width, samples, channels, epsilon); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	// II 
	// This is stored in delta_in
	// rdelta_in
	/*
		TODO: 
			We have to compute this term here, 
			We are no longer storing this during the backward pass instead
			we are storing the delta's coming into the batchNorm layer from the next layer... 

	copy_device( output, dx_temp, sizeof(real) * channels * height * width * samples, 
						ERROR_MEMCPY_DEVICE_DEVICE ); 
	*/
	copy_device( output, delta, sizeof(real) * channels * height * width * samples, 
			ERROR_MEMCPY_DEVICE_DEVICE ); 

	batchNormDerivativeHelper( delta, output, 
		z_out, 
		sum_across_samples_1, sum_across_samples_2, 
		height, width, channels, samples, nextDevPtr, hostPtr); 

	blocks = (height * width * channels * samples + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	ker_backprop_batch_norm_no_scale <<<blocks, BLOCK_SIZE>>> 
		( sum_across_samples_1, sum_across_samples_2, z_out, 
			output, variances, epsilon, samples, channels, height * width ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	// R( I ) * II
	ker_rop_helper <<<blocks, BLOCK_SIZE >>> 
		( sum_across_samples_3, output, channels, height, width, samples ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	//
	// Combined Output here. 
	// R(I) * II + I * R(II)
	//
	alpha = 1.;
	cublasCheckError( cublasDaxpy( cublasHandle, height * width * channels * samples, 
								&alpha, rIIOutput, 1, 
								output, 1 ) ); 

	//Update the rError term here. 
	copy_device( rdelta, output, sizeof(real) * height * width * channels * samples, 
		ERROR_MEMCPY_DEVICE_DEVICE ); 
}
