
#include <functions/dev_backprop_convolution.h> 

#include <device/device_defines.h>
#include <device/handles.h>
#include <device/cuda_utils.h>

#include <core/errors.h>
#include <utilities/print_utils.h>

// threads = height * width * imgChannels * samples
// assumes column major ordering for input and out. 
// weights are also stored in column major order. 
// Needs conversion of weights becuase of in and out channels
// MISMATCH... 
GLOBAL void kerBackPropConvolutionHelper( 
	real *delta, int dHeight, int dWidth, int channels,
	real *filter, int fHeight, int fWidth, 
	real *img, int height, int width, 
	int samples, int n )
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 

	if (idx < n) {
		int imgIdx = idx / (height * width * channels); 
		int imgThx = idx % (height * width * channels);
		int chIdx = imgThx / (height * width); 

		int myRow = (imgThx % (height * width)) % height; 
		int myCol = (imgThx % (height * width)) / width; 
		//int imgOffset = chIdx * height * width * samples + imgIdx * height * width; 
		//int deltaOffset = chIdx * dHeight * dWidth * samples + imgIdx * dHeight * dWidth;
		int imgOffset = imgIdx * height * width * fHeight * fWidth * channels + 
								chIdx * height * width * fHeight * fWidth  + 
								myCol * height + myRow; 
		int deltaOffset = chIdx * dHeight * dWidth + 	
									imgIdx * dHeight * dWidth * channels;

		int colLimits, rowLimits; 
		for (int c = myCol; c > myCol - fWidth; c -- ){
			for (int r = myRow; r > myRow - fHeight; r -- ) {
				colLimits = (c < dWidth) && (c >= 0); 
				rowLimits = (r >= 0) && ( r < dHeight ); 
				img[ imgOffset ] = ( rowLimits && colLimits ) ?  delta[ deltaOffset + c * dHeight + r ] : 0; 
				imgOffset += height * width; 
			}
		}
	}
}

/*
	Threads = Height * width * samples * (dChannels)
		
	This will generate a matrix of size: 
		(height * width * samples) X (dChannels * fHeight * fwidth). 

	Workload on each thread = fHeight * fWidth
*/

GLOBAL void kerBackPropConvolutionBatchHelper( 
	real *delta, int dHeight, int dWidth, int channels, 
	real *filter, int fHeight, int fWidth, 
	real *img, int height, int width, int padding, int samples, int n )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 

	if (idx < n ) {

		int chId = idx / (height * width * samples); 
		int chIdx = idx % (height * width * samples);

		int imgId 	= chIdx / (height * width ); 
		int imgIdx 	= chIdx % (height * width );

		int myRow = imgIdx % height; 
		int myCol = imgIdx / height;

		int imgOffset = chId * height * width * fHeight * fWidth * samples + 
							imgId * height * width + 
							myCol * height + myRow; 

		int deltaOffset = chId * samples * dHeight * dWidth + 
								imgId * dHeight * dWidth; 

		int colLimits, rowLimits; 
		for (int c = myCol + padding; c > myCol + padding - fWidth; c -- ){
			for (int r = myRow + padding; r > myRow + padding - fHeight; r -- ) {
				colLimits = (c < dWidth) && (c >= 0); 
				rowLimits = (r >= 0) && ( r < dHeight ); 

				img[ imgOffset ] = ( rowLimits && colLimits ) ?  delta[ deltaOffset + c * dHeight + r ] : 0; 
				imgOffset += height * width * samples; 
			}
		}
	}
}

/*
GLOBAL void kerBackPropConvolutionBatchHelper( 
	real *delta, int dHeight, int dWidth, int channels, 
	real *filter, int fHeight, int fWidth, 
	real *img, int height, int width, int padding, int samples, int n )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 

	if (idx < n ) {

		int chId = idx / (height * width * samples); 
		int chIdx = idx % (height * width * samples);

		int imgId 	= chIdx / (height * width ); 
		int imgIdx 	= chIdx % (height * width );

		int myRow = imgIdx % height; 
		int myCol = imgIdx / height;

		int imgOffset = chId * height * width * fHeight * fWidth * samples + 
							imgId * height * width + 
							myCol * height + myRow; 

		int deltaOffset = chId * samples * dHeight * dWidth + 
								imgId * dHeight * dWidth; 

		int colLimits, rowLimits; 
		for (int c = myCol + padding + 1 - fWidth; c < myCol + padding + 1; c ++ ){
			for (int r = myRow + padding + 1 - fHeight; r < myRow + padding + 1; r ++ ) {
				colLimits = (c < dWidth) && (c >= 0); 
				rowLimits = (r >= 0) && ( r < dHeight ); 

				img[ imgOffset ] = ( rowLimits && colLimits ) ?  delta[ deltaOffset + c * dWidth + r ] : 0; 
				imgOffset += height * width * samples; 
			}
		}
	}
}
*/



GLOBAL void kerReshapeMatrix( real *input, 
	int rightChannels, int leftChannels, int chunk, 
	real *output, int n ){

	int idx = threadIdx.x + blockDim.x * blockIdx.x; 

	if (idx < n){

		int nId = idx / (chunk * leftChannels); // 0 - rightChannels-1
		int nIdx = idx % (chunk * leftChannels );  // id within each N
		int cId = nIdx / chunk; // blocks of chunks
		int chunkId = nIdx % chunk; // chunkId within each Chunk

		output[ cId * (chunk * rightChannels) + nId * (chunk) + chunkId ] = 
			input[ idx ]; 
	}
}

void reshapeMatrix( real *input, int rightChannels, int leftChannels, 
		int chunk, real *output ){
	
	int threads = rightChannels * leftChannels * chunk; 
	int blocks = ( threads + BLOCK_SIZE - 1) / BLOCK_SIZE; 

	kerReshapeMatrix <<< blocks, BLOCK_SIZE >>> 
			( input, rightChannels, leftChannels, chunk, output, threads );
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}



//TO COMPUTE dZ
void backpropConvolution( real *delta, int dHeight, int dWidth, int dChannels, 
	real *filter, int fHeight, int fWidth,
	int height, int width, int padding, int channels, 
	 int samples, real *delta_1, real *devPtr, real *hostPtr)
{
	real *imgCols = devPtr; 
	real *nextDevPtr = imgCols + samples * height * width * fHeight * fWidth * dChannels; 
	real alpha, beta; 

	//convert to the expanded form here. 
	int threads= height * width * samples * dChannels ;
	int blocks = ( threads + BLOCK_SIZE - 1) / BLOCK_SIZE; 

/*
#ifdef DEBUG_DETAILED
fprintf( stderr, "Starting to compute dx_update\n"); 
fprintf( stderr, "Incoming delta.... \n"); 
copy_host_device( hostPtr, delta, sizeof(real)*dHeight * dWidth * dChannels, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print3DMatrix( hostPtr, dChannels, dHeight, dWidth ); 
#endif
*/
	
	//to the img2Col conversion of Delta... 
	kerBackPropConvolutionBatchHelper <<< blocks, BLOCK_SIZE >>> 
		( delta, dHeight, dWidth, dChannels, 
			filter, fHeight, fWidth, 
			imgCols, height, width, padding, 
			samples, threads); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 
/*
#ifdef DEBUG_DETAILED
fprintf( stderr, "ImgCol conversion of delta... \n"); 
copy_host_device( hostPtr, imgCols, sizeof(real) * height * width * channels * fHeight * fWidth, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, height * width, channels * fHeight * fWidth ); 
#endif
*/
	//weights reOrdering here. 
	threads = fHeight * fWidth * dChannels * channels; 
	blocks = (threads + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	kerReshapeMatrix <<<blocks, BLOCK_SIZE >>> 
		( filter, dChannels, channels, 
			fHeight * fWidth, nextDevPtr, threads ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

/*
#ifdef DEBUG_DETAILED
fprintf( stderr, "Reodering Weights to align the channels... \n"); 
copy_host_device( hostPtr, nextDevPtr, sizeof(real) * fHeight * fWidth * dChannels * channels, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print4DMatrix( hostPtr, dChannels, channels, fHeight, fWidth ); 
#endif
*/
	// perform the matrix multiplication to compute... 
	// delta for next layer... 
	// dZ = img2col ( delta ) * Weights (Reordered). 
	alpha = 1; beta = 0; 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
							samples * height * width, channels, fHeight * fWidth * dChannels, 
							&alpha, imgCols, samples * height * width, 
							nextDevPtr, fHeight * fWidth * dChannels, 
							&beta, delta_1, samples * height * width ) ); 

/*
#ifdef DEBUG_DETAILED
fprintf( stderr, "Updated value of the delta to be used later... \n"); 
copy_host_device( hostPtr, delta_1, sizeof(real) * height * width * channels, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print3DMatrix( hostPtr, channels, height, width ); 
#endif
*/
}
