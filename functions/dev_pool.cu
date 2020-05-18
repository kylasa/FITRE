#include <functions/dev_pool.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

#include <core/errors.h>

#include <nn/nn_decl.h>

#include <stdlib.h>
#include <float.h>


//Pooling Forward Pass... 
//one image -- one thread all pool entries per channel

GLOBAL void ker_avg_pool_layer_one_cell( int n, real *input, 
	int channels, int height, int width, int ksize, 
	real *output, real rOpScale ) {

	int myIdx = threadIdx.x + blockIdx.x * blockDim.x; 

	for (int index = myIdx; index < n; 
				index += blockDim.x * gridDim.x ) {
		
		int out_channel = index ;
		int in_channel = out_channel * height * width; 
		int idx = blockIdx.y; 

		real val = 0.;
		for (int r = 0; r < ksize; r ++){
			for (int c = 0; c < ksize; c ++ ){
				val += input[ c * height + r + in_channel + idx * channels * height * width]; 
			}
		}
		output[ out_channel + ( idx * channels ) ] = (val / (ksize * ksize)) * rOpScale; 
	}
}

GLOBAL void ker_avg_pool_layer( int n, real *input, 
	int channels, int height, int width, int ksize, int stride, int padding, 
	int samples, real *output, int p_height, int p_width, real rOpScale )
{
	int myIdx = threadIdx.x + blockDim.x * blockIdx.x;  // one thread per channel across all images. 
	int idx = blockIdx.y;  //samplesIdx

	for (int index = myIdx; index < n; 
			index += blockDim.x * gridDim.x ){ 

      int k = index % p_height; 
      int h_index = index / p_height; 
      int l = h_index % p_width; 

		int i = (k - 1) * stride + ksize - 2 * padding;			//in row
		int j = (l - 1) * stride + ksize - 2 * padding; 		//in col

		int out_channel = h_index / p_width; 
		int in_channel = out_channel * height * width; 

		real val = 0.;
		for (int r = 0 ; r < ksize ; r ++){
			for (int c = 0 ; c < ksize ; c ++ ){
				int frow = i + r; 
				int fcol = j + c; 
				if ((frow < 0) || (fcol < 0) || (frow >= height) || (fcol >= width)) continue; 
				val += input[ out_channel * samples * height * width + 
									idx * height * width + frow	 + 
									fcol * height ] ;
			}
		}

		output[ out_channel * p_height * p_width * samples + 
					idx * p_height * p_width + 
					l * p_height + k ] = (val / (ksize * ksize)) * rOpScale;
	}
}

GLOBAL void ker_max_pool_layer( int n, real *input, 
	int channels, int height, int width, int ksize, int stride, int padding, 
	int samples, real *output, int p_height, int p_width )
{
	int myIdx = threadIdx.x + blockDim.x * blockIdx.x;  // one thread per channel across all images. 
	int idx = blockIdx.y;  //samplesIdx
	real t; 

	for (int index = myIdx; index < n; 
			index += blockDim.x * gridDim.x ){ 

      int k = index % p_height; 
      int h_index = index / p_height; 
      int l = h_index % p_width; 

		int i = (k - 1) * stride + ksize - 2 * padding;			//in row
		int j = (l - 1) * stride + ksize - 2 * padding; 		//in col

		//int out_channel = index / (p_height * p_width);
		int out_channel = h_index / p_width; 
		int in_channel = out_channel * height * width; 

		real val = -DBL_MAX;
		for (int r = 0 ; r < ksize ; r ++){
			for (int c = 0 ; c < ksize ; c ++ ){
				int frow = i + r; 
				int fcol = j + c; 
				if ((frow < 0) || (fcol < 0) || (frow >= height) || (fcol >= width)) continue; 

				t = input[ out_channel * samples * height * width + 
									idx * height * width + frow	 + 
									fcol * height ];
				if (t > val) {
					val = t; 
				}
			}
		}

		output[ out_channel * p_height * p_width * samples + 
					idx * p_height * p_width + 
					l * p_height + k ] = val;
	}
}

GLOBAL void ker_rop_max_pool_layer( int n, real *rz, real *z, 
	int channels, int height, int width, int ksize, int stride, int padding, 
	int samples, real *output, int p_height, int p_width )
{
	int myIdx = threadIdx.x + blockDim.x * blockIdx.x;  // one thread per channel across all images. 
	int idx = blockIdx.y;  //samplesIdx
	real t = 0; 
	real rzVal = 0; 

	//for (int index = myIdx; index < n; 
	//		index += blockDim.x * gridDim.x ){ 
	int index = myIdx; 
	if (index < n) {

      int k = index % p_height; 
      int h_index = index / p_height; 
      int l = h_index % p_width; 

		int i = (k - 1) * stride + ksize - 2 * padding;			//in row
		int j = (l - 1) * stride + ksize - 2 * padding; 		//in col

		//int out_channel = index / (p_height * p_width);
		int out_channel = h_index / p_width; 
		int in_channel = out_channel * height * width; 

		real val = -DBL_MAX;
		for (int r = 0 ; r < ksize ; r ++){
			for (int c = 0 ; c < ksize ; c ++ ){
				int frow = i + r; 
				int fcol = j + c; 
				if ((frow < 0) || (fcol < 0) || (frow >= height) || (fcol >= width)) continue; 

					/*
					t = rz[ out_channel * samples * height * width + 
									idx * height * width + frow	 + 
									fcol * height ] ;
					*/
					t = z[ out_channel * samples * height * width + 
									idx * height * width + frow	 + 
									fcol * height ];
				if (t > val){
					val = t; 
					rzVal = rz[ out_channel * samples * height * width + idx * height * width + fcol * height + frow ];
				}
			}
		}

		output[ out_channel * p_height * p_width * samples + 
					idx * p_height * p_width + 
					l * p_height + k ] = rzVal; 
	}
}

void applyROpPoolLayer( real *rz, real *z, int samples, int out_channels, int height, int width, 
	int ksize, int stride, int padding, int poolFun, real *output, real rOpScale )
{
	int p_height = floor((float) ( height - ksize + 2 * padding ) / (float)stride + 1. ); 
	int p_width = floor( (float)( width - ksize  + 2 * padding) /(float) stride + 1.); 

	int num_kernels = p_height * p_width * out_channels; 
	int blocks = ( num_kernels + BLOCK_SIZE - 1) / BLOCK_SIZE; 

	//BUG FIX
	//dim3 grid (1, samples, blocks); 
	dim3 grid (blocks, samples, 1); 

#ifdef DEBUG_CNN
fprintf( stderr, "Number of Channels: %d, %d, %d \n", out_channels, p_height, p_width ); 
#endif

	switch( poolFun ){
		case AVG_POOL : 	
			if (p_height == 1 || p_width == 1) { 
				ker_avg_pool_layer_one_cell <<< grid, BLOCK_SIZE >>> 
					( num_kernels, rz, out_channels, height, width, ksize, output, rOpScale ); 
			} else {
				ker_avg_pool_layer <<< grid, BLOCK_SIZE >>> 
					( num_kernels, rz, out_channels, height, width, ksize, stride, padding, samples, 
						output, p_height, p_width, rOpScale ); 
			}
			cudaThreadSynchronize (); 
			cudaCheckError (); 
			break; 

		case MAX_POOL: 
			ker_rop_max_pool_layer <<< grid, BLOCK_SIZE >>> 
				( num_kernels, rz, z, out_channels, height, width, ksize, stride, padding, samples, 
						output, p_height, p_width ); 
			cudaThreadSynchronize (); 
			cudaCheckError (); 
			break; 

		default: 
			fprintf( stderr, "Undefined POOL FUNCTION.... please check... \n\n\n"); 
			exit (-1); 
	}
}



void applyPoolLayer( real *input, int samples, int out_channels, int height, int width, 
	int ksize, int stride, int padding, int poolFun, real *output, real rOpScale )
{
	int p_height = floor((float) ( height - ksize + 2 * padding ) / (float)stride + 1. ); 
	int p_width = floor( (float)( width - ksize  + 2 * padding) /(float) stride + 1.); 

	int num_kernels = p_height * p_width * out_channels; 
	int blocks = ( num_kernels + BLOCK_SIZE - 1) / BLOCK_SIZE; 


	//BUG-FIX
	//dim3 grid (1, samples, blocks); 
	dim3 grid (blocks, samples, 1); 

#ifdef DEBUG_CNN
fprintf( stderr, "Number of Channels: %d, %d, %d \n", out_channels, p_height, p_width ); 
#endif

	switch( poolFun ){
		case AVG_POOL : 	
			if (p_height == 1 || p_width == 1) { 
				ker_avg_pool_layer_one_cell <<< grid, BLOCK_SIZE >>> 
					( num_kernels, input, out_channels, height, width, ksize, output, rOpScale ); 
			} else {
				ker_avg_pool_layer <<< grid, BLOCK_SIZE >>> 
					( num_kernels, input, out_channels, height, width, ksize, stride, padding, samples, 
						output, p_height, p_width, rOpScale ); 
			}
			cudaThreadSynchronize (); 
			cudaCheckError (); 
			break; 

		case MAX_POOL: 
			ker_max_pool_layer <<< grid, BLOCK_SIZE >>> 
				( num_kernels, input, out_channels, height, width, ksize, stride, padding, samples, 
						output, p_height, p_width ); 
			cudaThreadSynchronize (); 
			cudaCheckError (); 
			break; 

		default: 
			fprintf( stderr, "Undefined POOL FUNCTION.... please check... \n\n\n"); 
			exit (-1); 
	}
}

GLOBAL void ker_pool_derivative( int n, real *delta, 
	int inSize, int kernel, 
	int channels, real *output, int samples )
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 

	if (idx < n) {
		int cId, cIdx, sCol, imgIdx, imgId, sRow; 

		//	Ids - according to the right side. 
		cId = idx / (inSize * inSize * samples); 
		cIdx = idx % (inSize * inSize * samples);

		imgId = cIdx / (inSize * inSize);
		imgIdx = cIdx % (inSize * inSize);

		sCol = imgIdx / inSize; 
		sRow = imgIdx % inSize; 

		for (int i = 0; i < kernel; i ++){ 
			for (int j = 0; j < kernel; j ++) {
				output[ cId * inSize * kernel * kernel * inSize * samples + 
							imgId * kernel * inSize * inSize * kernel + 
							sRow * kernel + 
							sCol * kernel * inSize * kernel + 
							i + j * kernel * inSize ] = delta[ idx ]; 
			}
		}
	}
}

/*
	One thread per pixel from the input view. 
*/
GLOBAL void ker_pool_derivative_in ( int n, real *delta, 
	int inSize, int outSize, int kernel, int stride, int padding, 
	int channels, real *output, int samples )
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 

	if (idx < n ){
		int cId, cIdx, inRow, inCol, imgIdx, imgId; 

		cId = idx / (inSize * inSize * samples);
		cIdx = idx % (inSize * inSize * samples ); 

		imgId = cIdx / (inSize * inSize); 
		imgIdx = cIdx % (inSize * inSize); 

		inCol = imgIdx / inSize; 
		inRow = imgIdx % inSize; 
	
		output[ idx ] = 0;

		int outRow = 0, outCol = 0; 
		for (int x = 0 - padding; x < inSize; x += stride, outCol += 1){ 

			outRow = 0; 
			for (int y = 0 - padding; y < inSize; y += stride, outRow += 1) {
				if ((inCol >= x) && (inCol < (x + kernel)) && ((x + kernel) <= inSize)
						&& (inRow >= y) && (inRow < (y + kernel)) && ((y + kernel) <= inSize) ) {
				
					output[ idx ] += delta [ cId * outSize * outSize * samples + 
														imgId * outSize * outSize + 
														outCol * outSize + 
														outRow ]; 
				}
			}
		}
	}
}

/*
GLOBAL void ker_max_pool_derivative( int n, real *input, 
	int channels, int height, int width, int ksize, int stride, int padding, 
	int samples, real *output, int p_height, int p_width )
*/
GLOBAL void ker_max_pool_derivative ( int n, real *delta, real *z_in, 
	int channels, int height, int width, int ksize, int stride, int padding, 
	int samples, real *output, int p_height, int p_width )
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;  // one thread per channel across all images. 
	int idx, out_channel; 
	real t; 
	int tid;
	int maxTargetIdx = INT_MAX; 
	real maxVal = - DBL_MAX;

	if (index < n ) { 

      int k = index % p_height; 
      int h_index = index / p_height; 
      int l = h_index % p_width; 

		int i = (k - 1) * stride + ksize - 2 * padding;			//in row
		int j = (l - 1) * stride + ksize - 2 * padding; 		//in col

		out_channel = (index / (p_height * p_width * samples)) ;
		idx = (index / (p_height * p_width )) % samples; 
/*
		tid = out_channel * samples * height * width + 
									idx * height * width + j * p_height + i;
		maxVal = z_in[ tid ]; 
		maxTargetIdx = tid; 
*/
		for (int c = 0 ; c < ksize ; c ++ ){
			for (int r = 0 ; r < ksize ; r ++){

				int fcol = j + c; 
				int frow = i + r; 
				if ((fcol < 0) || (fcol >= width) || (frow < 0) || (frow >= height)) continue; 

				tid = out_channel * samples * height * width + 
									idx * height * width + frow	 + 
									fcol * height;
				t = z_in[ tid ]; 

				if (t > maxVal) {
					maxVal = t; 
					maxTargetIdx = tid;
				}
			}
		}


		atomicAdd( output + maxTargetIdx, 
						delta[ out_channel * samples * p_height * p_width + 
									idx * p_height * p_width + l * p_height + k ] ); 
	}
}

/*
GLOBAL void ker_max_pool_derivative ( int n, real *delta, real *z_in, 
	int inSize, int outSize, int kernel, int stride, int padding, 
	int channels, real *output, int samples )
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 
	int tid; 
	int maxIdx; 
	int maxTargetIdx; 
	real maxVal = DBL_MIN; 
	real val; 

	if (idx < n ){
		int cId, cIdx, sCol, imgIdx, imgId, sRow; 

		cId = idx / (outSize * outSize * samples); 
		cIdx = idx % (outSize * outSize * samples);

		imgId = cIdx / (outSize * outSize);
		imgIdx = cIdx % (outSize * outSize);

		sCol = imgIdx / outSize; 
		sRow = imgIdx % outSize; 

		for (int i = 0; i < kernel; i ++){ 
			for (int j = 0; j < kernel; j ++) {
				
				tid = cId * outSize * kernel * kernel * outSize * samples + 
							imgId * kernel * outSize * outSize * kernel + 
							sRow * kernel + 
							sCol * kernel * outSize * kernel + 
							i + j * kernel * outSize ;
				val = z_in[ tid ];
				if( val > maxVal ){ 
					maxVal = val; 
					maxTargetIdx = tid; 	
				}
			}
		}

		//Max value computation is done. 
		output[ maxTargetIdx ] = delta[ idx ]; 	
	}
}
*/


void computePoolDerivative( real *delta, int sSize, 
	int channels, real *output, int kernelSize, int samples) {

	int count = sSize * sSize * channels * samples; // one thread per channel per image
	int blocks = ( count + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 

//fprintf( stderr, "******* Count: %d, BLOCKS: %d, sSize: %d, kernelSize: %d, channels: %d, samples: %d \n\n\n\n\n", 
						//count, blocks, sSize, kernelSize, channels, samples ); 
	
	ker_pool_derivative <<< blocks, BLOCK_SIZE >>> 
		( count, delta, sSize, kernelSize, channels, output, samples); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}

void computePoolDerivative_in( real *delta, int outSize, int inSize,  
	int channels, real *output, int kernelSize, int stride, int padding, int samples) {

	int count = inSize * inSize * channels * samples; // one thread per channel per image
	int blocks = ( count + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 
	
	ker_pool_derivative_in <<< blocks, BLOCK_SIZE >>> 
		( count, delta, inSize, outSize, kernelSize, stride, padding, channels, output, samples );
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}

void computeMaxPoolDerivative( real *delta, real *z_in, int channels, 
	int height, int width, int kernel, int stride, int padding, 
	int p_height, int p_width, int samples, real *output )
{
	int num_kernels = p_height * p_width * channels * samples; 
	int blocks = ( num_kernels + BLOCK_SIZE - 1) / BLOCK_SIZE; 

	//dim3 grid (1, samples, blocks); 
	//dim3 grid (blocks, samples); 

	cuda_memset( output, 0, sizeof(real) * height * width * channels * samples, 
			ERROR_MEMSET ); 

/*
fprintf( stderr, " height: %d, width: %d, channels: %d, samples: %d, outHeight: %d, outWidth: %d, kernel: %d, stride: %d, padding: %d \n\n", 
						height, width, channels, samples, p_height, p_width, kernel, stride, padding ); 
*/

	ker_max_pool_derivative <<< blocks, BLOCK_SIZE >>> 
		( num_kernels, delta, z_in, channels, height, width, kernel, stride, padding, 
			samples, output, p_height, p_width ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 
}

/*
void computeMaxPoolDerivative( real *delta, real *z_in, int outSize, int inSize,  
	int channels, real *output, int kernelSize, int stride, int padding, int samples) {

	int count = outSize * outSize * channels * samples; // one thread per channel per image
	int blocks = ( count + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 

	cuda_memset( output, 0, sizeof(real) * inSize * inSize * channels * samples, 
			ERROR_MEMSET ); 

	ker_max_pool_derivative <<< blocks, BLOCK_SIZE >>> 
		( count, delta, z_in, inSize, outSize, kernelSize, stride, padding, channels, output, samples );
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}

*/
