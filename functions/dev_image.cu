
#include <functions/dev_image.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

//
// one thread per source image pixel.
//dimensions = samples * channels * height * width -- Column Major format 
//
GLOBAL void ker_im2col( const int n, const real* data_im, 
	const int height, const int width, const int ksize, const int pad, 
	const int stride, const int height_col, const int width_col, int channels, real *data_col, int samples )
{
	int imgIdx = blockIdx.y; 
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; 
			index < n; index += blockDim.x * gridDim.x ){
		
		int h_out = index % height_col; 
		int h_index = index / height_col; 
		int w_out = h_index % width_col; 
		int channel_in = h_index / width_col; 
		int channel_out = channel_in * ksize * ksize; 

		int h_in = h_out * stride - pad; 
		int w_in = w_out * stride - pad; 


		//output pointer
		real *data_col_ptr = data_col; 
		data_col_ptr += channel_out * height_col * width_col * samples + 
							imgIdx * height_col * width_col + w_out * height_col + h_out; 


		//input pointer.
		const real* data_im_ptr = data_im + imgIdx * height * width; 
		data_im_ptr += (channel_in * width * samples + w_in) * height+ h_in; 

		for (int j = 0; j < ksize; j ++){ //columns
			for (int i = 0; i < ksize; i ++) { //rows
				int h = h_in + i; 
				int w = w_in + j; 

				*data_col_ptr = ((h >= 0) && (w >= 0) && (h < height) && (w < width )) ?  
						data_im_ptr[ j * height + i ] : 0; 
				data_col_ptr += height_col * width_col * samples;
			}
		}
	}
}

GLOBAL void ker_im2col_row_major( const int n, const real* data_im, 
	const int height, const int width, const int ksize, const int pad, 
	const int stride, const int height_col, const int width_col, int channels, real *data_col, int samples )
{
	int imgIdx = blockIdx.y; 
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; 
			index < n; index += blockDim.x * gridDim.x ){
		
		int h_out = index % height_col; 
		int h_index = index / height_col; 
		int w_out = h_index % width_col; 

		int channel_in = h_index / width_col; 
		int channel_out = channel_in * ksize * ksize; 

		int h_in = h_out * stride - pad; 
		int w_in = w_out * stride - pad; 

		//output pointer
		real *data_col_ptr = data_col; 
		data_col_ptr += channel_out * height_col * width_col * samples + 
							imgIdx * height_col * width_col + w_out + h_out * width_col; 


		//input pointer.
		const real* data_im_ptr = data_im + imgIdx * height * width; 
		data_im_ptr += (channel_in * width * samples + w_in) * height+ h_in; 

		for (int r = 0; r < ksize; r ++) { //rows
			for (int c = 0; c < ksize; c ++){ //columns
				int h = h_in + r; 
				int w = w_in + c; 

				*data_col_ptr = ((h >= 0) && (w >= 0) && (h < height) && (w < width )) ?  
						data_im_ptr[ c * height + r ] : 0; 
				data_col_ptr += height_col * width_col * samples;
			}
		}
	}
}


void getImageCols( real* data_im, const int channels, const int height, const int width, 
						const int ksize, const int pad, const int stride, real* data_col) 
{
	int height_col = (height + 2 * pad - ksize ) / stride + 1; 
	int width_col = (width + 2 * pad - ksize ) / stride + 1; 

	int num_kernels = channels * height_col * width_col; 
	//int num_kernels = 4;
	int blocks = (num_kernels + BLOCK_SIZE - 1) / BLOCK_SIZE; 

	//one thread per element kernel.
	ker_im2col <<<blocks, BLOCK_SIZE>>> 
		( num_kernels, data_im, height, width, ksize, pad, stride, height_col, width_col, channels, data_col, 1 ); 
	//ker_im2col <<<1, 28*28>>> 
	//	( 28*28, data_im, 32, 32, 5, 0, 1, 28, 28, data_col ); 
	cudaDeviceSynchronize ();
	cudaCheckError (); 
}

void getBatchImageCols( real* data_in, int samples, 
	const int channels, const int height, const int width, 
	const int ksize, const int pad, const int stride, real *data_col )
{
	int height_col = (height + 2 * pad - ksize ) / stride + 1; 
	int width_col = (width + 2 * pad - ksize ) / stride + 1; 

	int num_kernels = channels * height_col * width_col; 
	int x_blocks = (num_kernels + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	int y_blocks = samples;

	//BUG-FIX
	//dim3 blocks(1, y_blocks, x_blocks ); 
	dim3 blocks(x_blocks, y_blocks, 1); 
	
	//One thread per source image pixel
	//no. of grids = batch size... 
	ker_im2col <<< blocks, BLOCK_SIZE >>> 
		( num_kernels, data_in, height, width, ksize, pad, 
			stride, height_col, width_col, channels, data_col, samples ); 
	/*
	ker_im2col <<< blocks, BLOCK_SIZE >>> 
		( num_kernels, data_in, channels, height, width, ksize, pad, stride, 
			height_col, width_col, data_col, samples ); 
	*/
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}

void getBatchImageColsRowMajor( real* data_in, int samples, 
	const int channels, const int height, const int width, 
	const int ksize, const int pad, const int stride, real *data_col )
{
	int height_col = (height + 2 * pad - ksize ) / stride + 1; 
	int width_col = (width + 2 * pad - ksize ) / stride + 1; 

	int num_kernels = channels * height_col * width_col; 
	int x_blocks = (num_kernels + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	int y_blocks = samples;
	
	//BUG-FIX
	//dim3 blocks(1, y_blocks, x_blocks ); 
	dim3 blocks(x_blocks, y_blocks, 1); 
	
	//One thread per source image pixel
	//no. of grids = batch size... 
	ker_im2col_row_major<<< blocks, BLOCK_SIZE >>> 
		( num_kernels, data_in, height, width, ksize, pad, 
			stride, height_col, width_col, channels, data_col, samples ); 
	/*
	ker_im2col <<< blocks, BLOCK_SIZE >>> 
		( num_kernels, data_in, channels, height, width, ksize, pad, stride, 
			height_col, width_col, data_col, samples ); 
	*/
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}
