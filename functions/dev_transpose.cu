
#include <functions/dev_transpose.h>

/*
	Column major to Row-major format here. 
*/

GLOBAL void ker_transpose( real *input, int count, 
	int channels, int height, int width, int samples, real *output)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	int imgIdx, imgId, row, col, chIdx, chId; 

	if (idx < count){

		chId = idx / (samples * height * width); 
		chIdx = idx % (samples * height * width); 

		imgId = chIdx / (height * width); 
		imgIdx = chIdx % (height * width); 
	
		col = imgIdx / height; 
		row = imgIdx % height; 

		output[ chId * samples * height * width +
					imgId * height * width + 
					row * height + col ]  = input[ idx ]; 
	}
}

/*
	Row major to column major here. 
*/

GLOBAL void ker_transpose_rc( real *input, int count, 
	int channels, int height, int width, int samples, real *output)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	int imgIdx, imgId, row, col, chIdx, chId; 

	if (idx < count){

		chId = idx / (samples * height * width); 
		chIdx = idx % (samples * height * width); 

		imgId = chIdx / (height * width); 
		imgIdx = chIdx % (height * width); 
	
		col = imgIdx % width; 
		row = imgIdx / width; 

		output[ chId * samples * height * width +
					imgId * height * width + 
					col * height + row]  = input[ idx ]; 
	}
}

