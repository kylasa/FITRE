
#ifndef __H_IMAGE_FUNCTIONS__
#define __H_IMAGE_FUNCTIONS__

#include <core/datadefs.h>

/*
void getImageFromCols( const real* data_col, const int channels, 
   const int height, const int width, const int ksize, const int pad, 
   const int stride, real* data_im);

void getBatchImagesFromCols( const real* data_col, const int channels, 
   const int height, const int width, const int ksize, const int pad, 
   const int stride, real* data_im, int samples);
*/

void getImageCols( real* data_im, const int channels, const int height, const int width,
                  const int ksize, const int pad, const int stride, real* data_col);

void getBatchImageCols( real* data_in, int samples,
   const int channels, const int height, const int width,
   const int ksize, const int pad, const int stride, real *data_col );

void getBatchImageColsRowMajor( real* data_in, int samples,
   const int channels, const int height, const int width,
   const int ksize, const int pad, const int stride, real *data_col );

//void mergeColWeights( real *input, int rows, int cols, int channels, int samples );

#endif
