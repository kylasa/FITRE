#ifndef __H_PRINT_UTILS__
#define __H_PRINT_UTILS__

#include <core/datadefs.h> 

void printVector( real *src, int s, real *t , real *dscratch);
void printCustomVector( real *src, int s, int jump , real *dscratch);
void printIntVector( int *src, int s, int *t , real *dscratch);
void printHostVector( real *src, int s  , real *dscratch);
void writeMatrix (real *mat, int c);
void writeVector (real *mat, int c, char *file, int , real *dscratch);
void writeIntVector (int *mat, int c );
void writeSparseMatrix (real *dataPtr, int *rowIndex, int *colIndex, int m, int n, int nnz , real *dscratch);

real computeWeightSum( real *weights, int len , real *dscratch); 

int readVector( real *vec, int rows, char *file, int offset , real *dscratch);


void print2DMatrix( real *src, int h, int w, int ch, int img, int channels, int images);

void print2DMatrix( real *src, int h, int w );
void print3DMatrix( real *src, int c, int h, int w );
void print4DMatrix( real *src, int n, int c, int h, int w );

void writeIntMatrix ( real *mat, int images, int features );

#endif
