#ifndef __H_GEN_RANDOM__
#define __H_GEN_RANDOM__

#include <core/datadefs.h> 

typedef enum random_generator{
		RAND_NORMAL = 1, 
		RAND_UNIFORM
} RAND_GENERATOR; 

void sparseRandomMatrix( int rows, int cols, real density,   
   real *devPtr, real *hostPtr);
void getRandomVector (int n, real *hostPtr, real *devPtr, RAND_GENERATOR r);

template <class T>
void randomShuffle( T *idx, int m ); 
void genRandomVector( int *idx, int m, int n );


#endif
