#include <device/gen_random.h>
#include <device/cuda_utils.h>
#include <device/handles.h>
#include <core/errors.h>
#include <core/datadefs.h>

#include <time.h>

void sparseRandomMatrix( int rows, int cols, real density, 
	real *hostPtr, real *devPtr)
{
	int numElements = int (rows * cols * density ); 

	memset( hostPtr, 0, rows * cols * sizeof(real) ); 
	getRandomVector( numElements, hostPtr, devPtr, RAND_UNIFORM ); 

	randomShuffle<real>( hostPtr, rows * cols ); 

	copy_host_device( hostPtr, devPtr, sizeof(real) * rows * cols, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_DEVICE_HOST ); 
}

void getRandomVector (int n, real *hostPtr, real *devPtr, RAND_GENERATOR r) {

	//curandGenerator_t gen ;
	int m = n + n % 2;

	/* Create pseudo - random number generator */
	//curandCheckError ( curandCreateGenerator (&gen , CURAND_RNG_PSEUDO_DEFAULT ) );

	/* Set seed */
	//curandCheckError ( curandSetPseudoRandomGeneratorSeed ( gen , 1234ULL )) ;
	//curandCheckError ( curandSetPseudoRandomGeneratorSeed ( curandGeneratorHandle , time(NULL) )) ;

	/* Generate n floats on device */
	//standard normal distribution.
	if (r == RAND_NORMAL)
		curandCheckError ( curandGenerateNormalDouble ( curandGeneratorHandle, devPtr , m, 0, 1.)) ;
	else 
		curandCheckError ( curandGenerateUniformDouble ( curandGeneratorHandle, devPtr , m)) ;

	/* Copy device memory to host */
	if (hostPtr != NULL) {
		copy_host_device( hostPtr, devPtr, sizeof(real) * n, cudaMemcpyDeviceToHost,
			ERROR_MEMCPY_DEVICE_HOST );
	}

	/* Cleanup */
	//curandCheckError ( curandDestroyGenerator ( gen ) );
}

/*
Random Shuffle Here. 
https://stackoverflow.com/questions/15961119/how-to-create-a-random-permutation-of-an-array
*/
template <class T>
void randomShuffle( T *idx, int n)
{
	int j;
	T temp; 
	for (int i = n - 1; i >= 0; i --){
		j = rand () % (i+1); 	

		temp = idx[i]; 
		idx[i] = idx[j]; 
		idx[j] = temp;
	}
}


/*
Floyd's algorithm Here. 
https://stackoverflow.com/questions/1608181/unique-random-numbers-in-an-integer-array-in-the-c-programming-language
*/

void genRandomVector( int *idx, int m, int n ) {

	int in, im; 
	int rn, rm; 	
	im = 0; 

	for (in = 0; in < n && im < m; ++in ){
		rn = n - in; 
		rm = m - im; 

		if (rand () % rn < rm ){
			//idx[ im ++] = in + 1; 
			idx[ im ++] = in ; 
		}
	}

	if ( im != m ){
		fprintf( stderr, "Failed to generate required number of random numbers ... (%d, %d) ", im, m); 
		exit (-1); 
	}

	randomShuffle<int>( idx, m ); 
}

