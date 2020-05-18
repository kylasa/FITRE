
#include <functions/softmax_loss.h>

#include <functions/dev_dist_errors.h>
#include <functions/dev_initializations.h>

#include <utilities/reduce.h>
#include <utilities/print_utils.h>
#include <device/device_defines.h>

#include <device/cuda_utils.h>
#include <core/errors.h>

#include <device/handles.h>


GLOBAL void ker_compute_exp (real *matvec, int rows, int numclasses, 
				real *target, real *maxdots, real *probs )
{
	int idx =  blockIdx.x * blockDim.x + threadIdx.x; 
	int myRowId = idx ;

	//real sdata = 0; 
	real maxdot = 0; 
	
	for (int r = myRowId; r < rows; r += gridDim.x * blockDim.x ) {
		 maxdot = 0; 
		 for (int i = 0; i < numclasses; i += 1){
			if (maxdot < matvec[ i + r * numclasses ]) maxdot = matvec[ i + r * numclasses ]; 
	 	 }

		 maxdots[ r ] = maxdot; 

		 for (int i = 0; i < numclasses; i += 1){
			//if ((int)target[ r ] == (i + 1)) sdata += matvec[ i * rows + r ]; 
			//matvec[ i * rows + r ] = exp( matvec[ i * rows + r ]  - maxdot); 
			probs[ i + r * numclasses ] = exp( matvec[ i + r * numclasses ] - maxdot ); 
		 } 
	}
}

GLOBAL void ker_compute_probs( real *input, 
		const size_t rows, int cols)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	real sdata; 
   sdata = 0.;

	if (idx < rows){
		for (int c = 0; c < cols; c ++) {
			sdata += input [ c + idx * cols ]; 
		}
		for (int c = 0; c < cols; c ++) { 
			input [ c + idx * cols ] = input[ c + idx * cols ] / sdata + 1e-10;  //SK-4 TODO ***********
		}
	}
}

GLOBAL void ker_compute_ll (real *matvec, int rows, int numclasses, 
				real *target, real *indicators)
{
	int idx =  blockIdx.x * blockDim.x + threadIdx.x; 
	int myRowId = idx ;

	real sdata = 0; 
	real maxdot = 0; 
	
	for (int r = myRowId; r < rows; r += gridDim.x * blockDim.x ) {

		 indicators[ r ] = 0; 
		 for (int i = 0; i < numclasses; i += 1){
			if ((int)target[ r ] == (i + 1)){  
				sdata = matvec[ i + r * numclasses ]; 
				indicators[ r ] = - log(sdata); 
			}
		} 
	}
}


GLOBAL void ker_multinomial_dist( real *probs, real *distTarget, 
	int rows, int num_classes ) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	real classProb = 0; 
	int myClass = -1; 

	if (idx < rows) {
		for (int j = 0; j < num_classes; j ++ ) {
			//probs[ i + r * numclasses ] = exp( matvec[ i + r * numclasses ] - maxdot ); 
			if ( classProb < probs[ idx * num_classes + j ] ) {
				classProb = probs[ idx * num_classes + j ]; 
				myClass = j + 1; 
			}
		}

		distTarget[ idx ] = myClass; 
	}
}



void computeDistributionErrors ( real *probs, real *distTarget, real *distError, 
	int rows, int num_classes, real *devPtr, real *hostPtr ) {


/*
	int blocks; 
	blocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	ker_multinomial_dist <<< blocks, BLOCK_SIZE >>> 
		( probs, distTarget, rows, num_classes ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 
*/

	getMultinomialDistSample( probs, rows, num_classes, distTarget, devPtr ); 

/*
	copy_host_device( hostPtr, probs, sizeof(real) * rows * num_classes, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST); 
	fprintf( stderr, "Class Probabilities.. \n"); 
	print2DMatrix( hostPtr, num_classes, 4 ); 

	copy_host_device( hostPtr, distTarget, sizeof(real) * rows, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST); 
	fprintf( stderr, "Multinomial classes.... \n"); 
	print2DMatrix( hostPtr, 1, 10);  
*/

	computeCrossEntropyError( probs, rows, num_classes, distTarget, distError ); 

/*
	copy_host_device( hostPtr, distError, sizeof(real) * rows * num_classes, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST); 
	fprintf( stderr, "Distribution Errors to be back'ed are ... \n\n"); 
	print2DMatrix( hostPtr, num_classes, 4 ); 
*/
}

	


real computeProbsAndCrossEntropyLoss( real *input, real *target, 
	int rows, int num_classes, 
	real *probs, real *devPtr, real *pageLckPtr, real *host ){

	real *maxdots = devPtr; 
	real *indicators = maxdots + rows; 
	real *nextDevPtr = indicators + rows; 

#ifdef DEBUG_DETAILED
fprintf( stderr, "Zis.... \n"); 
copy_host_device( host, input, sizeof(real) * rows * num_classes, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( host, num_classes, rows );  
#endif

	//maxDots, numerators
	int blocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	ker_compute_exp <<< blocks, BLOCK_SIZE  >>> 
			( input, rows, num_classes, target, maxdots, probs); 
	cudaThreadSynchronize ();  
	cudaCheckError (); 

#ifdef DEBUG_DETAILED
fprintf( stderr, "probs ....... \n"); 
copy_host_device( host, probs, sizeof(real) * rows * num_classes, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( host, num_classes, rows );  
#endif

	//Probabilities... 
	ker_compute_probs <<< blocks, BLOCK_SIZE >>> 
		(probs, rows, num_classes);
	cudaThreadSynchronize ();  
	cudaCheckError (); 

#ifdef DEBUG_DETAILED
fprintf( stderr, "probabilities.... \n"); 
copy_host_device( host, probs, sizeof(real) * rows * num_classes, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( host, num_classes, rows );  

//for (int i = 0; i < num_classes; i ++) {
//	fprintf( stderr, "%g\n", log( host[ i ] ) ); 
//}
#endif
	
	//Log Likelihoods here. 
	ker_compute_ll <<< blocks, BLOCK_SIZE >>> 
		( probs, rows, num_classes, target, indicators ); 
	cudaThreadSynchronize ();  
	cudaCheckError (); 

#ifdef DEBUG_DETAILED
fprintf( stderr, "Indicators are as follows for probabilities.... \n"); 
copy_host_device( host, indicators, sizeof(real) * rows , 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( host, 1, rows );  
#endif

	//add the terms to generate log likelihood. 
	//myreduce( indicators, rows, nextDevPtr, pageLckPtr ); 

   //begin
   blocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE; 
   kerInitOneVector <<< blocks, BLOCK_SIZE>>> 
      ( nextDevPtr, rows);  
   cudaThreadSynchronize (); 
   cudaCheckError ();

	cublasCheckError( cublasDdot( cublasHandle, rows, indicators, 1, nextDevPtr, 1, pageLckPtr ) ); 
	
	return *(pageLckPtr) / rows; 
}

GLOBAL void ker_crossentropy_error ( real *input, int rows, int numclasses, 
	real *target, real *errors ){
	
	int rowId = threadIdx.x + blockIdx.x * blockDim.x; 

	if (rowId < rows) {
		for (int c = 0; c < numclasses; c ++) {
			if ( (c+1) == target[ rowId ] ) errors[ c + rowId * numclasses ] = input[ c + rowId * numclasses]  - 1.;
			else errors[ c + rowId * numclasses ] = input[ c + rowId * numclasses ] ;
		} 	
	}
}

//
// dE/do = p_i - y_i, E = crossEntropyLoss
//
void computeCrossEntropyError( real *input, int rows, int numclasses, 
	real *target, real *errors ){

	real alpha; 

	int blocks = (rows +  BLOCK_SIZE - 1) / BLOCK_SIZE; 
	ker_crossentropy_error<<< blocks, BLOCK_SIZE >>> 
		(input, rows, numclasses, target, errors ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	alpha = 1./(real)(rows); 
	cublasCheckError( cublasDscal( cublasHandle, rows * numclasses, &alpha, errors, 1 ) ); 
	
}

/*
	rz, probs, rError === numclasses * rows
*/

GLOBAL void ker_crossentropy_rop_error ( 
	real *rz, real *probs, int rows, int numclasses, real *rError ){

	int idx = threadIdx.x + blockDim.x * blockIdx.x; 
	if (idx < rows) {

		/*
		for (int i = 0; i < numclasses; i ++) {
			rError[ idx * numclasses + i ] = 
				probs[ idx * numclasses + i ] * ( 1. - probs[ idx * numclasses + i ] ) * 	
					rz[ idx * numclasses + i ];
		}
		*/

		// Sigma p_j * R{ z_j }
		real s = 0; 
		for (int j = 0; j < numclasses; j ++) {
			s += probs[ idx * numclasses + j ] * rz[ idx * numclasses + j ]; 
		}	

		for (int i = 0; i < numclasses; i ++) {
			rError[ idx * numclasses + i ] = 
				probs[ idx * numclasses + i ] * ( rz[ idx * numclasses + i ] - s ); 
		}
	}
}	

/*
	R{ dE/dO } = R{ p_i - y_i }
				  = R{ p_i }
				  = { p_i R{ z_i } - p_i Sigma(j)[ p_j R{ z_j } ] } 
*/

void computeROpCrossEntropyError_simple( real *rz, real *probs, 
	int rows, int numclasses, 
	real *rError ){

	real alpha; 

	int blocks = (rows + BLOCK_SIZE - 1)/BLOCK_SIZE; 
	ker_crossentropy_rop_error <<< blocks, BLOCK_SIZE >>> 
		( rz, probs, rows, numclasses, rError ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	alpha = 1./(real)(rows); 
	cublasCheckError( cublasDscal( cublasHandle, rows * numclasses, &alpha, rError, 1 ) ); 
	
}

GLOBAL void ker_compute_r_delta_l1( real *rz, real *probs, 
	real *target, real *rError, int rows, int numclasses, real *devPtr)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 

	if (idx < rows) {

		// Sigma p_j * R{ z_j }
		real s = 0; 
		real dl1 = 0; 
		real rdl1 = 0; 
		for (int j = 0; j < numclasses; j ++) {
			s += probs[ idx * numclasses + j ] * rz[ idx * numclasses + j ]; 
		}	

		// p_i ( Rz_i - Sigma_j( Rz_j * p_j) )
		for (int i = 0; i < numclasses; i ++) {
			devPtr[ idx * numclasses + i ] = 
				probs[ idx * numclasses + i ] * ( rz[ idx * numclasses + i ] - s ); 
		}

		// R{ p_i } * (y_i / p_i^2) = R{ delta^(l+1) }
		for (int i = 0; i < numclasses; i ++) {
			if ((i+1) == target[ idx ] )
				rdl1 += 
				( 1. / pow( probs[ idx * numclasses + i ], 2.) ) * devPtr[ idx * numclasses + i ]; 
		}

		//now compute R{ delta }
		/*
			R{ delta } = 
				R{ p_i } Sigma_j delta^(l+1) - p_i Sigma_j R{ delta^(l+1) } + R{ delta^(l+1)_i }
		*/
		s = 0;
		dl1 = 0; 
		for (int i = 0; i < numclasses; i ++) {
			s += devPtr[ idx * numclasses + i ]; 	
			if ((i+1) == target[ idx ])
				dl1 = -( 1. / probs[ idx * numclasses + i ] ); 
		}

		for (int i = 0; i < numclasses; i ++) {
			rError[ idx * numclasses + i ] = 
				(devPtr[ idx * numclasses + i ] * dl1) - 
				(probs[ idx * numclasses + i ]  * rdl1); 
			if (( i + 1) == target[ idx ] ){
				rError[ idx * numclasses + i ] += 
					( 1. / pow(probs[ idx * numclasses + i ], 2.0) ) * devPtr[ idx * numclasses + i ]; 
			}
		}
	}
}

/*
R{ dE / dO } = R{ Sigma_j da_j / dz_i * delta_j }
*/

void computeROpCrossEntropyError( real *rz, real *probs,  real *target, 
	int rows, int numclasses, real *rError, real *devPtr )
{

	int blocks = ( rows + BLOCK_SIZE - 1 ) / BLOCK_SIZE; 
	/*
		 = { y_i / ( p_i * p_i) } * R{ p_i }
		R{ p_i } = p_i Rz_i - p_i Sigma_j p_j * Rz_j
	*/
	ker_compute_r_delta_l1 <<< blocks, BLOCK_SIZE >>> 
			( rz, probs, target, rError, rows, numclasses, devPtr ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}


