
#include <utilities/derivative_test.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/gen_random.h>
#include <device/handles.h>

#include <core/datadefs.h>
#include <core/errors.h>

#include <functions/eval_gradient.h>
#include <functions/eval_hessian_vec.h>
#include <functions/dev_initializations.h>

#include <utilities/print_utils.h>

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

/*
	(W,b) = 0.
	(Wr, br) = random point.
	(Wr', br') = random point.

	f(W,b) = model_eval @ (W,b)
	g(W,b) = gradient_eval @ (W,b)
	
	for (N points){
		(Wc, bc) = (W,b) + (Wr', br');

		f(Wc,bc) = model_eval @ (Wc, bc)
		f_error = f(Wc,bc) - { f(W,b) + g(W,b)*(Wr',br') + (1/2)*(Wr',br') * h(W,b) * (Wr',br') }
	
		//compute first order error
		//compute second order error.

		(Wr', br') = (Wr', br') / 2
	}
*/

void readVecFromFileDT( real *dev, real *host, char *f ) { 

   int rows = readVector( host, INT_MAX, f, 0, NULL);
   copy_host_device( host, dev, rows * sizeof(real), cudaMemcpyHostToDevice, 
      ERROR_MEMCPY_HOST_DEVICE );  
}

void derivativeTest( NN_MODEL *model, DEVICE_DATASET *data, SCRATCH_AREA *scratch )
{
	int numPoints = 30; 

	real *devPtr = scratch->nextDevPtr; 
	real *hostPtr = scratch->nextHostPtr; 
	real *pagedMem = scratch->pageLckWorkspace; 

	//device space
	real *W0 = devPtr; 
	real *Wr = W0 + model->pSize; 
	real *Wc = Wr + model->pSize;
	real *z = Wc + model->pSize; 
	real *dx = z + model->zSize; 
	real *gradient = dx + model->rFullSize; 
	real *nextDevPtr = gradient + model->pSize; 

	//host space
	real *d2error = hostPtr; 
	real *d3error = d2error + numPoints; 
	real *dx2 = d3error + numPoints; 
	real *dx3 = dx2 + numPoints;
	real *dxs = dx3 + numPoints;
	real *nextHostPtr = dxs + numPoints; 

	//page space
	real *ll0 = pagedMem; 
	real *llc = ll0 + 1;
	real *modelErr0 = llc + 1; 
	real *modelErrC= modelErr0 + 1; 
	real *firstOrderTerm = modelErrC+ 1; 
	real *secondOrderTerm = firstOrderTerm + 1; 
	real *nextPageLckPtr = secondOrderTerm + 1; 

	//auxilaries here. 
	int vecSize = model->pSize; 
	real alpha = 1; 
	
	//reset the scratch area
	scratch->nextDevPtr = nextDevPtr; 
	scratch->nextHostPtr = nextHostPtr; 
	scratch->nextPageLckPtr = nextPageLckPtr; 

	//Zero Point
	cuda_memset( W0, 0, sizeof(real) * vecSize, ERROR_MEMSET ); 
	cuda_memset( Wc, 0, sizeof(real) * vecSize, ERROR_MEMSET ); 

	readVecFromFileDT( W0, nextHostPtr, "./weights.txt" ); 

	//Random Point
	//getRandomVector( vecSize, NULL, Wr, RAND_UNIFORM ); 

    int numElements = model->pSize;
    int numBlocks = numElements / BLOCK_SIZE +
                           (( numElements % BLOCK_SIZE  == 0) ? 0 : 1 );
    kerInitOneVector <<< numBlocks, BLOCK_SIZE>>>
            ( Wr, numElements );
    cudaThreadSynchronize ();
    cudaCheckError ();
	readVecFromFileDT( Wr, nextHostPtr, "./weights2.txt" ); 


	//copy_device( Wc, Wr, sizeof(real) * vecSize, 
	//	ERROR_MEMCPY_DEVICE_DEVICE ); 

	//f0
	copy_device( data->weights, W0, sizeof(real) * vecSize, 
			ERROR_MEMCPY_DEVICE_DEVICE ); 
	evaluateModel( model, data, scratch, data->weights, ll0, modelErr0, SAMPLED_DATASET, TRAIN_DATA ); 
fprintf( stderr, "Done ....... f0, %e, modelErr: %e\n", *ll0, *modelErr0); 

	//g0
	copy_device( data->weights, W0, sizeof(real) * vecSize, 
			ERROR_MEMCPY_DEVICE_DEVICE ); 
	computeGradient( model, data, scratch, data->weights, 
			z, dx, gradient, nextPageLckPtr, nextPageLckPtr + 1, SAMPLED_DATASET ) ;
fprintf( stderr, "Done ........ g0\n"); 
   //printVector( z, 10, NULL, scratch->hostWorkspace );  
   //printVector( dx, 10, NULL, scratch->hostWorkspace );  

	memset( d2error, 0, sizeof(real) * numPoints ); 
	memset( d3error, 0, sizeof(real) * numPoints ); 
		
fprintf( stderr, "Begin .... \n"); 
	for (int i = 0; i < numPoints; i ++) {

		//add two points Wc = W0 + Wr
		copy_device( Wc, W0, sizeof(real) * vecSize, ERROR_MEMCPY_DEVICE_DEVICE ); 
		alpha = 1; 
		cublasCheckError( cublasDaxpy( cublasHandle, vecSize, &alpha, 
									Wr, 1,  //x
									Wc, 1 ) ); //y

		//f(c)
		copy_device( data->weights, Wc, sizeof(real) * vecSize, 
				ERROR_MEMCPY_DEVICE_DEVICE ); 
		evaluateModel( model, data, scratch, data->weights, llc, modelErrC, SAMPLED_DATASET, TRAIN_DATA ); 	

		//Wc * gradient
		cublasCheckError( cublasDdot( cublasHandle, vecSize, 
									Wr, 1, 
									gradient, 1, 
									firstOrderTerm ) ); 									

		//d2error[ i ] = *modelErrC - (*modelErr0 + *firstOrderTerm); 
		d2error[ i ] = fabs(( *llc ) - (*ll0 + *firstOrderTerm)) / fabs( *ll0 ); 
		//fprintf( stderr, "Analytical == %e, Numerical == %e, firstOrder: %e \n", 
		//						*modelErr0 + *firstOrderTerm, *modelErrC, *firstOrderTerm ); 

		//second order term computation
		//hessian * Wc
		//result is stored in the nextDevPtr
		copy_device( data->weights, W0, sizeof(real) * vecSize, 
				ERROR_MEMCPY_DEVICE_DEVICE ); 
		hessianVec( model, data, z, dx, Wr, data->weights, scratch, SAMPLED_DATASET ); 

		//Wc * (hessian * Wc )
		cublasCheckError( cublasDdot( cublasHandle, vecSize, 
										scratch->nextDevPtr, 1, 
										Wr, 1, 
										secondOrderTerm ) ); 
real temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, vecSize, scratch->nextDevPtr, 1, &temp )); 
fprintf( stderr, "HessianVec norm: %6.20f \n", temp ); 
	
		d3error[ i ] = fabs( ( *llc ) - (*ll0 + *firstOrderTerm + 0.5 * (*secondOrderTerm) )) / fabs( *ll0 ); 	
		fprintf( stderr,"f == %g, f0 == %g, firstOrder: %g, secondOrder == %g \n\t\t firstOrderErr: %g \t\t secondOrderErr: %g\n", 
								(*llc), 
								*ll0, 
								*firstOrderTerm, 
								*secondOrderTerm, 
								d2error[i], 
								d3error[i] ); 

		cublasCheckError( cublasDnrm2 ( cublasHandle, vecSize, 
									Wc, 1, nextPageLckPtr ) ); 
		dx2[ i ] = pow( *nextPageLckPtr, 2.); 
		dx3[ i ] = pow( *nextPageLckPtr, 3.); 
		dxs[ i ] = *nextPageLckPtr; 

		//Wr = Wr / 2
		alpha = 0.5; 
		cublasCheckError( cublasDscal( cublasHandle, vecSize, &alpha, Wr, 1 ) ); 

fprintf( stderr, "Done with ....... %d\n", i ); 
	}

	//write the results in a file here. 
	writeVector( d2error, numPoints, "./d2_errors.txt", 1, d2error ); 
	writeVector( d3error, numPoints, "./d3_errors.txt", 1, d3error ); 

	writeVector( dx2, numPoints, "./dx2_order.txt", 1, dx2 ); 
	writeVector( dx3, numPoints, "./dx3_order.txt", 1, dx3 ); 
	writeVector( dxs, numPoints, "./dxs.txt", 1, dx3 ); 
}
