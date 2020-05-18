
#include <utilities/cnn_derivative_test.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/gen_random.h>
#include <device/handles.h>

#include <core/datadefs.h>
#include <core/errors.h>

#include <functions/cnn_gradient.h>
#include <functions/cnn_hessian_vec.h>
#include <functions/cnn_eval_model.h>
#include <functions/dev_initializations.h>

#include <utilities/print_utils.h>
#include <utilities/utils.h>

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

void cnnDerivativeTest( CNN_MODEL *model, DEVICE_DATASET *data, SCRATCH_AREA *scratch )
{
	int numPoints = 25; 

	real *devPtr = scratch->nextDevPtr; 
	real *hostPtr = scratch->nextHostPtr; 
	real *pagedMem = scratch->pageLckWorkspace; 

	//device space
	real *W0 = devPtr; 
	real *Wr = W0 + model->pSize; 
	real *Wc = Wr + model->pSize;

	//Gradient Space
	real *z = 			Wc + model->pSize; 
	real *dx = 			z + model->zSize; 
	real *gradient = 	dx + model->zSize; 

	//hv Space
	real *lossFuncErrors = gradient + model->pSize; 
	real *rz = 			lossFuncErrors + model->maxDeltaSize; 
	real *rerror = 	rz + model->zSize; 
	real *probs = 		rerror + model->maxDeltaSize; 
	real *Hv = 			probs + model->pSize;

	//Scratch if needed. 
	real *nextDevPtr = Hv + model->pSize; 

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
	real *firstOrderTermCum = firstOrderTerm + 1; 
	real *secondOrderTerm = firstOrderTermCum + 1; 
	real *secondOrderTermCum = secondOrderTerm + 1; 
	real *nextPageLckPtr = secondOrderTermCum + 1; 

	//auxilaries here. 
	int vecSize = model->pSize; 
	real alpha = 1; 
	real discard; 
	int offset, numSamples; 
	real start, total; 
	
	//reset the scratch area
	scratch->nextDevPtr = nextDevPtr; 
	scratch->nextHostPtr = nextHostPtr; 
	scratch->nextPageLckPtr = nextPageLckPtr; 

	//Zero Point
	cuda_memset( W0, 0, sizeof(real) * vecSize, ERROR_MEMSET ); 
	cuda_memset( Wc, 0, sizeof(real) * vecSize, ERROR_MEMSET ); 

	//Random Point
	getRandomVector( vecSize, NULL, Wr, RAND_UNIFORM ); 
	getRandomVector( vecSize, NULL, W0, RAND_UNIFORM ); 

	alpha = 0.1;
	cublasCheckError( cublasDscal( cublasHandle, vecSize, &alpha, Wr, 1 ) ); 
	cublasCheckError( cublasDscal( cublasHandle, vecSize, &alpha, W0, 1 ) ); 
fprintf( stderr, "Initialized Random Points... \n"); 

	memset( d2error, 0, sizeof(real) * numPoints ); 
	memset( d3error, 0, sizeof(real) * numPoints ); 
		
fprintf( stderr, "Begin .... \n"); 
	for (int i = 0; i < numPoints; i ++) {

		offset = 0; 
		numSamples = 0; 
		*modelErr0 = *modelErrC = 0; 
		*firstOrderTermCum = *secondOrderTermCum = 0; 
		start = total = 0; 
		start = Get_Time (); 
		for (int j = 0; j < (data->trainSizeX); j += model->batchSize){

			offset = j;
			if( (j + model->batchSize) <= data->trainSizeX )
				numSamples = model->batchSize; 
			else
				numSamples = data->trainSizeX % model->batchSize; 

			//f0, g0
			/*
			copy_device( data->weights, W0, sizeof(real) * vecSize, 
				ERROR_MEMCPY_DEVICE_DEVICE ); 
			*modelErr0 += computeCNNGradient( model, data, scratch, 
				z, dx, probs, lossFuncErrors, gradient, offset, numSamples); 
			*/

			//add two points Wc = W0 + Wr
			copy_device( Wc, W0, sizeof(real) * vecSize, ERROR_MEMCPY_DEVICE_DEVICE ); 
			alpha = 1; 
			cublasCheckError( cublasDaxpy( cublasHandle, vecSize, &alpha, 
									Wr, 1,  //x
									Wc, 1 ) ); //y

			//f(c)
			copy_device( data->weights, Wc, sizeof(real) * vecSize, 
				ERROR_MEMCPY_DEVICE_DEVICE ); 
			*modelErrC += evaluateCNNModel( model, data, scratch, 
							z, probs, lossFuncErrors, offset, numSamples);

			//second order term computation
			//hessian * Wc
			//result is stored in the nextDevPtr
			copy_device( data->weights, W0, sizeof(real) * vecSize, 
							ERROR_MEMCPY_DEVICE_DEVICE ); 
			*modelErr0 += computeCNNGradient( model, data, scratch, 
						z, dx, probs, lossFuncErrors, gradient, offset, numSamples, 0); 
			cnnHv( model, data, z, probs, lossFuncErrors, dx, Wr, Hv, offset, numSamples,
					nextDevPtr, nextHostPtr, 0 ); 

			//fprintf( stderr, "Offset: %d, samples: %d, j: %d, Err0: %g, ErrC: %g \n", 
			//			offset * 3 * 1024, numSamples, j, *modelErr0, *modelErrC ); 

			//Wr * gradient
			cublasCheckError( cublasDdot( cublasHandle, vecSize, 
									Wr, 1, 
									gradient, 1, 
									firstOrderTerm ) ); 									
			*firstOrderTermCum += *firstOrderTerm;

			//Wr * (hessian * Wr )
			cublasCheckError( cublasDdot( cublasHandle, vecSize, 
										Hv, 1, 
										Wr, 1, 
										secondOrderTerm ) ); 
			*secondOrderTermCum += *secondOrderTerm; 
		}
		fprintf( stderr, "\n");
		total = Get_Timing_Info( start ); 

		//Normalize by the size of the dataset. 
		*modelErrC /= (double)data->trainSizeX; 
		*modelErr0 /= (double)data->trainSizeX; 
		*firstOrderTermCum /= (double)data->trainSizeX; 
		*secondOrderTermCum /= (double)data->trainSizeX; 

		d2error[ i ] = fabs( (*modelErrC) - ((*modelErr0) + (*firstOrderTermCum))) / fabs(*modelErrC); 
		d3error[ i ] = fabs( ( *modelErrC ) - 	
									((*modelErr0) + (*firstOrderTermCum) + 0.5 * (*secondOrderTermCum) )) / fabs( *modelErrC ); 	
		fprintf( stderr, "First ORder Error: %e, %g \n", 
								*firstOrderTermCum, *firstOrderTermCum );
		fprintf( stderr, "Second ORder Error: %e, %g, %g \n", 
								*secondOrderTermCum, *secondOrderTermCum, 0.5 * (*secondOrderTermCum) ); 
		fprintf( stderr, "f == %g, f0 == %g, firstOrderErr: %g, secondOrderErr: %g\n", 
						(*modelErrC), 
						(*modelErr0),
						d2error[i], 
						d3error[i] ); 
		fprintf( stderr, " dg == %g, dHd == %g, estimated val == %g, error == %g \n", 
					*firstOrderTermCum, 
					0.5 * (*secondOrderTermCum), 
					((*modelErr0) + (*firstOrderTermCum) + 0.5 * (*secondOrderTermCum)), 
					(*modelErrC) - ((*modelErr0) + (*firstOrderTermCum) + 0.5 * (*secondOrderTermCum))  ) ; 

		cublasCheckError( cublasDnrm2 ( cublasHandle, vecSize, 
									Wr, 1, nextPageLckPtr ) ); 
		dx2[ i ] = pow( *nextPageLckPtr, 2.); 
		dx3[ i ] = pow( *nextPageLckPtr, 3.); 
		dxs[ i ] = *nextPageLckPtr; 

		//Wr = Wr / 2
		alpha = 0.5; 
		cublasCheckError( cublasDscal( cublasHandle, vecSize, &alpha, Wr, 1 ) ); 

fprintf( stderr, "Done with ....... %d in %f seconds \n\n\n", i, total ); 
	}

	//write the results in a file here. 
	writeVector( d2error, numPoints, "./d2_errors.txt", 1, d2error ); 
	writeVector( d3error, numPoints, "./d3_errors.txt", 1, d3error ); 

	writeVector( dx2, numPoints, "./dx2_order.txt", 1, dx2 ); 
	writeVector( dx3, numPoints, "./dx3_order.txt", 1, dx3 ); 
	writeVector( dxs, numPoints, "./dxs.txt", 1, dx3 ); 
}
