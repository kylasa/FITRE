
#include <functions/model_reduction.h>
#include <functions/cnn_hessian_vec.h>

#include <core/errors.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

#include <utilities/print_utils.h>
#include <utilities/reduce.h>



/*
	m = gradient * vector + 0.5 * vector * hessianvec ( vector )
*/

real computeQuadraticModel( CNN_MODEL *model, DEVICE_DATASET *data, 
	real *z, real *probs, real *lossFuncErrors, real *dx,  real *gradient, 
	real *vector, real delta, int offset, int curBatchSize, real weightDecay, 
	real *devPtr, real *hostPtr, real *pageLckPtr )
{
	//locals
	real *gradVecDot = pageLckPtr; 
	real *vHv = gradVecDot + 1 ;
	real *nrmVec = vHv + 1; 

	real *hv = devPtr; 
	real *nextDevPtr = hv + model->pSize; 

	real *nextHostPtr = hostPtr; 

	int components = model->pSize; 

	real alpha, step; 

	//begin
	cnnHv( model, data, z, probs, lossFuncErrors, dx, vector, 
		hv, offset, curBatchSize, nextDevPtr, nextHostPtr, weightDecay ); 

#ifdef DEBUG_TRUST_REGION
	real temp = 0; 
	cublasCheckError( cublasDnrm2( cublasHandle, components, 
			hv, 1, &temp ) ); 
	fprintf( stderr, "Norm ( H * v, 2 ) = %.10f \n", temp ); 
#endif

	alpha = weightDecay;
	cublasCheckError( cublasDaxpy( cublasHandle, components, 
				&alpha, data->weights, 1, hv, 1) ); 

	cublasCheckError( cublasDdot( cublasHandle, components, 
			hv, 1, vector, 1, vHv) ); 


	//STUPID BUG..... 
	//WHY IN THE HELL IT WAS COMMENTED OUT>.. 
	cublasCheckError( cublasDnrm2( cublasHandle, components, 
								vector, 1, nrmVec ) ); 
#ifdef DEBUG_TRUST_REGION
	fprintf( stderr, "Model Reduction: vHv: %f \n", *vHv ); 
#endif

	if ( *vHv < 0 ) {
		//Negative Curvature. 
		cublasCheckError( cublasDdot( cublasHandle, components, 
				gradient, 1, vector, 1, gradVecDot ) ); 

		alpha = delta / (*nrmVec) ; 
		cublasCheckError( cublasDscal( cublasHandle, components, 	
								&alpha, vector, 1 ) ); 

		return  0.5 * (*vHv) * alpha * alpha - (*gradVecDot) * alpha ;
		
	} else {
		// Positive Curvature. 
		cublasCheckError( cublasDdot( cublasHandle, components, 
				gradient, 1, vector, 1, gradVecDot ) ); 

		step = (*gradVecDot) / (*vHv + 1e-6); 

		step = min( step, delta / (*nrmVec + 1e-16) ); 

		alpha = step; 
		cublasCheckError( cublasDscal( cublasHandle, components, 
								&alpha, vector, 1 ) ); 

		return (*vHv) * 0.5 * step * step - (*gradVecDot) * step; 
	}

}
