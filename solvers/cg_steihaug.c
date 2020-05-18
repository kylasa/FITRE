
#include <solvers/cg_steihaug.h>
#include <solvers/params.h>

#include <core/datadefs.h>
#include <core/structdefs.h>
#include <core/errors.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>

#include <functions/eval_hessian_vec.h>
#include <functions/eval_gauss_newton_hessian_vec.h>
#include <functions/eval_gradient.h>

#include <utilities/utils.h>

#include <nn/nn_decl.h>

#include <stdio.h>
#include <stdlib.h>


/*
Solve the trust region problem here. 
*/

void ConjugateGradientNonLinear( NN_MODEL *model, 
		DEVICE_DATASET *data, SCRATCH_AREA *scratch, 
		CG_PARAMS *params, real *weights )		
{
	//scratch area here. 
	real *devPtr = scratch->nextDevPtr; 
	real *hostPtr = scratch->nextHostPtr; 
	real *pageLckPtr = scratch->nextPageLckPtr; 

	//locals
	real cg_alpha = 1, beta, alpha;
	real ac, bc, cc;
	real rho; 
	real rhoOld;
	real tst; 
	real terminate; 
	real hatdel;
	real xnorm;
	real xb; 
	
	int iter; 

	//initializations here. 
	int vecSize = model->pSize;
	real *g = devPtr;
	real *r = g + vecSize;
	real *z = r + vecSize;
	real *w = z + vecSize; 
	real *p = w + vecSize;
	real *hessianZ = p + vecSize;
	real *hessianDX = hessianZ + 2 * model->sampledZSize; 
	real *nextDevPtr = hessianDX + 2 * model->sampledRSize;

	//code here. 
	scratch->nextDevPtr = nextDevPtr; 
	scratch->nextHostPtr = hostPtr; 
	scratch->nextPageLckPtr = pageLckPtr; 


#ifdef DEBUG_CG
real temp = 0; 
#endif

	/*
		Compute the Z and Dx evaluated at weights and use it repeatedly..
	*/
	//cuda_memset( hessianZ, 0, sizeof (real) * model->zSize, ERROR_MEMSET ); 
	//SUDHIR-2
	computeGradient( model, data, scratch, weights, 
				hessianZ, hessianDX, g, pageLckPtr, pageLckPtr + 1, SAMPLED_DATASET ); 
	scratch->nextDevPtr = nextDevPtr; 
	scratch->nextHostPtr = hostPtr; 
	scratch->nextPageLckPtr = pageLckPtr; 
	/*
		Done evaluation of hessianZ and hessianDx
	*/

	//g = b
	cublasCheckError( cublasDcopy( cublasHandle, vecSize, params->b, 1, g, 1) ); 

	//r = -g - A(x)
	cublasCheckError( cublasDcopy( cublasHandle, vecSize, g, 1, r, 1) ); 
	alpha = -1;
	cublasCheckError( cublasDscal( cublasHandle, vecSize, &alpha, r, 1) ); 

	//(*computeHessianVec)( data, params->x, Ax );
	if (params->hessianType == TRUE_HESSIAN)
		hessianVec( model, data, hessianZ, hessianDX, params->x, weights, scratch, SAMPLED_DATASET );
	else 
		gaussNewtonHessianVec( model, data, hessianZ, hessianDX, params->x, weights, scratch, SAMPLED_DATASET );

	alpha = -1; 
	cublasCheckError( cublasDaxpy( cublasHandle, vecSize, &alpha, scratch->nextDevPtr, 1, r, 1) ); 

#ifdef DEBUG_CG
cublasCheckError( cublasDnrm2( cublasHandle, vecSize, scratch->nextDevPtr, 1, &temp )); 
fprintf( stderr, "Norm of A(x): %6.10f \n", temp ); 
#endif

	//z = r
	cublasCheckError( cublasDcopy( cublasHandle, vecSize, r, 1, z, 1) );

	//rho = z'r
	cublasCheckError( cublasDdot( cublasHandle, vecSize, z, 1, r, 1, &rho) ); 

	//tst = norm(r, 2)
	cublasCheckError( cublasDnrm2( cublasHandle, vecSize, r, 1, &tst) ); 
	
	terminate = params->errTol * tst; 
	iter = 1; 
	hatdel = params->delta * 1;
	rhoOld = 1.0;

#ifdef DEBUG_CG
	fprintf( stderr, "                  CG-Steihaug Output              \n\n");
	fprintf( stderr, "--------------------------------------------------\n"); 
#endif

	
	//begin here. 
	params->flag = CG_STEIHAUG_WE_DONT_KNOW;
	if (tst <= terminate) {
		params->flag = CG_STEIHAUG_SMALL_NORM_G;
	}

	cublasCheckError( cublasDnrm2( cublasHandle, vecSize, params->x, 1, &xnorm) ); 
	while( (tst > terminate) && (iter <= ((int)params->maxIt)) && (xnorm <= hatdel) ){

		if ( iter == 1) {
			//p = z
			cublasCheckError( cublasDcopy( cublasHandle, vecSize, z, 1, p, 1) ); 
		} else {
			beta = rho / rhoOld;	

			//p = z + beta * p;
			alpha = 1; 
			cublasCheckError( cublasDscal( cublasHandle, vecSize, &beta, p, 1) ); 
			cublasCheckError( cublasDaxpy( cublasHandle, vecSize, &alpha, z, 1, p, 1) ); 
		}

		//w = A(p)
		if (params->hessianType == TRUE_HESSIAN)
			hessianVec( model, data, hessianZ, hessianDX, p, weights, scratch, SAMPLED_DATASET );
		else
			gaussNewtonHessianVec( model, data, hessianZ, hessianDX, p, weights, scratch, SAMPLED_DATASET );
		cublasCheckError( cublasDcopy( cublasHandle, vecSize, scratch->nextDevPtr, 1, w, 1 )); 
		cublasCheckError( cublasDdot( cublasHandle, vecSize, w, 1, p, 1, &cg_alpha) ); 


#ifdef DEBUG_CG
	cublasCheckError( cublasDnrm2( cublasHandle, vecSize, p, 1, &temp) ); 
fprintf( stderr, "Norm P: %6.10f \n", temp ); 
	cublasCheckError( cublasDnrm2( cublasHandle, vecSize, scratch->nextDevPtr, 1, &temp) ); 
fprintf( stderr, "Alpha : %6.10f, hessvec: %6.10f \n", cg_alpha, temp ); 
#endif
		
		if (cg_alpha <= 0) {
			cublasCheckError( cublasDdot( cublasHandle, vecSize, p, 1, p, 1, &ac )); 

			cublasCheckError( cublasDdot( cublasHandle, vecSize, params->x, 1, p, 1, &bc) ); 
			bc *= 2.0;
			
			cublasCheckError( cublasDdot( cublasHandle, vecSize, params->x, 1, params->x, 1, &cc) ); 
			cc -= params->delta * params->delta; 

			cg_alpha = (-bc + sqrt( bc * bc - 4.0 * ac * cc ) ) / (2.0 * ac); 
			params->flag = CG_STEIHAUG_NEGATIVE_CURVATURE; 

			cublasCheckError( cublasDaxpy( cublasHandle, vecSize, &cg_alpha, p, 1, params->x, 1) ); 
			break;
			
		} else {

			cg_alpha = rho / cg_alpha; 

			cublasCheckError( cublasDcopy( cublasHandle, vecSize, params->x, 1, nextDevPtr, 1) ); 
			cublasCheckError( cublasDaxpy( cublasHandle, vecSize, &cg_alpha, p, 1, nextDevPtr, 1) ); 
			cublasCheckError( cublasDnrm2( cublasHandle, vecSize, nextDevPtr, 1, &xnorm) ); 

			if (xnorm > params->delta) {
				cublasCheckError( cublasDdot( cublasHandle, vecSize, p, 1, p, 1, &ac )); 

				cublasCheckError( cublasDdot( cublasHandle, vecSize, params->x, 1, p, 1, &bc) ); 
				bc *= 2.0;
			
				cublasCheckError( cublasDdot( cublasHandle, vecSize, params->x, 1, params->x, 1, &cc) ); 
				cc -= params->delta * params->delta; 

				cg_alpha = (-bc + sqrt( bc * bc - 4.0 * ac * cc ) ) / (2.0 * ac); 

				params->flag = CG_STEIHAUG_HIT_BOUNDRY; 

				cublasCheckError( cublasDaxpy( cublasHandle, vecSize, &cg_alpha, p, 1, params->x, 1) ); 
				break;
			}
		} //end of the if-else part here. 	

		alpha = cg_alpha;
		cublasCheckError( cublasDaxpy( cublasHandle, vecSize, &alpha, p, 1, params->x, 1) ); 
		alpha *= -1.0; 	
		cublasCheckError( cublasDaxpy( cublasHandle, vecSize, &alpha, w, 1, r, 1) ); 

		//tst = norm(r)
		cublasCheckError( cublasDnrm2( cublasHandle, vecSize, r, 1, &tst) ); 

		if (tst <= terminate) {
			params->flag = CG_STEIHAUG_RS_CASE; 
			break;
		}

		cublasCheckError( cublasDnrm2( cublasHandle, vecSize, params->x, 1, &xnorm) ); 
		if (xnorm >= hatdel) {
			params->flag = CG_STEIHAUG_CLOSE_BOUNDRY;
			break;
		}

		//debug print here
#ifdef DEBUG_CG
		fprintf( stderr, "%3d\t\t%14.8e\t\t%d\n", iter, tst, params->flag ); 
#endif

		rhoOld = rho; 
		cublasCheckError( cublasDcopy( cublasHandle, vecSize, r, 1, z, 1) ); 

		//rho = z'r
		cublasCheckError( cublasDdot( cublasHandle, vecSize, z, 1, r, 1, &rho) ); 

		iter ++; 
		
	} //end of while loop

	if ( iter > params->maxIt) params->flag = CG_STEIHAUG_MAX_IT; 

	params->cgIterConv = iter;

	//evaluate TR Model @ params->x
	//trust_region_model( model, data, scratch, 
	//		hessianZ, hessianDX, params->x, params->b, &params->m );

	hessianVec( model, data, hessianZ, hessianDX, params->x, weights, scratch, SAMPLED_DATASET );
	cublasCheckError( cublasDdot( cublasHandle, vecSize, 
							scratch->nextDevPtr, 1, 
							params->x, 1, &params->m ) ); 	
	params->m *= 0.5;
//fprintf( stderr, " A(x) * x * 0.5 == %6.10f \n", params->m ); 

	cublasCheckError( cublasDdot( cublasHandle, vecSize, 
							params->x, 1, params->b, 1, 
							&xb ) ); 	
	params->m += xb; 

//fprintf( stderr, " A(x) * x * 0.5  + xb == %6.10f \n", params->m ); 
}
