
#include <core/datadefs.h>
#include <core/structdefs.h>
#include <core/errors.h>

#include <solvers/sampled_trust_cg.h>
#include <solvers/params.h>
#include <solvers/cg_steihaug.h>

#include <device/cuda_utils.h>
#include <device/subsampling_helpers.h>
#include <device/handles.h>
#include <device/device_defines.h>
#include <device/gen_random.h>

#include <functions/eval_gradient.h>
#include <functions/dev_initializations.h>

#include <utilities/sample_matrix.h>
#include <utilities/utils.h>
#include <utilities/alloc_sampled_dataset.h>
#include <utilities/print_utils.h>
#include <utilities/norm_inf.h>

#include <nn/read_nn.h>

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

/*
Implementation of Subsampled Trust Region
using Nonlinear Conjugate Gradient
*/

void readVecFromFileCG( real *dev, real *host ) { 

   int rows = readVector( host, INT_MAX, "./weights2.txt", 0, NULL);
   copy_host_device( host, dev, rows * sizeof(real), cudaMemcpyHostToDevice, 
      ERROR_MEMCPY_HOST_DEVICE );  
   
   fprintf( stderr, "Finished reading Vec (%d) from file \n", rows );  
   for (int i = 0; i < 10; i ++) fprintf( stderr, "%6.10f \n", host[i] );  
}

#ifdef STATS
void outputModelParams( FILE *out, TRUST_REGION_PARAMS *trParams )
{
	fprintf( out, "\n"); 
	fprintf( out, "MaxDelta: %8.6f\n", trParams->maxDelta ); 
	fprintf( out, "eta1: %8.6f\n", trParams->eta1); 
	fprintf( out, "eta2: %8.6f\n", trParams->eta2); 
	fprintf( out, "gamma1: %8.6f\n", trParams->gamma1); 
	fprintf( out, "gamma2: %8.6f\n", trParams->gamma2); 
	fprintf( out, "MaxProps: %d\n", trParams->maxProps); 
	fprintf( out, "MaxMatVecs: %d\n", trParams->maxMatVecs); 
	fprintf( out, "MaxEpochs: %d\n", trParams->maxEpochs); 
	fprintf( out, "MaxCGIterations: %d\n", trParams->maxIters); 
	fprintf( out, "Hessian Sampling: %d\n", int(trParams->hs)); 
	fprintf( out, "Alpha: %8.6f\n", trParams->alpha); 
	fprintf( out, "Regularization: %8.6f\n", trParams->lambda); 
	fprintf( out, "Type of Hessian: %d\n", (int)trParams->hessianType); 
	fprintf( out, "\n"); 
	fprintf( out, "\n"); 
}

void initOutParams( OUT_PARAMS *out )
{
   //Model Part
   out->trainLL = DBL_MAX;
   out->trainModelErr = DBL_MAX;
   out->testLL = DBL_MAX;
   out->testModelErr = DBL_MAX;

   //TR-Part
   out->iteration = -1;
   out->delta = -1;
   out->rho = -1;
   out->normGrad = DBL_MAX;
   out->failCount = INT_MAX;
   out->numMVP = INT_MAX;
   out->numProps = INT_MAX;

   //CG-Part
   out->cgIterations = INT_MAX;
   out->hessVecTime = 0;
   out->normS = DBL_MAX; 

	out->trTime = 0; 
	out->cgTime = 0; 
}

void cleanup (OUT_PARAMS *out ) 
{
	if (out->out) fclose ( out->out ); 
}

void initOutputFile(OUT_PARAMS *out, TRUST_REGION_PARAMS *trParams)
{
   if ( (out->out = fopen("FNN_OUTPUT.txt", "w")) == NULL ) { 
      fprintf( stderr, "Error opening output write file....... !\n" );
      exit( -1 );
   }   

	//Header line
	outputModelParams( out->out, trParams ); 

	fprintf (out->out, "%6s  ", "Iter.No"); 
	fprintf (out->out, "%12s  ", "Tr.Loss"); 
	fprintf (out->out, "%12s  ", "Tr.Mod.Err"); 
	fprintf (out->out, "%12s  ", "Test.Loss"); 
	fprintf (out->out, "%12s  ", "Test.Mod.Err"); 
	fprintf (out->out, "%12s  ", "Delta"); 
	fprintf (out->out, "%12s  ", "rho"); 
	fprintf (out->out, "%12s  ", "Grad.Norm"); 
	fprintf (out->out, "%10s  ", "noProps"); 
	fprintf (out->out, "%6s  ", "noMVP"); 
	fprintf (out->out, "%11s  ", "Tr.Time(ms)"); 
	fprintf (out->out, "%11s  ", "CG.Time(ms)"); 
	fprintf (out->out, "%8s  ", "CG.Iters"); 
	fprintf (out->out, "%12s\n", "norm(s)"); 

	//init the output values here. 
	initOutParams( out ); 
}

void writeOutLine( OUT_PARAMS *out )
{
	fprintf( out->out, "%6d  ", out->iteration ); 
	fprintf( out->out, "%12e  ", out->trainLL); 
	fprintf( out->out, "%12e  ", out->trainModelErr); 
	fprintf( out->out, "%12e  ", out->testLL); 
	fprintf( out->out, "%12e  ", out->testModelErr); 
	fprintf( out->out, "%12e  ", out->delta); 
	fprintf( out->out, "%12e  ", out->rho); 
	fprintf( out->out, "%12e  ", out->normGrad); 
	fprintf( out->out, "%lu  ", out->numProps); 
	fprintf( out->out, "%6d  ", out->numMVP); 
	fprintf( out->out, "%4.3f  ", out->trTime); 
	fprintf( out->out, "%4.3f  ", out->cgTime); 
	fprintf( out->out, "%d(%d)  ", out->cgIterations, out->failCount); 
	fprintf( out->out, "%12e\n", out->normS); 
}

#endif

void subsampledTrustRegionCG( NN_MODEL *nnModel, DEVICE_DATASET *data, 
		TRUST_REGION_PARAMS *params, SCRATCH_AREA *scratch ) {

	//local declarations
	real *hostPtr = scratch->nextHostPtr; 
	real *devPtr = scratch->nextDevPtr; 
	real *pageLckPtr = scratch->nextPageLckPtr;

	//device space
	real *steihaugS0 = devPtr;							//CG Random Initialization
	real *x = steihaugS0 + nnModel->pSize;			// CG initial/final Solution
	real *cgSol = x + nnModel->pSize; 
	real *lWeights = cgSol + nnModel->pSize; 
	real *gradient = lWeights + nnModel->pSize; 				// gradient vector
	real *nextDevPtr = gradient + nnModel->pSize;

	//host space
	real *nextHostPtr = hostPtr;

	//page locked area
	real *trainLogLikelihoodCur = pageLckPtr; 
	real *trainModelErrorCur = trainLogLikelihoodCur + 1;
	real *trainModelError = trainModelErrorCur + 1;
	real *testModelError = trainModelError + 1; 
	real *trainLogLikelihood = testModelError + 1;
	real *testLogLikelihood = trainLogLikelihood + 1; 
	real *nrmS0= testLogLikelihood + 1; 
	real *xdot = nrmS0 + 1;
	real *nextPageLckPtr = xdot + 1;

	//local automatics here. 
	real alpha = 1, beta = 0; 
#ifdef STATS
	real start, total; 
	real cg_start, cg_total; 
	OUT_PARAMS outParams; 
#endif

#ifdef DEBUG_TRUST
real temp;
#endif
	
	int failCount = 0; 
	int n = data->trainSizeX; 
	int sampleSize = params->hs;

	//solver params here. 
	unsigned long int numProps; 
	int numMatVecs;
	CG_PARAMS cg_steihaug_params;
	STEIHAUG_PARAMS steihaugParams; 
	real rho; 

	//begin coding
	steihaugParams.maxIterations = 250; 
	steihaugParams.alpha = 0; 
	steihaugParams.tolerance = 1e-9;

	//allocate the sampled datasets here... 
	//also compute the offsets for temp. storage
   allocSampledDataset( data, sampleSize ); 
   initSampledROffsets( nnModel, sampleSize ); 
   initSampledZOffsets( nnModel, sampleSize ); 

	//init the scratch area with correct pointers here. 
	scratch->nextDevPtr = nextDevPtr; 
	scratch->nextHostPtr = nextHostPtr; 
	scratch->nextPageLckPtr = nextPageLckPtr; 

	numProps = 1; 
	numMatVecs = 1; 
	rho = 0; 

#ifdef STATS
initOutputFile( &outParams, params ); 
#endif

	for (int iter = 0; iter < params->maxEpochs; iter ++) { 

		if (numProps >= params->maxProps || numMatVecs >= params->maxMatVecs) {
			iter --; 
			break;
		}

//STATSs... 
#ifdef STATS
		outParams.iteration = iter;
		start = Get_Time (); 
#endif

#ifdef DEBUG_TRUST
fprintf( stderr, "TrustRegion: ***Epoch: %d ..... Begin\n", iter); 
cublasCheckError( cublasDnrm2( cublasHandle, nnModel->pSize, data->weights, 1, &temp ) );
fprintf( stderr, "TrustRegion: WEIGHTS: %6.10f\n", temp); 

if (temp > 1253 && temp < 1254)
{
	writeVector( data->weights, nnModel->pSize, "pgpu.txt", 0, nextHostPtr ); 
}
#endif

		//compute Gradient
		//ll, gradient and error
		//g @ weights
		// Full Dataset evaluation here. 
		computeGradient( nnModel, data, scratch, data->weights,
			NULL, NULL, gradient, trainLogLikelihood, trainModelError, FULL_DATASET ); 

#ifdef DEBUG_TRUST
cublasCheckError( cublasDnrm2( cublasHandle, nnModel->pSize, gradient, 1, &temp ) );
fprintf( stderr, "TrustRegion: ll: %6.10f, mErr: %6.10f, norm(gradient): %6.10f \n", *trainLogLikelihood, *trainModelError, temp); 
#endif

		//statistics here. 
		numProps += n; 

		//evaluate the model here. on the Test Dataset here. 
		//model @ weights
		evaluateModel( nnModel, data, scratch, data->weights, 
			testLogLikelihood, testModelError, FULL_DATASET, TEST_DATA ); 	
#ifdef DEBUG_TRUST
fprintf( stderr, "TrustRegion: Test LL: %6.10f, Test MErr: %6.10f \n", *testLogLikelihood, testModelError);
#endif

		//update the loss with regularization term here. 
		cublasCheckError ( cublasDnrm2( cublasHandle, nnModel->pSize,
					data->weights, 1, nextPageLckPtr));
		*trainLogLikelihood += 0.5 * params->lambda * (*nextPageLckPtr); 
		
		//udpate the gradient with regularization term;
		alpha = params->lambda;
		cublasCheckError( cublasDaxpy( cublasHandle, 
									nnModel->pSize, &alpha, 
									data->weights, 1, 
									gradient, 1 ) ); 	

#ifdef STATS
outParams.trainLL = *trainLogLikelihood; 
outParams.trainModelErr = *trainModelError; 
outParams.testLL = *testLogLikelihood; 
outParams.testModelErr = *testModelError; 
outParams.delta = params->delta;
#endif

							
		//sanity check on the gradient. 
		// norm( gradient, Inf ) <= 1e-10

		norm_inf( gradient, nextHostPtr, nnModel->pSize, nextPageLckPtr, nextPageLckPtr+1, nextDevPtr); 
#ifdef DEBUG_TRUST
		fprintf( stderr, "grad(inf) == %6.10f, %f \n", *nextPageLckPtr, *(nextPageLckPtr+1) ); 
#endif

		if ( fabs( *nextPageLckPtr ) <= 1e-10 ){ 
			fprintf( stderr, "SubSampledTrustRegion: Infinity norm is below the "
							"allowed limit: %e, index: %d \n", *nextPageLckPtr, (int)*(nextPageLckPtr+1) ); 
			break;
		} 


		//Trust Region subproblem here. 
		//sudhir
		//cublasCheckError( cublasDcopy( cublasHandle, nnModel->pSize, data->weights, 1, lWeights, 1) ); 
		copy_device( lWeights, data->weights, nnModel->pSize * sizeof(real), ERROR_MEMCPY_DEVICE_DEVICE ); 

		failCount = 0; 
		while (1) {
			
			if (failCount == 0) {
	
				//init... initial solution the non-linear solve. -- trust region
				getRandomVector( nnModel->pSize, NULL, steihaugS0, RAND_NORMAL ); 

#ifdef DEBUG_TRUST
readVecFromFileCG( steihaugS0, nextHostPtr ); 
#endif

				//normalize the s0 vector here. 
				cublasCheckError( 
					cublasDnrm2( cublasHandle, nnModel->pSize, steihaugS0, 1, nrmS0 )
				);

				alpha = (0.99 * params->delta) / *nrmS0; 
				cublasCheckError( 
					cublasDscal( cublasHandle, nnModel->pSize, &alpha, steihaugS0, 1 )
				);

			} // end of the failCount if statement.

			//params, hessianZ, hessianDX
			copy_device( x, steihaugS0, sizeof(real) * nnModel->pSize, 
				ERROR_MEMCPY_DEVICE_DEVICE ); 

#ifdef DEBUG_TRUST
temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, nnModel->pSize, x, 1, &temp) );
fprintf( stderr, "TrustRegion: Norm(s) ... %6.10f \n", temp);			
#endif

			cg_steihaug_params.x = x ; //initial solution
			cg_steihaug_params.b = gradient;  //gradient
			cg_steihaug_params.delta = params->delta;; 

			cg_steihaug_params.errTol = steihaugParams.tolerance;
			cg_steihaug_params.maxIt = steihaugParams.maxIterations;
			cg_steihaug_params.cgIterConv = 0; 
			cg_steihaug_params.flag = CG_STEIHAUG_WE_DONT_KNOW;

			cg_steihaug_params.m = 0;
			cg_steihaug_params.hessianType = params->hessianType;
			
			// solve the Trust Region problem here 
			// using Conjugate Gradient Nonlinear Extensions.... 

			//sample X and Y
			data->sampleSize = sampleSize;
			sampleColumnMatrix( data, scratch, 1 );

#ifdef STATS
cg_start = Get_Time (); 
#endif

			ConjugateGradientNonLinear( nnModel, data, scratch, &cg_steihaug_params, lWeights ); 

#ifdef STATS
cg_total = Get_Timing_Info ( cg_start ); 
#endif

#ifdef DEBUG_TRUST
cublasCheckError( cublasDnrm2( cublasHandle, nnModel->pSize, x, 1, &temp) );
fprintf( stderr, "TrustRegion: Iteration --> %d, NonLinear CG Done ... m = %e, norm: %6.10f\n", iter, cg_steihaug_params.m, temp ); 
#endif

			//init the scratch area with correct pointers here. 
			scratch->nextDevPtr = nextDevPtr; 
			scratch->nextHostPtr = nextHostPtr; 
			scratch->nextPageLckPtr = nextPageLckPtr; 

			copy_device( cgSol, x, nnModel->pSize * sizeof(real), 
				ERROR_MEMCPY_DEVICE_DEVICE );  

			numProps += cg_steihaug_params.cgIterConv * 2 * sampleSize; 
			numMatVecs += cg_steihaug_params.cgIterConv; 

			//evaluate the model with the increment from the CG. 
			// evaluate the model @ (data->weights + steihaugS0)
			alpha = 1; 
			cublasCheckError( cublasDaxpy( cublasHandle, nnModel->pSize, &alpha, 
											lWeights, 1, cgSol, 1 ) ); 
			evaluateModel( nnModel, data, scratch, cgSol, 
				trainLogLikelihoodCur, trainModelErrorCur, FULL_DATASET, TRAIN_DATA  ); 	
	
			numProps += data->trainSizeX; 

			//update the loss with regularization term here. 
			cublasCheckError ( cublasDdot( cublasHandle, nnModel->pSize, cgSol, 1, cgSol, 1, xdot));
			(*trainLogLikelihoodCur) += 0.5 * params->lambda * (*xdot); 
#ifdef DEBUG_TRUST
fprintf( stderr, "subsampledTrustRegion: TrLL(new): %6.10f TrErr(new): %6.10f \n", 
						*trainLogLikelihoodCur, *trainModelErrorCur ); 
#endif

			//update the TR parameter here. 
			rho = (*trainLogLikelihood - *trainLogLikelihoodCur) / (-cg_steihaug_params.m); 

			cublasCheckError( cublasDnrm2( cublasHandle, nnModel->pSize, x, 1, nrmS0 ) );

			//Whether to accept the current Increment or NOT
			if ( (cg_steihaug_params.m >= 0) || (rho < params->eta2) ) {
				//unacceptable increment
				failCount ++; 
#ifdef DEBUG_TRUST
				fprintf( stderr, "subsampledTrustRegion: Failure #: %d Delta: %e, rho: %e, iters: %d \n", 
						failCount, params->delta, rho, cg_steihaug_params.cgIterConv ); 
#endif
				//update delta. 
				params->delta /= params->gamma1; 

				//update init Solution Vector; 
				// S0 = (delta / norm(s)) * s
				//SUDHIR -- TODO
				copy_device( steihaugS0, x, sizeof(real) * nnModel->pSize, ERROR_MEMCPY_DEVICE_DEVICE ); 

				alpha = params->delta / *nrmS0;
				cublasCheckError (
					cublasDscal( cublasHandle, nnModel->pSize, &alpha, steihaugS0, 1 )
				);
			} else if ( rho < params->eta1 ) {
				//report Success. 
#ifdef DEBUG_TRUST
				fprintf(stderr, "subsampledTrustRegion: Success !!!  Epoch: %d, 	 "
						" delta: %e, rho: %e, nrm(s): %e, iters: %d, MVs: %d, props: %lu \n", 
						iter, params->delta, rho, *nrmS0, cg_steihaug_params.cgIterConv, numMatVecs, numProps ); 
#endif

				alpha = 1;
				cublasCheckError (
					cublasDaxpy( cublasHandle, nnModel->pSize, &alpha, x, 1, data->weights, 1 ) ); 

				//min here
				if (params->maxDelta <= ( params->gamma2 * params->delta ) )
					params->delta = params->maxDelta; 
				else 
					params->delta = params->gamma2 * params->delta;

				break;
			} else {
#ifdef DEBUG_TRUST
				fprintf( stderr, "subsampledTrustRegion: Super Success !!!! , Epoch: %d, "
						" delta: %e, rho: %e, nrm(s): %e, iters: %d, MVs: %d, props: %lu \n", 
						iter, params->delta, rho, *nrmS0, cg_steihaug_params.cgIterConv, numMatVecs, numProps ); 
#endif

				alpha = 1;
				cublasCheckError (
					cublasDaxpy( cublasHandle, nnModel->pSize, &alpha, x, 1, data->weights, 1 ) ); 

				if (params->maxDelta <= ( params->gamma1 * params->delta ) )
					params->delta = params->maxDelta; 
				else 
					params->delta = params->gamma1 * params->delta;

				break;
			} 
//print the solution norm here. 
#ifdef DEBUG_TRUST
cublasCheckError( cublasDnrm2( cublasHandle, nnModel->pSize, data->weights, 1, &temp) );
fprintf( stderr, "Norm of the weights: %6.10f \n", temp); 
#endif

		} // End of the trust region sub-problem.
		
#ifdef STATS
		total = Get_Timing_Info ( start ); 
		//fprintf (stderr, "subsampledTrustRegion: Epoch: %d ------------- Time(secs): %4.3f\n", iter, total ); 

		//journal the rest of the parameters here and output the line here. 
		outParams.delta = params->delta; 
		outParams.rho = rho; 
		outParams.numMVP = numMatVecs; 
		outParams.numProps = numProps; 
		outParams.failCount = failCount; 
	
		cublasCheckError( cublasDnrm2( cublasHandle, nnModel->pSize, gradient, 1, nextPageLckPtr ) );
		outParams.normGrad = *nextPageLckPtr; 

		outParams.trTime = total * 1000; 
		outParams.cgTime = cg_total * 1000; 

		outParams.cgIterations = cg_steihaug_params.cgIterConv; 
		
		outParams.normS = *nrmS0;  //norm of the solution from CG.

		writeOutLine( &outParams );

		fprintf( stderr, "." ); 
#endif

	} // End of the main for-loop... 
	fprintf( stderr, "\n");

#ifdef STATS
	cleanup ( &outParams ); 
#endif
}
