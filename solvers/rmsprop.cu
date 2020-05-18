
#include <solvers/rmsprop.h>

#include <core/datadefs.h>
#include <core/structdefs.h>
#include <core/errors.h>

#include <device/cuda_utils.h>
#include <device/subsampling_helpers.h>
#include <device/handles.h>
#include <device/device_defines.h>
#include <device/gen_random.h>

#include <functions/eval_gradient.h>
#include <functions/dev_initializations.h>
#include <functions/dev_elem_sqr.h>
#include <functions/dev_elem_scale.h>
#include <functions/dev_elem_sqr_decay.h>

#include <utilities/sample_matrix.h>
#include <utilities/utils.h>
#include <utilities/alloc_sampled_dataset.h>
#include <utilities/print_utils.h>

#include <nn/read_nn.h>

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#ifdef STATS
void RMSPropOutputModelParams( FILE *out, RMSPROP_PARAMS *mParams )
{
   fprintf( out, "\n"); 
   fprintf( out, "********* RMSProp REPORT ********* \n"); 
   fprintf( out, "\n"); 
   fprintf( out, "learning_rate: %8.6f\n", mParams->step); 
   fprintf( out, "DecayRate: %8.6f\n", mParams->decayRate ); 
   fprintf( out, "eps: %8.6f\n", mParams->eps); 
   fprintf( out, "MaxProps: %d\n", mParams->maxProps); 
   fprintf( out, "MaxEpochs: %d\n", mParams->maxEpochs); 
   fprintf( out, "Regularization: %8.6f\n", mParams->lambda); 
   fprintf( out, "\n"); 
   fprintf( out, "\n");
}

void RMSPROPCleanup (RMSPROP_OUT_PARAMS *out )
{  
   if (out->out) fclose ( out->out );
}

void RMSPROPInitOutParams( RMSPROP_OUT_PARAMS *rmsPropOut)
{
		rmsPropOut->iteration = 0;
		rmsPropOut->trainLL = 0; 
		rmsPropOut->trainModelErr = 0;  
		rmsPropOut->testLL = 0; 
		rmsPropOut->testModelErr = 0; 
		rmsPropOut->normGrad = 0;
		rmsPropOut->numProps = 0; 
		rmsPropOut->iter_time = 0; 
		rmsPropOut->total_time = 0; 
}

void RMSPropInitOutputFile(RMSPROP_OUT_PARAMS *out, RMSPROP_PARAMS *mParams)
{
   if ( (out->out = fopen("RMSPROP_OUTPUT.txt", "w")) == NULL ) {
      fprintf( stderr, "Error opening output write file....... !\n" );
      exit( -1 );
   }

   //Header line
   RMSPropOutputModelParams( out->out, mParams );

   fprintf (out->out, "%6s  ", "Iter.No");
   fprintf (out->out, "%12s  ", "Tr.Loss");
   fprintf (out->out, "%12s  ", "Tr.Mod.Err");
   fprintf (out->out, "%12s  ", "Test.Loss");
   fprintf (out->out, "%12s  ", "Test.Mod.Err");
   fprintf (out->out, "%12s  ", "Grad.Norm");
   fprintf (out->out, "%10s  ", "noProps");
   fprintf (out->out, "%11s  ", "Tr.Time(ms)");
   fprintf (out->out, "%14s  ", "Total.Time(ms)");

   //init the output values here. 
   RMSPROPInitOutParams( out );
}

void RMSPROPWriteOutLine( RMSPROP_OUT_PARAMS *out )
{
   fprintf( out->out, "%6d  ", out->iteration );
   fprintf( out->out, "%12e  ", out->trainLL);
   fprintf( out->out, "%12e  ", out->trainModelErr);
   fprintf( out->out, "%12e  ", out->testLL);
   fprintf( out->out, "%12e  ", out->testModelErr);
   fprintf( out->out, "%12e  ", out->normGrad);
   fprintf( out->out, "%lu  ", out->numProps);
   fprintf( out->out, "%4.3f  ", out->iter_time);
   fprintf( out->out, "%4.3f  ", out->total_time);
	fprintf( out->out, "\n" ); 
}

#endif


void rmsprop (NN_MODEL *model, DEVICE_DATASET *data, 
					SCRATCH_AREA *scratch, RMSPROP_PARAMS *params)
{
	//locals 
   int n = data->trainSizeX;
   int sampleSize = params->sampleSize;
	int datasetLoops = n / sampleSize + 1; 
	real iter_start, iter_end; 
	real iter_running; 
	RMSPROP_OUT_PARAMS rmsPropOut; 
	real normGrad;
	real alpha; 
	unsigned long int numProps = 0; 

	//pts
	real *devPtr = scratch->nextDevPtr; 
	real *hostPtr = scratch->nextHostPtr; 
	real *pageLckPtr = scratch->nextPageLckPtr; 

	//device space
	real *gradient = devPtr; 
	real *cache = gradient + model->pSize;
	real *nextDevPtr = cache + model->pSize; 

   //page locked area
   real *trainLogLikelihoodCur = pageLckPtr; 
   real *trainModelErrorCur = trainLogLikelihoodCur + 1;
   real *trainModelError = trainModelErrorCur + 1;
   real *testModelError = trainModelError + 1;  
   real *trainLogLikelihood = testModelError + 1;
   real *testLogLikelihood = trainLogLikelihood + 1;  
	real *nextPageLckPtr = testLogLikelihood + 1; 

	//sampling space here
   allocSampledDataset( data, sampleSize );
   initSampledROffsets( model, sampleSize );
   initSampledZOffsets( model, sampleSize );

	//initializations here
	iter_start = iter_end = iter_running = 0; 
#ifdef STATS
	RMSPropInitOutputFile( &rmsPropOut, params ); 	
#endif
	cuda_memset( cache, 0, sizeof(real) * model->pSize, ERROR_MEMSET ); 

	//begin main iterations here. 
	for (int iter = 0; iter < params->maxEpochs; iter ++){

		//begin
#ifdef STATS
		iter_start = Get_Time (); 
#endif

		for (int j = 0; j < datasetLoops; j ++) {

			//sample dataset
			data->sampleSize = params->sampleSize;
			sampleColumnMatrix(data, scratch, 0);

			//gradient
      	computeGradient( model, data, scratch, data->weights,
        		 NULL, NULL, gradient, trainLogLikelihood, trainModelError, SAMPLED_DATASET );

      	//statistics here. 
      	numProps += params->sampleSize;

      	//udpate the gradient with regularization term;
      	alpha = params->lambda;
      	cublasCheckError( cublasDaxpy( cublasHandle, model->pSize, &alpha, data->weights, 1, gradient, 1 ) );

			// update the step RMSProp algorithm
			// cache = decay_rate * cache + (1 - decay) * grad ** 2
			// x += -step * grad / (sqrt( cache ) + eps)

			// cache = grad ** 2
			int numBlocks = model->pSize / BLOCK_SIZE + 
						(( model->pSize % BLOCK_SIZE == 0 ) ? 0 : 1 ); 
			kerElemSqrDecay <<< numBlocks, BLOCK_SIZE >>> 
					(cache, gradient, model->pSize, params->decayRate, cache ); 
			cudaThreadSynchronize ();
			cudaCheckError ();
		
			// cache = gradient / (sqrt( cache ) + eps)
			kerElemInvScale <<< numBlocks, BLOCK_SIZE >>> 
					(cache, gradient, params->eps, model->pSize, cache ); 
			cudaThreadSynchronize ();
			cudaCheckError ();

			//update weights
			// x += - step * cache
			alpha = -params->step; 
			cublasCheckError( cublasDaxpy( cublasHandle, model->pSize, &alpha, cache, 1, data->weights, 1 ));
		}

#ifdef STATS
		//end
		iter_end = Get_Timing_Info( iter_start ); 
		iter_running += iter_end; 

		//gradient
      computeGradient( model, data, scratch, data->weights,
        	 NULL, NULL, gradient, trainLogLikelihood, trainModelError, FULL_DATASET );

      //udpate the gradient with regularization term;
      alpha = params->lambda;
      cublasCheckError( cublasDaxpy( cublasHandle, model->pSize, &alpha, data->weights, 1, gradient, 1 ) );

      //update the loss with regularization term here. 
      cublasCheckError ( cublasDnrm2( cublasHandle, model->pSize,
               data->weights, 1, nextPageLckPtr));
      *trainLogLikelihood += 0.5 * params->lambda * (*nextPageLckPtr);

		//normGradient
		cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, gradient, 1, &normGrad ) ); 

      //evaluate the model here. on the Test Dataset here. 
      //model @ weights
      evaluateModel( model, data, scratch, data->weights,
         testLogLikelihood, testModelError, FULL_DATASET, TEST_DATA );

		//stats here. 
		rmsPropOut.iteration = iter;
		rmsPropOut.trainLL = *trainLogLikelihood; 
		rmsPropOut.trainModelErr = *trainModelError; 
		rmsPropOut.testLL = *testLogLikelihood; 
		rmsPropOut.testModelErr = *testModelError; 
		rmsPropOut.normGrad = normGrad;
		rmsPropOut.numProps = numProps; 
		rmsPropOut.iter_time = iter_end; 
		rmsPropOut.total_time = iter_running; 

		RMSPROPWriteOutLine (&rmsPropOut); 
#endif
	}
}
