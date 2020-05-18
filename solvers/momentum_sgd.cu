
#include <solvers/momentum_sgd.h>

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

#include <utilities/sample_matrix.h>
#include <utilities/utils.h>
#include <utilities/alloc_sampled_dataset.h>
#include <utilities/print_utils.h>

#include <nn/read_nn.h>

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#ifdef STATS
void SGDOutputModelParams( FILE *out, MOMENTUM_PARAMS *mParams )
{
   fprintf( out, "\n"); 
   fprintf( out, "alpha: %8.6f\n", mParams->alpha); 
   fprintf( out, "beta: %8.6f\n", mParams->beta); 
   fprintf( out, "MaxProps: %d\n", mParams->maxProps); 
   fprintf( out, "MaxEpochs: %d\n", mParams->maxEpochs); 
   fprintf( out, "Regularization: %8.6f\n", mParams->lambda); 
   fprintf( out, "\n"); 
   fprintf( out, "\n");
}

void SGDCleanup (SGD_OUT_PARAMS *out )
{  
   if (out->out) fclose ( out->out );
}

void SGDInitOutParams( SGD_OUT_PARAMS *sgdOut)
{
		sgdOut->iteration = 0;
		sgdOut->trainLL = 0; 
		sgdOut->trainModelErr = 0;  
		sgdOut->testLL = 0; 
		sgdOut->testModelErr = 0; 
		sgdOut->normGrad = 0;
		sgdOut->numProps = 0; 
		sgdOut->iter_time = 0; 
		sgdOut->total_time = 0; 
	
}

void SGDInitOutputFile(SGD_OUT_PARAMS *out, MOMENTUM_PARAMS *mParams)
{
   if ( (out->out = fopen("SGD_OUTPUT.txt", "w")) == NULL ) {
      fprintf( stderr, "Error opening output write file....... !\n" );
      exit( -1 );
   }

   //Header line
   SGDOutputModelParams( out->out, mParams );

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
   SGDInitOutParams( out );
}

void SGDWriteOutLine( SGD_OUT_PARAMS *out )
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


void momentum_sgd (NN_MODEL *model, DEVICE_DATASET *data, 
					SCRATCH_AREA *scratch, MOMENTUM_PARAMS *params)
{
	//locals 
   int n = data->trainSizeX;
   int sampleSize = params->sampleSize;
	int datasetLoops = n / sampleSize + 1; 
	real iter_start, iter_end; 
	real iter_running; 
	SGD_OUT_PARAMS sgdOut; 
	real normGrad;
	real alpha; 
	unsigned long int numProps; 

	//pts
	real *devPtr = scratch->nextDevPtr; 
	real *hostPtr = scratch->nextHostPtr; 
	real *pageLckPtr = scratch->nextPageLckPtr; 

	//device space
	real *gradient = devPtr; 
	real *momentum_wts = gradient + model->pSize;
	real *nextDevPtr = momentum_wts + model->pSize; 

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
	SGDInitOutputFile( &sgdOut, params ); 	
#endif
	cuda_memset( momentum_wts, 0, sizeof(real) * model->pSize, ERROR_MEMSET ); 

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

	      //udpate the gradient with regularization term;
  		   alpha = params->lambda;
  		   cublasCheckError( cublasDaxpy( cublasHandle, model->pSize, &alpha, data->weights, 1, gradient, 1 ) );

      	//statistics here. 
      	numProps += params->sampleSize;

			//update step at the end
			// v = mu * v - learning_rate * gradient
			alpha = params->beta; 
			cublasCheckError( cublasDscal( cublasHandle, model->pSize, &alpha, momentum_wts, 1 ));
			alpha = -(params->alpha); 
			cublasCheckError( cublasDaxpy( cublasHandle, model->pSize, &alpha, gradient, 1, momentum_wts, 1) ); 

			//update weights
			//x += v; 
			alpha = 1; 
			cublasCheckError( cublasDaxpy( cublasHandle, model->pSize, &alpha, momentum_wts, 1, data->weights, 1 ));
		}

#ifdef STATS
		//end
		iter_end = Get_Timing_Info( iter_start ); 
		iter_running += iter_end; 

		//gradient
      computeGradient( model, data, scratch, data->weights,
         	NULL, NULL, gradient, trainLogLikelihood, trainModelError, FULL_DATASET );

		//normGradient
		cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, gradient, 1, &normGrad ) ); 

      //update the loss with regularization term here. 
      cublasCheckError ( cublasDnrm2( cublasHandle, model->pSize,
               data->weights, 1, nextPageLckPtr));
      *trainLogLikelihood += 0.5 * params->lambda * (*nextPageLckPtr);

      //evaluate the model here. on the Test Dataset here. 
      //model @ weights
      evaluateModel( model, data, scratch, data->weights,
         testLogLikelihood, testModelError, FULL_DATASET, TEST_DATA );

		//stats here. 
		sgdOut.iteration = iter;
		sgdOut.trainLL = *trainLogLikelihood; 
		sgdOut.trainModelErr = *trainModelError; 
		sgdOut.testLL = *testLogLikelihood; 
		sgdOut.testModelErr = *testModelError; 
		sgdOut.normGrad = normGrad;
		sgdOut.numProps = numProps; 
		sgdOut.iter_time = iter_end; 
		sgdOut.total_time = iter_running; 

		SGDWriteOutLine (&sgdOut); 
#endif
	}
}
