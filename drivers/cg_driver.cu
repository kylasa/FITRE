
#include <drivers/cg_driver.h>

#include <solvers/cg_steihaug.h>
#include <solvers/params.h>

#include <core/errors.h>

#include <device/cuda_utils.h>
#include <device/gen_random.h>
#include <device/device_defines.h>
#include <device/handles.h>

#include <functions/dev_initializations.h>
#include <functions/eval_gradient.h>

#include <solvers/cg_steihaug.h>

#include <utilities/print_utils.h>
#include <utilities/alloc_sampled_dataset.h>
#include <utilities/sample_matrix.h>

#include <nn/read_nn.h>

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>


void readVecFromFileST( real *dev, real *host, char *f ) { 

   int rows = readVector( host, INT_MAX, f, 0, NULL);
   copy_host_device( host, dev, rows * sizeof(real), cudaMemcpyHostToDevice, 
      ERROR_MEMCPY_HOST_DEVICE );  
}

void testCG (NN_MODEL *model, DEVICE_DATASET *data, 
		SCRATCH_AREA *scratch ) {

	real *ptr = scratch->nextDevPtr; 

	real *x = ptr; 
	real *gradient = x + model->pSize; 
	real *nextDevPtr = gradient + model->pSize; 

	real *ll = scratch->nextPageLckPtr; 
	real *mErr = ll + 1; 
	real *nextPageLck = mErr + 1; 

	scratch->nextDevPtr = nextDevPtr; 
	scratch->nextPageLckPtr = nextPageLck;

   int sampleSize; 
   sampleSize = int( 0.1 * (real) data->trainSizeX );
   allocSampledDataset( data, sampleSize );  
   initSampledROffsets( model, sampleSize );  
   initSampledZOffsets( model, sampleSize );  
	sampleColumnMatrix( data, scratch, 0 );

	readVecFromFileST( data->weights, scratch->nextHostPtr, "./cg-weights.txt" );
	readVecFromFileST( x, scratch->nextHostPtr, "./cg-initial.txt" );

//gradient here. 
   computeGradient( model, data, scratch, data->weights,
            NULL, NULL, gradient, ll, mErr, FULL_DATASET );  

fprintf( stderr, "Starting CG with weights from file.... \n"); 

   CG_PARAMS cg_steihaug_params;
	cg_steihaug_params.x = x ; //initial solution
	cg_steihaug_params.b = gradient;  //gradient
	cg_steihaug_params.delta = 12.0932;
	cg_steihaug_params.errTol = 1e-9;
	cg_steihaug_params.maxIt = 250;
	cg_steihaug_params.cgIterConv = 0;  
	cg_steihaug_params.flag = CG_STEIHAUG_WE_DONT_KNOW;
	cg_steihaug_params.m = 0;


	ConjugateGradientNonLinear( model, data, scratch, &cg_steihaug_params, data->weights );  

fprintf( stderr, "Done with CG.... \n"); 

//Results here. 
real temp = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, model->pSize, x, 1, &temp ) ); 
fprintf( stderr, "Norm of the CG solution is : %6.10f \n\n\n", temp ); 
fprintf( stderr, "Iteration Count: %d \n\n", cg_steihaug_params.cgIterConv ); 
fprintf( stderr, "Model Reduction : %6.10f \n\n", cg_steihaug_params.m); 

}
