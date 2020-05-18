
#include <drivers/derivative_driver.h>

#include <nn/read_nn.h>

#include <utilities/sample_matrix.h>
#include <utilities/derivative_test.h>
#include <utilities/cnn_derivative_test.h>
#include <utilities/alloc_sampled_dataset.h>

void runDerivativeTest (NN_MODEL *model, 
	DEVICE_DATASET *data, SCRATCH_AREA *scratch )
{

   int sampleSize; 
   sampleSize = int( 0.1 * (real) data->trainSizeX );
   allocSampledDataset( data, sampleSize );  
   
   fprintf( stderr, "Hessian Vec testing started .... %d \n", sampleSize );  

   initSampledROffsets( model, sampleSize );  
   initSampledZOffsets( model, sampleSize );  

   fprintf( stderr, "Creating sampling matrix... \n"); 
   sampleColumnMatrix( data, scratch, 0 );  
   fprintf( stderr, "Sampling done... \n"); 


	derivativeTest( model, data, scratch ); 
}

void runCNNDerivativeTest( CNN_MODEL *model, 
	DEVICE_DATASET *data, SCRATCH_AREA *scratch )
{
	cnnDerivativeTest( model, data, scratch ); 
}
	
