
#include <drivers/test_cnn_gradient.h>

#include <functions/cnn_forward.h>

inline int getMaxZSizes( int *elements, int count )
{  
   int m = elements [ 0 ]; 
   for (int i = 1; i < count; i ++)
      if ( m < elements[ i ] ) m = elements[ i ];
   
   return m;
}

void testCNNGradient( CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch )
{

   real *z = scratch->nextDevPtr; 
	real *probs = z + model->pSize; 
   real *errors = probs + data->numClasses * data->trainSizeX; 
   real *nextDevPtr = errors + getMaxZSizes( model->zOffsets, model->cLayers + model->lLayers + 1); 

   scratch->nextDevPtr = nextDevPtr; 
   real nll = cnnForward( model, data, scratch, z, probs, errors, 0, 256, MODEL_TRAIN );  

	fprintf( stderr, "Log Likelihood for CIFAR-10 is %4.8f \n", nll ); 

}
