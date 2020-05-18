
#include <utilities/mem_estimate.h>

#include <core/datadefs.h>

#include <stdio.h>
#include <stdlib.h>

void estimateMem( NN_MODEL *model, DEVICE_DATASET *data )
{

	fprintf( stderr, "--------------------------------------\n");
	fprintf( stderr, "\t Generating Estimated Mem. usage (Mb)\n"); 
	fprintf (stderr, "--------------------------------------\n");

	fprintf( stderr, "Size of: size_t   %d \n", sizeof(size_t) ); 
	fprintf( stderr, "Size of: int      %d \n", sizeof(int) ); 
	fprintf( stderr, "Size of: real		%d \n", sizeof(real) ); 

	//weights Size
	real w = ((real)(model->pSize * sizeof(real))) / (1000. * 1000.);
	fprintf( stderr, "\n");
	fprintf( stderr, "\n");
	fprintf( stderr, "Weights size: %4.2f \n", w ); 

	//Z size
	real z = ((real)model->zSize * sizeof(real)) / (1000. * 1000.);
	fprintf( stderr, "\n");
	fprintf( stderr, "\n");
	fprintf( stderr, "Z size: %4.2f \n", z ); 

	//dataset size
	real d = ((real) data->features * data->trainSizeX) / (1000. * 1000.);
	fprintf( stderr, "\n");
	fprintf( stderr, "\n");
	fprintf( stderr, "Dataset Size : %4.2f \n", d ); 

	//Gradient scratch
	fprintf( stderr, "\n");
	fprintf( stderr, "\n");
	fprintf( stderr, "Mem requirement for gradient..... \n"); 
	fprintf( stderr, "\t(1) gradient, (1) error, (2) Z \n"); 
	fprintf( stderr, "\t \t %4.2f \n", w + 2.*z + d ); 

	//Hessian vec scratch... 
	fprintf( stderr, "\n");
	fprintf( stderr, "\n");
	fprintf( stderr, "Mem requirement for hessian * vec...\n"); 
	fprintf( stderr, "\t(1) hessvec, (4) z \n" ); 
	fprintf( stderr, "\t\t %4.2f\n", w + 4.*z ); 
}
