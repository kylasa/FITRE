
#ifndef __KFAC_STRUCTS_H__
#define __KFAC_STRUCTS_H__

#include <core/datadefs.h>
#include <nn/nn_decl.h>


/*
	KFAC Curvature Matrix here. 	
*/

typedef struct kfac_curvature{

	// Used to store ZZT and GGT
	real *OmegaZZT; 
	real *LambdaGGT; 

	// Use to store ZZT and GGT INVERSES
	real *OmegaZInv; 
	real *LambdaGInv; 

	real *vec;
	real *nGradient; 
	real *gradient; 

	int	OmegaZOffsets [ MAX_LAYERS ]; 
	int	LambdaGOffsets [ MAX_LAYERS ];

	// MOMENTUM Params here. 
	real	stats_decay; 
	real 	momentum; 
	real  regLambda; 
	real  dampGamma; 

	int 	checkGrad; 

	//speed up options
	int 	dampedInputSize; 
	real 	*dampedInput; 
	real 	*dampedZ; 	
	real	*dampedLambda; 

   // Threaded Implementation. 
   // no. of mini-batches... 
   // Typically = { 1, 10, 20, 50 }
   int   inverseFreq; 

} KFAC_CURVATURE_INFO;


#endif
