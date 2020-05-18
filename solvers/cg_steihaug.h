
#ifndef __H_CG_STEIHAUG__
#define __H_CG_STEIHAUG__

#include <core/datadefs.h>
#include <core/structdefs.h>

#include <nn/nn_decl.h>

typedef enum CG_STEIHAUG_FLAGS {
	CG_STEIHAUG_WE_DONT_KNOW = 0, 
	CG_STEIHAUG_NEGATIVE_CURVATURE = 1, 
	CG_STEIHAUG_HIT_BOUNDRY = 2,
	CG_STEIHAUG_RS_CASE = 3	,
	CG_STEIHAUG_CLOSE_BOUNDRY = 4,
	CG_STEIHAUG_MAX_IT = 5,
	CG_STEIHAUG_SMALL_NORM_G = 6
	
} STEIHAUG_FLAGS;

typedef struct cg_params {
	real *b;
	real delta; 
	real *x; 

	real errTol;
//	real relResidual; 
	real maxIt; 
	int 	cgIterConv; 
	int 	flag; 
	int	hessianType;

	real m;

} CG_PARAMS;

void ConjugateGradientNonLinear( NN_MODEL *model, 
      DEVICE_DATASET *data, SCRATCH_AREA *scratch, 
      CG_PARAMS *params, real *weights ); 

#endif
