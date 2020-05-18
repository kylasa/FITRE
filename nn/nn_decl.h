#ifndef __H_NN_DECLARATIONS__
#define __H_NN_DECLARATIONS__

#define MAX_LAYERS		20

#include <core/datadefs.h>
#include <stdio.h>
#include <stdlib.h>


//int BATCH_NORM_MOMENTUM = 0.1;

typedef enum network_name{ 
	CNN_LENET = 0, 
	CNN_ALEXNET =1, 
	CNN_VGG11NET = 2, 
	CNN_VGG13NET = 3, 
	CNN_VGG16NET = 4, 
	CNN_VGG19NET = 5 
	} NETWORK_NAME; 

typedef enum batch_norm_types{ 
	PERFORM_NO_BATCH_NORM = 0, 
	PERFORM_BATCH_NORM = 10, 
	PERFORM_BATCH_NORM_TRAINABLE = 20
} BATCH_NORM_TYPES; 

typedef enum eval_type { 
   MODEL_TRAIN = 10, 
   MODEL_TRAIN_ACCURACY = 20, 
   MODEL_TEST_ACCURACY = 30
} EVAL_TYPE; 

typedef enum act_funs{ 

	ACT_NONE = 0, 

	ACT_LOGISTIC = 1, 

	ACT_TANH = 2, 

	ACT_LINEAR = 3, 

	ACT_SOFTMAX = 4

} ACTIVATION_FUNCTIONS; 

typedef enum cnn_act_funs{ 
	CNN_ACT_NONE = 10, 
	CNN_ACT_SOFTPLUS = 11, 
	CNN_ACT_ELU = 12, 
	CNN_ACT_RELU = 13,
	CNN_ACT_SWISH = 14
} CNN_ACTIVATION_FUNCTIONS; 

typedef enum model_type { 

	MODEL_TYPE_MSE = 0,

	MODEL_TYPE_CLASSIFICATION = 1

} MODEL_TYPES;

typedef struct conv{ 
	int inChannels; 
	int outChannels; 

	int width; 
	int height; 
	int outWidth; 
	int outHeight;

	int kSize; 

	int stride; 
	int padding; 

	int activationOffset; 
	int poolOffset;
	int batchNormOffset; 
	int outputOffset; 

	BATCH_NORM_TYPES batchNorm; 
	int meansOffset; 
	int variancesOffset; 

	int runningMeansOffset; 
	int runningVariancesOffset; 

	int convVolumn; 
	int activationVolumn; 
	int poolVolumn; 
	int batchNormVolumn; 

} CONV_LAYER; 

typedef enum pool_type{ 
	MAX_POOL = 0,
	AVG_POOL = 1,
	NO_POOL = 2
} POOL_TYPE; 

typedef struct pool{ 
	int height; 
	int width; 
	int outHeight; 
	int outWidth;
	int type;	
	int pSize; 
	int padding; 
	int stride;

} POOL_LAYER; 

typedef struct full_connect{ 
	int in; 
	int out; 
	int actFun; 
	int batchNorm; 
	int offset;  //TODO temp storage
} FC_LAYER; 

typedef struct cnn{ 

	//Convolutions
	CONV_LAYER convLayer[ MAX_LAYERS ];
	int actFuns[ MAX_LAYERS ]; 
	POOL_LAYER poolLayer[ MAX_LAYERS ];
	int cLayers; 

	//Linear Layers
	int lLayers; 
	FC_LAYER fcLayer[ MAX_LAYERS ]; 

	//common variables. 
	int bias;
	int pSize; 
	int zSize; 
	int batchSize; 
	int maxDeltaSize; 
	int name; 
	int enableBatchNorm;
	
	int wOffsets[ MAX_LAYERS ];
	int bOffsets[ MAX_LAYERS ];
	int zOffsets[ MAX_LAYERS ];

	int zztSize; 
	int zztOffsets[ MAX_LAYERS ]; 


	//Reverse mode variables. 
	int   rZOffsets[ MAX_LAYERS ];

} CNN_MODEL;

typedef struct nn{ 
	
	int 	layerSizes[ MAX_LAYERS ]; 
	int 	actFuns[ MAX_LAYERS ]; 

	int 	numLayers; 
	int	type; 

	//int 	weightsSize; 
	//int	biasSize; 
	int 	pSize; 
	int   zSize; 
	int	sampledZSize;
	int   sampledRSize;
	int   rFullSize;

	//increments for dW and dB pointers
	int	wOffsets[ MAX_LAYERS ]; 
	int 	bOffsets[ MAX_LAYERS ];

	int 	zOffsets[ MAX_LAYERS ];
	int 	sZOffsets[ MAX_LAYERS ];

	int   rZOffsets[ MAX_LAYERS ];
	int	sRZOffsets[ MAX_LAYERS ];
	
} NN_MODEL; 

//#ifdef STATS
typedef struct OUT_PARAMS {

	//Model Part
	real trainLL; 
	real trainModelErr; 
	real testLL; 
	real testModelErr; 

	//TR-Part
	int iteration; 
	real delta; 
	real rho; 
	real normGrad;
	int failCount; 
	int numMVP; 
	unsigned long int numProps; 
	real trTime; 
	
	//CG-Part
	int cgIterations; 
	real hessVecTime; 
	real normS;	 //CG solution per Trust Iteration
	real cgTime;

	FILE *out;
	
} OUT_PARAMS; 
//#endif

#endif
