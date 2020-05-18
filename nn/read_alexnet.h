#ifndef __H_READ_ALEXNET__
#define __H_READ_ALEXNET__

#include <nn/nn_decl.h>
#include <core/structdefs.h>

void readAlexNetCNN( CNN_MODEL *model,  int batchSize, int height, int width, 
	int numClasses, int enableBias, int enableBatchNorm );

#endif
