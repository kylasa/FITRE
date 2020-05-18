
#ifndef __VGG_NET__
#define __VGG_NET__

#include <nn/nn_decl.h>
#include <core/structdefs.h>

void readTestVGG( CNN_MODEL *model, int batchSize, int height, int width, int numClasses, int inChannels, int bias, int bn, DATASET_TYPE d );
void readVGG11( CNN_MODEL *model, int batchSize, int height, int width, int numClasses, int inChannels, int bias, int bn, DATASET_TYPE d );
void readVGG13( CNN_MODEL *model, int batchSize, int height, int width, int numClasses, int inChannels, int bias, int bn, DATASET_TYPE d  );
void readVGG16( CNN_MODEL *model, int batchSize, int height, int width, int numClasses, int inChannels, int bias, int bn, DATASET_TYPE d  );
void readVGG19( CNN_MODEL *model, int batchSize, int height, int width, int numClasses, int inChannels, int bias, int bn, DATASET_TYPE d  );

#endif
