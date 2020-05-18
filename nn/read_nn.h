#ifndef __H_READ_NN__
#define __H_READ_NN__

#include <nn/nn_decl.h>
#include <core/structdefs.h>

void readFCCNN( CNN_MODEL *model, int batchSize ) ;

void readConvCNN( CNN_MODEL *model,  
		int channels, int out_channels, int numclasses, int width, int height, int batchSize );

void readConv2CNN( CNN_MODEL *model,  
		int channels, int out_channels, int numclasses, int width, int height, int batchSize );

void readTestCNN( CNN_MODEL *model,  
		int channels, int out_channels, int numclasses, int width, int height, int batchSize );

void readLenetCNN( CNN_MODEL *model,  
		int channels, int width, int height, int batchSize, int enableBias, int enableBatchNorm );

void readNeuralNet( NN_MODEL *model, int inputSize, 
		int outputSize, int numPoints );

void initSampledROffsets( NN_MODEL *model, int sampleSizes); 
void initSampledZOffsets( NN_MODEL *model, int points); 
void initRZOffsets( NN_MODEL *model, int points); 

void autoencoderInitializations ( NN_MODEL *model, 	
		DEVICE_DATASET *data );

void cnnInitializations( CNN_MODEL *mode, DEVICE_DATASET *data ); 

#endif
