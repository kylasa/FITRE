
#ifndef __H_SAMPLE_MATRIX__
#define __H_SAMPLE_MATRIX__

#include <core/datadefs.h>
#include <core/structdefs.h>

void sampleRowMatrix( DEVICE_DATASET *data, SCRATCH_AREA *s ); 
void sampleColumnMatrix( DEVICE_DATASET *data, SCRATCH_AREA *s, int xy ); 

void selectColumnMatrix(DEVICE_DATASET *data,
					int sampleSize, int *indices, int offset ) ;
	

#endif
