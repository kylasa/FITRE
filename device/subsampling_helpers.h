#ifndef __SUB_SAMPLING_HELPERS_H__
#define __SUB_SAMPLING_HELPERS_H__

#include <core/datadefs.h>
#include <core/sparsedefs.h>
#include <core/structdefs.h>

void initSamplingMatrix (int n, SparseDataset *spSamplingMatrix, 
							real *, int sampledSize );

void prepareForSampling (SparseDataset *spSamplingMatrix, 
								real *, real *, int n, int sampleSize, 
								int *hostPtr);

void sampleDataset( SparseDataset *spSamplingMatrix, real *dataset,
                        int n, int cols, int numClasses,
                        real *sampledDataset, int sampleSize );

#endif
