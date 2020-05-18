#ifndef _H_DATASET__
#define _H_DATASET__

#include <core/datadefs.h>
#include <core/structdefs.h>

void convertToBatchColumnMajor( real *src, int rows, int cols, 
   real *tgt, int memBatchSize, int channels, int volumn,  real normConst );

void readDataset( char *f_train_features, char *f_train_labels,
                char *f_test_features, char *f_test_labels, HOST_DATASET *data, 
					int batchSize, int channels, int volume );

int tokenize_string( char *line);
void tokenize_populate(char *line, real *train_set, int *count, int size );

void initialize_device_data( HOST_DATASET *s, DEVICE_DATASET *t);
void initialize_device_image( HOST_DATASET *s, DEVICE_DATASET *t) ;

void cleanup_dataset( HOST_DATASET *s, DEVICE_DATASET *t);

void readCIFARDataset( char *dir, char *train, int numFiles, char *test, 
		HOST_DATASET *data, SCRATCH_AREA *s, int raw, int batchSize, int datasetType, int height, int width);


#endif
