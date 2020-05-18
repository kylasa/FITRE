#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <utilities/dataset.h>
#include <device/cuda_utils.h>
#include <utilities/utils.h>
#include <utilities/print_utils.h>
#include <core/errors.h>


#define MAX_LINE 	4 * 1024 * 1024
#define MAX_IDX 	256 * 1024
#define HEAP_LINE_SIZE 4 * 1024 * 1024

#define CIFAR_LINE_SIZE    3073
#define CIFAR100_LINE_SIZE    3074
#define TINY_LINE_SIZE    ((64 * 64 * 3) + 1)


void convertToColumnMajor (real *src, int rows, int cols, real *tgt ) {
   for (int i = 0; i < rows; i ++ )
      for (int j = 0; j < cols; j ++)
         tgt[j * rows + i]  = src[i * cols + j];
}


void convertCIFARToColumnMajor( real *src, real *tgt, int datasetSize, int channels, int height, int width) {

	int volumn = height * width; 
	for (int img = 0; img < datasetSize; img ++ ){
		for (int c = 0; c < channels; c ++ ) {
			for (int v = 0; v < volumn; v ++) {
				int row = v / width; 
				int col = v % height; 
				tgt[ img * channels * volumn + 
						c * volumn + col * height + row ] = 
				src[ img * channels * volumn + c * volumn + row * width + col ]; 
			}
		}
	}
}


void convertToBatchColumnMajor( real *src, int rows, int cols, 
	real *tgt, int memBatchSize, int channels, int height, int width,  real normConst ){

	int imgInBatch, chIdx, batchIdx, volumnIdx;
	int batchSize = memBatchSize; 
	int rowInVol, colInVol; 
	int volumn = height * width;
	int numBatches = (rows + batchSize - 1) / batchSize; 

	fprintf( stderr, "ConvertToBatchColumnMajor: Rows: %d, Cols: %d, batchSize: %d, Channels: %d, Volumn: %d \n", rows, cols, memBatchSize, channels, height * width); 

	for (int b = 0; b < numBatches; b ++) {

		int start = b * batchSize; 
		int end = (b+1) * batchSize; 
		if (end > rows)  { 
			end = rows; 
			batchSize = end - start; 
		}

		for (int i = start; i < end; i ++) {

			imgInBatch = i % batchSize;  // 1024
			batchIdx = b; //i / batchSize;  // 1024
			for (int j = 0; j < cols; j ++) { // cols = 1 .. 3072

				chIdx = j / (height * width);  // 3
				volumnIdx = j % volumn;  //1024
				rowInVol = volumnIdx / width; 
				colInVol = volumnIdx % height; 
				if( (
					batchIdx * channels * memBatchSize * volumn + 	// 0 - num batches
					chIdx * volumn * batchSize + 						// 0 - channels
					imgInBatch * volumn + 								// 0 - batchSize
					volumnIdx 												// 0 - 1023
					) >= (rows * cols) ) { 
					fprintf( stderr, " DOOOOOOOOM..... \n"); 
					fprintf( stderr, "batchIdx: %d, imgInBatch: %d, chIdx: %d, volumnIdx: %d \n", 
								batchIdx, imgInBatch, chIdx, volumnIdx ); 
					exit( -1 ); 
				} else {
					/*	 ROW MAJOR FORMAT HERE 
					tgt [
						batchIdx * channels * batchSize * volumn + 	// 0 - num batches
						chIdx * volumn * batchSize + 						// 0 - channels
						imgInBatch * volumn + 								// 0 - batchSize
						volumnIdx 												// 0 - 1023
					] = src[ i * cols + j ] / normConst; 
					*/

					// Column Major Format for each of teh channels 
					tgt[ 
						batchIdx * channels * memBatchSize * volumn + 
						chIdx * volumn * batchSize + 
						imgInBatch * volumn + 
						colInVol * height + rowInVol
						] = 
							src[ i * cols + j ] ;
				}
			}
		}
	}
}

/*
CIFAR10 Statistics:  (tensor([0.4914, 0.4822, 0.4465]), tensor([0.2470, 0.2435, 0.2616]))
CIFAR100 Statistics:  (tensor([0.5071, 0.4865, 0.4409]), tensor([0.2673, 0.2564, 0.2762]))
ImageNet Statistics:  (tensor([0.4799, 0.4478, 0.3972]), tensor([0.2769, 0.2690, 0.2820]))
*/

void imageNormalization( real *src, int numImages, int channels, int height, int width, real constant, int datasetType )
{

	real mean_cifar10[3] = {0.4914, 0.4822, 0.4465}; 
	real std_cifar10[3] = {0.2470, 0.2435, 0.2616};

	real mean_cifar100[3] = {0.5071, 0.4865, 0.4409}; 
	real std_cifar100[3] = {0.2673, 0.2564, 0.2762};

	real mean_imagenet[3] = {0.4799, 0.4478, 0.3972};
	real std_imagenet[3] = {0.2769, 0.2690, 0.2820};

	real *mean, *std;

	switch( datasetType ) {
		case 1: 
			mean = mean_cifar10;
			std = std_cifar10; 
			break;
		case 2: 
			mean = mean_cifar100;
			std = std_cifar100; 
			break;
		case 3: 
			mean = mean_imagenet;
			std = std_imagenet; 
			break;

		default: 
			fprintf( stderr, "Unknown DatasetType here... \n\n"); 
			exit( -1 ); 
	}

	int count = numImages * channels * height * width; 

	int chIdx = -1; 
	for (int i = 0; i < count; i ++) {
			
		src[ i ] /= constant; 

		chIdx = i / (height * width * numImages); 
		src[ i ] = (src[ i ] - mean[ chIdx ]) / std[ chIdx ]; 
	}
}

void columnNormalize( real *src, int rows, int cols, real *train, int tr ){
        real norm = 0;
        for (int c = 0; c < cols; c ++){
                norm = pow( src[ c * rows ], 2. );
                for (int r = 1; r < rows; r ++) {
                        norm += pow( src[ c * rows + r ], 2. );
                }
                for (int r = 0; r < tr; r ++){
                        norm += pow( src[ c * tr + r ], 2. );
                }

                if (norm < 1e-8) {
                        norm = sqrt( norm );
                        for (int r = 0; r < rows; r ++)
                                src[ c * rows + r ] /= norm;

                        for (int r = 0; r < tr; r ++)
                                train[ c * tr + r ] /= norm;
                }
        }
}

void columnNormalizeTrain( real *src, int rows, int cols ){
        real norm = 0;
        for (int c = 0; c < cols; c ++){
                norm = pow( src[ c * rows ], 2. );
                for (int r = 1; r < rows; r ++) {
                        norm += pow( src[ c * rows + r ], 2. );
                }

                if (norm > 1e-8) norm = sqrt( norm );
                	for (int r = 0; r < rows; r ++)
                  	src[ c * rows + r ] /= norm;
        }
}



void readDelimitedMatrix( char *fileName, real **data, int *rows, int *cols )
{
	FILE *dataset_file;
	char line[HEAP_LINE_SIZE];

	fprintf( stderr, "Reading file: %s \n", fileName );

	if ( (dataset_file = fopen(fileName, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	*rows = *cols = 0;
	while (!feof( dataset_file) ){
		memset( line, 0, HEAP_LINE_SIZE);
		fgets( line, HEAP_LINE_SIZE, dataset_file);

		if (line[0] == 0) break;
		*cols = tokenize_string( line );
		(*rows) ++;
	}
	fprintf(stderr, "Rows : %d, Columns : %d \n", *rows, *cols ); 

	*data = (real *)malloc(  (*cols) * (*rows) * sizeof(real));

	//read the file here and fill the matrix. 
	rewind( dataset_file );	
	*rows = 0;

	while (!feof( dataset_file )){
		memset( line, 0, HEAP_LINE_SIZE);
		fgets( line, HEAP_LINE_SIZE, dataset_file);
		if (line[0] == 0) break;
		tokenize_populate( line, *data, rows, *cols);
		(*rows) ++;
	}
	fclose( dataset_file );
	fprintf( stderr, "Done reading: %s \n", fileName );
}

void readDataset( char *f_train_features, char *f_train_labels, 
		char *f_test_features, char *f_test_labels, HOST_DATASET *data, 
		int batchSize, int channels, int volume )
{
	int rows, cols;

	//read train X
	rows = cols = 0;
	real *temp;
	readDelimitedMatrix (f_train_features, &temp, &rows, &cols ); 
	//convert to the column major order. 
	data->trainSetX = (real *)malloc( sizeof(real) * rows * cols );

	//convertToColumnMajor( temp, rows, cols, data->trainSetX );
	convertToBatchColumnMajor( temp, rows, cols, data->trainSetX, 
		batchSize, channels, 32, 32, 1. ); 
	fprintf( stderr, "Done with Column Major conversion... \n"); 

	free( temp );
	data->trainSizeX = rows;
	data->features = cols;

	//read train Y
	rows = cols = 0;
	readDelimitedMatrix( f_train_labels, &data->trainSetY, &rows, &cols ); 
	data->trainSizeY = rows;

	//read test X
	/*
	rows = cols = 0;
	readDelimitedMatrix (f_test_features, &temp, &rows, &cols ); 
	data->testSetX = (real *) malloc( sizeof (real ) * rows * cols ); 
	convertToColumnMajor( temp, rows, cols, data->testSetX ); 
	free( temp ); 
	data->testSizeX = cols;
	*/

/*
	//read test Y
	rows = cols = 0;
	readDelimitedMatrix( f_test_labels, &data->testSetY, &rows, &cols ); 	
	data->testSizeY = cols;
*/
}

int readCIFAR10File (char *filename, size_t lineSize, real *train_labels, real *train_set, int numLines ) { 

   FILE *dataset_file;
   char line[MAX_LINE];
	unsigned char minLabel, maxLabel; 
   size_t output;
	int i; 
	minLabel = 255; maxLabel = 0; 

	if ( (dataset_file = fopen(filename, "r")) == NULL ) {
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	if( (train_labels != NULL) && ( train_set != NULL ) ) {
	
   	while (!feof( dataset_file) ){
   		memset( line, 0, MAX_LINE );
      	output = fread( line, (size_t)1, lineSize, dataset_file);

      	if (output <= 0) break;

      	train_labels[ numLines ] = ((unsigned char) line[0] ) + 1;
      	for (i = 0; i < lineSize -1; i ++)
      		train_set[ numLines * (lineSize - 1) + i ] = (unsigned char) line[i + 1];

			/*
			fprintf( stderr, "Label: %d \n", (unsigned char) line[ 0 ] + 1 ); 
			for (int j = 0; j < 3; j ++) { 
				for ( i = 0; i < 3; i ++) 
					fprintf( stderr, " %d ", (unsigned char) train_set[ numLines * (lineSize - 1) + i * 64 + j ] ); 
				fprintf( stderr, "Done with Image...\n"); 
			}
			*/

      	numLines ++;
   	}
	} else { 

		//int printDone = 0; 
   	while (!feof( dataset_file) ){
   		memset( line, 0, MAX_LINE );
      	output = fread( line, (size_t)1, lineSize, dataset_file);

      	if (output <= 0) break;

			/*
			if ( printDone == 0){ 
				for (int i = 0; i < 20; i ++ ) fprintf( stderr, "%u ", (unsigned char)line[i + 1] ); 
				fprintf( stderr, "\n"); 
				printDone = 1; 
			} 
			*/
  
			/*
			fprintf( stderr, "Label: %d \n", (unsigned char) line[ 0 ] + 1 ); 
			for (int j = 0; j < 3; j ++) { 
				for ( i = 0; i < 3; i ++) 
					fprintf( stderr, " %d ", (unsigned char)line[ 64 * 64 + i * 64 + j  + 1] ); 
				fprintf( stderr, "\n"); 
			}
			*/

			if ( (unsigned char)(line[0] + 1) < minLabel ) minLabel = (unsigned char) (line[0] + 1); 
			if ( maxLabel < (unsigned char)(line[0] + 1) ) maxLabel = (unsigned char) (line[0] + 1); 

      	numLines ++;
   	}
   	fprintf( stderr, "Done with reading %d points from the input files, Label( %u, %u ) .... \n", numLines, (unsigned int)minLabel, (unsigned int) maxLabel );
	}

	return numLines; 
}


int readCIFAR100File (char *filename, size_t lineSize, real *train_labels, real *train_set, int numLines ) { 

   FILE *dataset_file;
   char line[MAX_LINE];
	int minLabel, maxLabel; 
   size_t output;
	int i; 
	minLabel = INT_MAX; maxLabel = 0; 

	if ( (dataset_file = fopen(filename, "r")) == NULL ) {
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	if( (train_labels != NULL) && ( train_set != NULL ) ) {
	
   	while (!feof( dataset_file) ){
   		memset( line, 0, MAX_LINE );
      	output = fread( line, (size_t)1, lineSize, dataset_file);

      	if (output <= 0) break;

      	train_labels[ numLines ] = (unsigned char)(line[1] + 1);
      	for (i = 0; i < lineSize -1; i ++)
      		train_set[ numLines * (lineSize - 2) + i ] = (unsigned char) line[i + 2];

      	numLines ++;
   	}
	} else { 

   	while (!feof( dataset_file) ){
   		memset( line, 0, MAX_LINE );
      	output = fread( line, (size_t)1, lineSize, dataset_file);

      	if (output <= 0) break;

			if ( ((unsigned char)line[1] + 1) < minLabel ) minLabel = (unsigned char) line[1] + 1; 
			if ( maxLabel < ((unsigned char)line[1] + 1) ) maxLabel = (unsigned char)line[1] + 1; 

      	numLines ++;
   	}
   	fprintf( stderr, "Done with reading %d points from the input files, Label( %d, %d ) .... \n", numLines, minLabel, maxLabel );
	}

	return numLines; 
}



//CIFAR-10 dataset here. 
void readCIFARDataset( char *dir, char *train, int TRAIN_FILES, char *test,
		HOST_DATASET *data, SCRATCH_AREA *s, int imgNormalize, int batchSize, int datasetType, int height, int width) {

   int idx = 0;

   int TRAIN_IMAGES = 50000;
	int TEST_IMAGES = 10000; 
	int lineSize;
	int NUM_CLASSES;
	int offset; 

   char filename[MAX_LINE]; 
   real *train_set, *train_labels, *test_set, *test_labels;
   real *scratch = s->hostWorkspace;


	TRAIN_IMAGES = 0; 
   for (idx = 1; idx <= TRAIN_FILES; idx ++) {
      sprintf( filename, "%s%s%d.bin", dir, train, idx);
      fprintf( stderr, "Reading file : %s \n", filename );

		switch( datasetType ) {
			case 1:
				lineSize = CIFAR_LINE_SIZE; 
				NUM_CLASSES = 10; 
				TRAIN_IMAGES = readCIFAR10File( filename, lineSize, NULL, NULL, TRAIN_IMAGES ); 
				break;
			case 2: 	
				lineSize = CIFAR100_LINE_SIZE; 
				NUM_CLASSES = 100; 
				TRAIN_IMAGES = readCIFAR100File( filename, lineSize, NULL, NULL, TRAIN_IMAGES ); 
				break;
			case 3: 
				lineSize = TINY_LINE_SIZE; 
				NUM_CLASSES = 200; 
				TRAIN_IMAGES = readCIFAR10File( filename, TINY_LINE_SIZE, NULL, NULL, TRAIN_IMAGES ); 
				break;
		}
   }
	fprintf( stderr, "Train Images: %d \n", TRAIN_IMAGES ); 


   //test data here. 
	TEST_IMAGES = 0; 
   memset( filename, 0, MAX_LINE );
   sprintf( filename, "%s%s", dir, test);
	if (datasetType == 1)
		TEST_IMAGES = readCIFAR10File( filename, lineSize, NULL, NULL, TEST_IMAGES ); 
	else if (datasetType == 2)
		TEST_IMAGES = readCIFAR100File( filename, lineSize, NULL, NULL, TEST_IMAGES ); 
	else { 
		TEST_IMAGES = 0;
   	memset( filename, 0, MAX_LINE );
   	sprintf( filename, "%s%s%d.bin", dir, test, 1);
      fprintf( stderr, "Reading file : %s \n", filename );
		TEST_IMAGES = readCIFAR10File( filename, lineSize, NULL, NULL, TEST_IMAGES ); 

   	memset( filename, 0, MAX_LINE );
   	sprintf( filename, "%s%s%d.bin", dir, test, 2);
      fprintf( stderr, "Reading file : %s \n", filename );
		TEST_IMAGES = readCIFAR10File( filename, lineSize, NULL, NULL, TEST_IMAGES ); 
	} 


   train_set = (real *) malloc( (size_t)TRAIN_IMAGES * (lineSize - 1 - ((datasetType != 2) ? 0 : 1)) * sizeof(real) );
   train_labels = (real *) malloc ( (size_t)TRAIN_IMAGES * sizeof(real) ); 
   test_set = (real *) malloc( (size_t)TEST_IMAGES * (lineSize - 1 - ((datasetType != 2) ? 0 : 1)) * sizeof(real) );
   test_labels = (real *) malloc ( (size_t)TEST_IMAGES * sizeof(real) ); 

   fprintf( stderr, " Allocated memory for the dataset : %lu \n", TRAIN_IMAGES * (lineSize - 1 - ((datasetType != 2) ? 0 : 1)) * sizeof(real));
   fprintf( stderr, " Allocated memory for the dataset (GB): %d \n", (TRAIN_IMAGES * (lineSize - 1 - ((datasetType != 2) ? 0 : 1)) * sizeof(real)) / (1024 * 1024 * 1024));

	TRAIN_IMAGES = 0; 
   for (idx = 1; idx <= TRAIN_FILES; idx ++) {
      sprintf( filename, "%s%s%d.bin", dir, train, idx);
      fprintf( stderr, "Reading file : %s \n", filename );

		switch(datasetType) {
			case 1: 
				TRAIN_IMAGES = readCIFAR10File( filename, lineSize, train_labels, train_set, TRAIN_IMAGES ); 
				break;
			case 2: 	
				TRAIN_IMAGES = readCIFAR100File( filename, lineSize, train_labels, train_set, TRAIN_IMAGES ); 
				break;
			case 3: 
				TRAIN_IMAGES = readCIFAR10File( filename, lineSize, train_labels, train_set, TRAIN_IMAGES ); 
				break;
		}
   }
   fprintf( stderr, "Done with reading %d points from the input files .... \n", TRAIN_IMAGES );

   //test data here. 
   memset( filename, 0, MAX_LINE );
   sprintf( filename, "%s%s", dir, test);

	TEST_IMAGES = 0 ; 
	if (datasetType == 1)
		TEST_IMAGES = readCIFAR10File( filename, lineSize, test_labels, test_set, TEST_IMAGES ); 
	else if (datasetType == 2)
		TEST_IMAGES = readCIFAR100File( filename, lineSize, test_labels, test_set, TEST_IMAGES ); 
	else { 
		TEST_IMAGES = 0;
   	memset( filename, 0, MAX_LINE );
   	sprintf( filename, "%s%s%d.bin", dir, test, 1);
		TEST_IMAGES = readCIFAR10File( filename, lineSize, test_labels, test_set, TEST_IMAGES ); 

   	memset( filename, 0, MAX_LINE );
   	sprintf( filename, "%s%s%d.bin", dir, test, 2);
		TEST_IMAGES = readCIFAR10File( filename, lineSize, test_labels , test_set , TEST_IMAGES ); 
	}

   fprintf( stderr, "Done with reading %d points from the input files .... \n", TEST_IMAGES );

   //inititalize the device data here. 
   data->trainSizeX = TRAIN_IMAGES;
   data->testSizeX = TEST_IMAGES;

   data->trainSetX = train_set;
   data->trainSetY = train_labels;
   data->testSetX = test_set;
   data->testSetY = test_labels;

   data->features = lineSize - 1 - ((datasetType != 2) ? 0 : 1);
   data->numClasses = NUM_CLASSES;

   fprintf( stderr, "Converting to column major format here.... \n");
   //train_features
	convertCIFARToColumnMajor( train_set, scratch, data->trainSizeX, 3, height, width ); 
   //convertToColumnMajor( train_set, data->trainSizeX, data->features, scratch);
   fprintf( stderr, "Done with conversion... \n");
   memcpy( train_set, scratch, (size_t)(sizeof(real) * data->trainSizeX * data->features) );

	/*
   //test_features
   convertToColumnMajor( test_set, data->testSizeX, data->features, scratch);
   fprintf( stderr, "Done with conversion... \n");
   memcpy( test_set, scratch, (size_t)(sizeof(real) * data->testSizeX * data->features) );
	*/
/*
	//Train features... 
	fprintf( stderr, "Converting to BATCH column major format here ... batchSize: %d\n", batchSize); 
	convertToBatchColumnMajor( train_set, data->trainSizeX, data->features, 
			scratch, batchSize, 3, 1024, 1. ); 
	fprintf( stderr, "Done with conversion... \n"); 
	//columnNormalizeTrain( train_set, data->trainSizeX, data->features ); 
	memcpy( train_set, scratch, (size_t)( sizeof(real) * data->trainSizeX * data->features ) ); 
*/

	//Test features
	fprintf( stderr, "Converting to BATCH column major format here ... batchSize: %d\n", batchSize); 

	convertCIFARToColumnMajor( test_set, scratch, data->testSizeX, 3, height, width ); 
	//convertToBatchColumnMajor( test_set, data->testSizeX, data->features, 
	//		scratch, batchSize, 3, height , width, 1. ); 
	//columnNormalizeTrain( test_set, data->testSizeX, data->features ); 
	fprintf( stderr, "Done with conversion... \n"); 
   memcpy( test_set, scratch, (size_t)(sizeof(real) * data->testSizeX * data->features) );

	/*
   if (raw == 0){
       fprintf( stderr, "Normalizing the data ... ");
       columnNormalize( train_set, data->trainSizeX, data->features, test_set, data->testSizeX );
       fprintf( stderr, "Done... \n");
   }
	*/
	
/*
	if (imgNormalize != 0){
		//Image Normalization... 
		//imageNormalization( train_set, data->features * data->trainSizeX, imgNormalize ); 
		//imageNormalization( test_set, data->features * data->testSizeX, imgNormalize ); 

		//imageNormalization( train_set, data->trainSizeX, 3, 32, 32, imgNormalize ); 
		imageNormalization( test_set, data->testSizeX, 3, height, width, imgNormalize, datasetType ); 
	}
*/
}


int tokenize_string( char *line )
{
	const char *sep = ", \n";
	char *word;
	char temp[MAX_LINE];
	int index = 0; 

	strncpy( temp, line, MAX_LINE );
	for( word = strtok(temp, sep); word; word = strtok(NULL, sep) ) index ++;

	return index;
}

void tokenize_populate(char *line, real *train_set, int *count, int size ){

	const char *sep = ", \n";
	char *word;
	char temp[MAX_LINE];
	int index = 0; 
	real cur_row[MAX_LINE]; 

	strncpy( temp, line, MAX_LINE );
	for( word = strtok(temp, sep); word; word = strtok(NULL, sep) ) cur_row[ index ++] = atof( word );
	memcpy( &train_set[ (*count) * (size)], cur_row, sizeof(real) * size);
}


//
//
// Device Functions here. 
//
//
void initialize_device_data( HOST_DATASET *s, DEVICE_DATASET *t)
{
	t->trainSizeX = s->trainSizeX;
	t->trainSizeY = s->trainSizeY;

	t->testSizeX = s->testSizeX;
	t->testSizeY = s->testSizeY;

	t->features =  s->features;
	t->height = s->height; 
	t->width = s->width; 
	t->numClasses = s->numClasses; 

	fprintf( stderr, "Dataset Report: ...\n"); 
	fprintf( stderr ,"Train Set Size: %d \n", s->trainSizeX ); 
	fprintf( stderr, "Features: %d \n", s->features ); 
	fprintf( stderr, "numClasses: %d \n", s->numClasses ); 

	fprintf( stderr, "initialize_device_data: Synching the device... %d, %ld \n", 
			t->trainSizeX * t->features * sizeof(real), 
			(long)(t->trainSizeX * t->features * sizeof(real))); 

	cuda_malloc( (void **)&t->trainSetX, t->trainSizeX * t->features * sizeof(real), 0, ERROR_MEM_ALLOC );
	copy_host_device( s->trainSetX, t->trainSetX, t->trainSizeX * t->features * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSETX );
	fprintf( stderr, "Allocated the memory... \n"); 
	printHostVector( s->trainSetX, 10, NULL ); 

	cuda_malloc( (void **)&t->trainSetY, t->trainSizeX  * sizeof(real), 0, ERROR_MEM_ALLOC );
	copy_host_device( s->trainSetY, t->trainSetY, t->trainSizeX * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSETY );
	printHostVector( s->trainSetY, 10, NULL ); 
	fprintf( stderr, "Allocated the memory... %zu\n", t->trainSizeX); 

	cuda_malloc( (void **)&t->testSetX, t->testSizeX * t->features * sizeof(real), 0, ERROR_MEM_ALLOC );
	fprintf( stderr, "Allocated the memory... \n"); 
	copy_host_device( s->testSetX, t->testSetX, t->testSizeX * t->features * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TESTSETX );

	cuda_malloc( (void **)&t->testSetY, t->testSizeX * sizeof(real), 0, ERROR_MEM_ALLOC );
	fprintf( stderr, "Allocated the memory... \n"); 
	copy_host_device( s->testSetY, t->testSetY, t->testSizeX * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TESTSETY);

	fprintf (stderr, " -------------- \n");
	fprintf( stderr, "Train Set size: %d, %d, %d \n", t->features, t->trainSizeX, t->numClasses);
	fprintf( stderr, "Test Set size: %d, %d, %d \n", t->features, t->testSizeX, t->numClasses);
	fprintf (stderr, " -------------- \n");

}

void initialize_device_image( HOST_DATASET *s, DEVICE_DATASET *t)
{
	t->trainSetX = NULL; 
	t->trainSetY = NULL; 
	t->trainSizeX = s->trainSizeX;
	t->trainSizeY = s->trainSizeY;

	t->testSizeX = s->testSizeX;
	t->testSizeY = s->testSizeY;

	t->features =  s->features;
	t->height = s->height; 
	t->width = s->width; 
	t->numClasses = s->numClasses; 

	fprintf( stderr, "Dataset Report: ...\n"); 
	fprintf( stderr ,"Train Set Size: %d \n", s->trainSizeX ); 
	fprintf( stderr, "Features: %d \n", s->features ); 
	fprintf( stderr, "numClasses: %d \n", s->numClasses ); 

	fprintf( stderr, "initialize_device_data: Synching the device... %d, %ld \n", 
			t->trainSizeX * t->features * sizeof(real), 
			(long)(t->trainSizeX * t->features * sizeof(real))); 

	t->trainSetX = NULL; 
	t->trainSetY = NULL; 

	printHostVector( s->trainSetX, 10, NULL ); 
	printHostVector( s->trainSetY, 10, NULL ); 

	cuda_malloc( (void **)&t->testSetX, t->testSizeX * t->features * sizeof(real), 0, ERROR_MEM_ALLOC );
	fprintf( stderr, "Allocated the memory... \n"); 
	copy_host_device( s->testSetX, t->testSetX, t->testSizeX * t->features * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TESTSETX );

	cuda_malloc( (void **)&t->testSetY, t->testSizeX * sizeof(real), 0, ERROR_MEM_ALLOC );
	fprintf( stderr, "Allocated the memory... \n"); 
	copy_host_device( s->testSetY, t->testSetY, t->testSizeX * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TESTSETY);

	fprintf (stderr, " -------------- \n");
	fprintf( stderr, "Train Set size: %d, %d, %d \n", t->features, t->trainSizeX, t->numClasses);
	fprintf( stderr, "Test Set size: %d, %d, %d \n", t->features, t->testSizeX, t->numClasses);
	fprintf (stderr, " -------------- \n");
}


void cleanup_dataset( HOST_DATASET *s, DEVICE_DATASET *t){
	if (s->trainSetX) release_memory( (void **)&s->trainSetX );
	if (s->trainSetY ) release_memory( (void **)&s->trainSetY);
	if (s->testSetX ) release_memory( (void **)&s->testSetX );
	if (s->testSetY ) release_memory( (void **)&s->testSetY );

	if (t->trainSetX) cuda_free ( t->trainSetX, ERROR_MEM_CLEANUP );
	if (t->trainSetY ) cuda_free ( t->trainSetY , ERROR_MEM_CLEANUP );
	if (t->testSetX ) cuda_free ( t->testSetX, ERROR_MEM_CLEANUP );
	if (t->testSetY ) cuda_free ( t->testSetY , ERROR_MEM_CLEANUP );
}
