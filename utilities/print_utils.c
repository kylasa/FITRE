#include <utilities/print_utils.h>
#include <device/cuda_utils.h>
#include <core/errors.h>


#include <string.h>

real computeWeightSum( real *src, int len, real *dscratch){
	real *t = (real *)dscratch;
	copy_host_device( t, src, len * sizeof(real), cudaMemcpyDeviceToHost, ERROR_DEBUG);	
	real s = 0; 
	
	for (int i=0 ; i < len; i ++) s += t[i];
	return s; 
}

void printVector( real *src, int c, real *r, real *dscratch){
	real *t = (real *)dscratch;
	int count = c;// > 20 ? 20 : c;
	copy_host_device( t, src, c * sizeof(real), cudaMemcpyDeviceToHost, ERROR_DEBUG);	

	for (int i = 0; i < count; i ++){
		if ((i % 20 == 0) && (i != 0)) fprintf (stderr, "\n");
		fprintf( stderr, " %e ", t[i] );
	}
	fprintf (stderr, "\n");
}

void printCustomVector( real *src, int c, int jump, real *dscratch){
	real *t = (real *)dscratch;
	int count = c;
	copy_host_device( t, src, c * sizeof(real), cudaMemcpyDeviceToHost, ERROR_DEBUG);	

	for (int i = 0; i < count; i += jump){
		fprintf( stderr, " %f ", t[i] );
	}
	fprintf (stderr, "\n");
}

void printIntVector( int *src, int c, int *r, real *dscratch){
	//int *t = (int *)dscratch;
	int count = c;// > 20 ? 20 : c;
	//copy_host_device( t, src, c * sizeof(int), cudaMemcpyDeviceToHost, ERROR_DEBUG);	

	for (int i = 0; i < count; i ++){
		if ((i % 20 == 0) && (i != 0)) fprintf (stderr, "\n");
		fprintf( stderr, " %d ", src[i] );
	}
	fprintf (stderr, "\n");
}

void printHostVector( real *src, int c , real *dscratch){
	real *t = src;
	int count = c;// > 20 ? 20 : c;

	for (int i = 0; i < count; i ++){
		if ((i % 20 == 0) && (i != 0)) fprintf (stderr, "\n");
		fprintf( stderr, " %e ", t[i] );
	}
	fprintf (stderr, "\n");
}

void writeMatrix ( real *mat, int rows, real *dscratch )
{
        FILE *dataset_file;
	real *t = (real *) dscratch;

        if ( (dataset_file = fopen("./hessian.txt", "w")) == NULL ) {
                fprintf( stderr, "Error opening the hessian.... !\n" );
                exit( -1 );
        }

	fprintf (stderr, "Copying data to host \n");
	copy_host_device( t, mat, rows * rows * sizeof(real), cudaMemcpyDeviceToHost, ERROR_DEBUG);	
	fprintf (stderr, "Done Copying data to host \n");

        for (int i = 0; i < rows; i ++){
                fprintf (dataset_file, "%6.2f", t[ i * rows ] );
                for (int j = 1; j < rows; j ++){
                        fprintf( dataset_file, ",%6.2f", t[ i * rows + j ] );
                }
                fprintf( dataset_file, "\n");
        }
        fclose (dataset_file);
}

void writeSparseMatrix (real *dataPtr, int *rowIndex, int *colIndex, int m, int n, int nnz, real *dscratch )
{
        FILE *dataset_file;
	int *t = (int *) dscratch;
	real *t1 = (real *) dscratch;

        if ( (dataset_file = fopen("./rowindex.txt", "w")) == NULL ) {
                fprintf( stderr, "Error opening the hessian.... !\n" );
                exit( -1 );
        }

	fprintf (stderr, "Copying data to host \n");
	copy_host_device( t, rowIndex, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost, ERROR_DEBUG);	
	fprintf (stderr, "Done Copying data to host \n");
	
	for (int i = 0; i < m + 1; i ++){
        	fprintf( dataset_file, "%d\n", t[ i ] );
	}
	fclose (dataset_file);
	
        if ( (dataset_file = fopen("./colindex.txt", "w")) == NULL ) {
                fprintf( stderr, "Error opening the hessian.... !\n" );
                exit( -1 );
        }

	fprintf (stderr, "Copying data to host \n");
	copy_host_device( t, colIndex, sizeof(int) * (nnz), cudaMemcpyDeviceToHost, ERROR_DEBUG);	
	fprintf (stderr, "Done Copying data to host \n");

	for (int i = 0; i < nnz; i ++){
        	fprintf( dataset_file, "%d\n", t[ i ] );
	}
	fclose (dataset_file);

        if ( (dataset_file = fopen("./data.txt", "w")) == NULL ) {
                fprintf( stderr, "Error opening the hessian.... !\n" );
                exit( -1 );
        }

	fprintf (stderr, "Copying data to host \n");
	copy_host_device( t1, dataPtr, sizeof(real) * (nnz), cudaMemcpyDeviceToHost, ERROR_DEBUG);	
	fprintf (stderr, "Done Copying data to host \n");

	for (int i = 0; i < nnz; i ++){
        	fprintf( dataset_file, "%6.10f\n", t1[ i ] );
	}
	fclose (dataset_file);
}

void writeVector ( real *mat, int rows, char *file, int hostData, real *dscratch )
{
        FILE *dataset_file;
	real *t = (real *) dscratch;

        if ( (dataset_file = fopen( file, "w")) == NULL ) {
                fprintf( stderr, "Error opening the path .... !\n" );
                exit( -1 );
        }

	if (hostData == 1) {
		t = mat;
	} else {
		fprintf (stderr, "Copying data to host \n");
		copy_host_device( t, mat, rows * sizeof(real), cudaMemcpyDeviceToHost, ERROR_DEBUG);	
		fprintf	(stderr, "Done Copying data to host \n");
	}

        for (int i = 0; i < rows; i ++){
        	fprintf( dataset_file, "%6.10f\n", t[ i ] );
        }
        //fprintf( dataset_file, "\n");
        fclose (dataset_file);
}

int readVector( real *vec, int rows, char *file, int offset , real *dscratch){
	FILE *handle; 
	char line[1024];
	int index = 0; 
	char *word;
	
	if ( (handle= fopen( file, "r" )) == NULL ) {
		fprintf( stderr, "Error opening the path... \n");
		exit(-1); 
	}

	index = 0; 
        while (!feof( handle )){
                memset( line, 0, 1024);
                fgets( line, 1024, handle);
                if (line[0] == 0) break;
	
		word = strtok( line, "\n"); 
		vec[ index ++ ] = atof( word ) + offset; 

		if (index >= rows) break;
        }
        fclose( handle );	

	return index;
}

void writeIntVector ( int *mat, int rows, real *dscratch )
{
        FILE *dataset_file;
	int *t = (int *) dscratch;

        if ( (dataset_file = fopen( "./vector.txt", "w")) == NULL ) {
                fprintf( stderr, "Error opening the path .... !\n" );
                exit( -1 );
        }

	fprintf (stderr, "Copying data to host \n");
	copy_host_device( t, mat, rows * sizeof(int), cudaMemcpyDeviceToHost, ERROR_DEBUG);	
	fprintf (stderr, "Done Copying data to host \n");

        for (int i = 0; i < rows; i ++){
        	fprintf( dataset_file, "%d\n", t[ i ] );
        }
        //fprintf( dataset_file, "\n");
        fclose (dataset_file);
}

void writeIntMatrix ( real *mat, int images, int features )
{
        FILE *dataset_file;

        if ( (dataset_file = fopen( "./augmented_images.txt", "w")) == NULL ) {
                fprintf( stderr, "Error opening the path .... !\n" );
                exit( -1 );
        }

			features = 1024; 
        for (int i = 0; i < images; i ++){
			for (int j = 0; j < features; j ++) fprintf( dataset_file, "%d,", (int)mat[ i * features + j ] );
			for (int j = 0; j < features; j ++) fprintf( dataset_file, "%d,", (int)mat[ 1 * 500 * 32 * 32 + i * features + j ] );
			for (int j = 0; j < features; j ++) fprintf( dataset_file, "%d,", (int)mat[ 2 * 500 * 32 * 32 + i * features + j ] );
			
			fprintf( dataset_file, "\n"); 
        }
        //fprintf( dataset_file, "\n");
        fclose (dataset_file);
}


void print2DMatrix( real *src, int h, int w)
{
	int height = h; 
	int width = w; 

	//if (h > 4) height = 4; 
	//if (w > 4) width = 4; 

	for (int i = 0; i < height; i ++) {
		for (int j = 0; j < width; j ++) {
			fprintf( stderr, "%.10f ", src[ j * h + i ] ); 
		}
		fprintf( stderr, "\n"); 
	}	
}

void print2DMatrix( real *src, int h, int w, int ch, int img, int channels, int images)
{
	int height = h; 
	int width = w; 

	//if (h > 4) height = 4; 
	//if (w > 4) width = 4; 

	for (int i = 0; i < height; i ++) {
		for (int j = 0; j < width; j ++) {
			fprintf( stderr, "%.10f ", src[ ch * h * w * images + img * h * w  +  j * h + i ] ); 
		}
		fprintf( stderr, "\n"); 
	}	
}

void print3DMatrix( real *src, int c, int h, int w, int img, int images )
{
	int high = c;
	//int high = (c > 4) ? 4 : c; 

	for (int ch = 0; ch < high; ch ++) {
		fprintf( stderr, "[ %d, :, :] --> \n", ch ); 
		print2DMatrix( src, h, w, ch, img, high, images ); 
	}
}

void print4DMatrix( real *src, int n, int c, int h, int w )
{
	int high = n;
	//int high = n > 4 ? 4 : n; 
	for (int i = 0; i < high; i ++) {
		fprintf( stderr, "[ %d, :, :, : ]\n", i ); 
		print3DMatrix( src , c, h, w, i, n ); 
	}
}
