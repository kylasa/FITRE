#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <core/datadefs.h>
#include <core/errors.h>

#include <utilities/utils.h>
#include <utilities/print_utils.h>
#include <utilities/reduce_helper.h>

#include <device/device_defines.h>
#include <device/cuda_utils.h>
#include <device/handles.h>
#include <device/reduce.h>

#include <functions/eval_gradient.h>
#include <functions/dev_activations.h>
#include <functions/dev_loglikelihood.h>
#include <functions/dev_layer_error.h>
#include <functions/dev_mse_error.h>
#include <functions/dev_mat_vec_addition.h>
#include <functions/dev_initializations.h>
#include <functions/swish.h>

/*
	R{ f( wz + b ) }
		 = f'(wz + b) R{ wz + b }
		 = f'(wz + b) { R{ w } z + w R{ z } + vb }
		 = f'(wz + b) { vw * z + w R{ z } + vb }
*/

void applyROpLayerActivation (int actFunction, 
			real *W, int wRows, int wCols,  
			real *b, int bRows, 
			real *z, int zRows, int zCols, 
			real *z1, int z1Rows, int z1Cols, 
			real *VW, real * Vb, real *rz,
			real *output, int offset, real *devPtr, real *hostPtr)
{

	int matElements;
	int numBlocks;
	int wBlocks;

	real *vec1 = NULL;
	real *colSums = NULL;

	real alpha = 1.0, beta = 0; 

	//compute VW * Z
	cublasCheckError (
		cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
						wRows, zCols, wCols,
						&alpha, VW, wRows, 
						z, zRows, &beta, output, wRows ) ); 

	//output += W * Rz
	cublasCheckError( 
			cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
						wRows, zCols, wCols, 
						&alpha, W, wRows, 
						rz, zRows, &beta, devPtr, wRows ) ); 

	cublasCheckError( 
			cublasDaxpy( cublasHandle, wRows * zCols,
								&alpha, devPtr, 1, output, 1 ) ); 

	matElements = wRows * zCols; 
	numBlocks = matElements / BLOCK_SIZE + 
					((matElements % BLOCK_SIZE == 0) ? 0 : 1);

	//add + Vb
	if (Vb != NULL ) {
		kerUtilsAddColumnToMatrix <<<numBlocks ,BLOCK_SIZE>>> 
				( output, wRows, zCols, Vb ); 
		cudaThreadSynchronize (); 	
		cudaCheckError (); 
	}

#ifdef DEBUG_DETAILED
fprintf( stderr, ".... Done with ROp Forward R{ wz + b }...... \n\n"); 
copy_host_device( hostPtr, output, sizeof(real) * wRows * zCols, 
	cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
print2DMatrix( hostPtr, wRows, zCols ); 
#endif

//TODO -- temp storeage for Wz + b result
#ifdef DEBUG_CNN
fprintf( stderr, ".... Copying the temp resutl (Wz + b) to extended Z \n\n" ); 
#endif
copy_device( output + offset, output, sizeof(real) * wRows * zCols , ERROR_MEMCPY_DEVICE_DEVICE ); 
//TODO -- temp storeage for Wz + b result

	switch( actFunction ){

		// SK-2 Fixed the dimensions here. 
		case CNN_ACT_SOFTPLUS: 
			kerNNBackPropSOFTPLUS<<< numBlocks, BLOCK_SIZE >>> 
				(output, z1 + offset, z1Rows * z1Cols ); 
			cudaThreadSynchronize (); 
			cudaCheckError ();
			break;

		// This computes the RZ in the forward pass... 
		case CNN_ACT_SWISH: 
			//TODO -- zin and zout are not present
			//kerNNBackPropSwish <<< numBlocks, BLOCK_SIZE >>> 
				//( z - (wRows * zCols), z, output, z1Rows * z1Cols ); 
				//( z + wRows * zCols, z, output, z1Rows * z1Cols ); 

			kerNNBackPropSwish <<< numBlocks, BLOCK_SIZE >>> 
				( z1 + offset, z1, output, z1Rows * z1Cols ); 

			/*
			kerNNROpSwish <<< numBlocks, BLOCK_SIZE >>> 
				( z1 + offset, output, z1Rows * z1Cols ); 
			*/
			cudaThreadSynchronize (); 
			cudaCheckError ();
				
			break;

		case CNN_ACT_NONE: 
			break;
		

		default: 
			fprintf (stderr, "ROp_FC_LAYER: Error Unknown Activation Function \n"); 
			exit (-1); 
	}
}
