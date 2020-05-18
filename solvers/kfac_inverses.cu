
#include <solvers/kfac_inverses.h>
#include <solvers/kfac_utils.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>

#include <core/errors.h>

#include <functions/dev_image.h>
#include <functions/dev_initializations.h>
#include <functions/dev_backprop_convolution.h>

#include <utilities/print_utils.h>
#include <utilities/utils.h>

/*
https://stackoverflow.com/questions/28794010/solving-dense-linear-systems-ax-b-with-cuda
https://devtalk.nvidia.com/default/topic/767806/gpu-accelerated-libraries/matrix-inversion-with-cublassgetri/
https://stackoverflow.com/questions/37731103/cublas-matrix-inverse-much-slower-than-matlab
*/

GLOBAL void ker_pivot_mat( int *indices, real *mat, int n )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (idx < n){
		mat[ idx * n + indices[ idx ] ] = 1.; 
	}
}

GLOBAL void ker_identity( real* mat, int n )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < n){
		mat[ idx + n * idx ] = 1.; 
	}
}

GLOBAL void ker_init_last_row( real *mat, int rows, int cols, real value )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < cols) {
		mat[ idx * rows + rows - 1 ] = value; 
	}
}

GLOBAL void ker_add_regularization( real *mat, int rows, real scale )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if ( idx < rows ) {
		mat[ idx + idx * rows ] += scale; 
	}
}

real group_products( real *mat, int numElements )
{
	real sum = 0; 

	cublasCheckError( cublasDdot( cublasHandle, numElements, mat, 1, mat, 1, &sum )); 

	return sqrt( sum ); 
}

void initKFACData( CNN_MODEL *model, KFAC_CURVATURE_INFO *kfac_info )
{
	int *omegaOffsets = kfac_info->OmegaZOffsets; 
	int *lambdaOffsets = kfac_info->LambdaGOffsets; 

	int zRows, zCols; 

	for (int l = 0; l < model->cLayers; l ++){
		CONV_LAYER *convLayer = &model->convLayer[ l ]; 

		zRows = model->batchSize * convLayer->outHeight * convLayer->outWidth; 
		zCols = convLayer->inChannels * convLayer->kSize * convLayer->kSize ; 
		if (model->bias != 0) zCols += 1;

		cuda_memset( kfac_info->OmegaZZT + omegaOffsets[ l ], 0, sizeof(real) * zCols * zCols, 
							ERROR_MEMSET ); 

		cuda_memset( kfac_info->LambdaGGT + lambdaOffsets[ l ], 0, 
					sizeof(real) * convLayer->outChannels * convLayer->outChannels, ERROR_MEMSET ); 
	}

	for (int l = 0; l < model->lLayers; l ++) {
		FC_LAYER *ll = &model->fcLayer[ l ]; 

		if (model->bias != 0) 
			cuda_memset( kfac_info->OmegaZZT + omegaOffsets[ model->cLayers + l ], 0, 
				sizeof(real) * (ll->in + 1) * (ll->in + 1), ERROR_MEMSET ); 
		else
			cuda_memset( kfac_info->OmegaZZT + omegaOffsets[ model->cLayers + l ], 0, 
				sizeof(real) * ll->in  * ll->in, ERROR_MEMSET ); 

		cuda_memset( kfac_info->LambdaGGT + lambdaOffsets[ model->cLayers + l ], 0, 
				sizeof(real) * ll->out * ll->out, ERROR_MEMSET ); 
	}
}


void updateRunningStats( real *src, real *update, int numElements, real momentum )
{

	real alpha; 

	//	theta *= momentum / (1 - momentum )
	// theta += udpate
	// theta *= 1 - momentum
	if (momentum != 0) {
		alpha = momentum / (1. - momentum ); 
		cublasCheckError( cublasDscal( cublasHandle, numElements, &alpha, src, 1 ) ); 

		alpha = 1.; 
		cublasCheckError( cublasDaxpy( cublasHandle, numElements, &alpha, update, 1, src, 1 ) ); 

		alpha = 1. - momentum; 
		cublasCheckError( cublasDscal( cublasHandle, numElements, &alpha, src, 1 ) ); 
	} else {
		copy_device( src, update, sizeof(real) * numElements, ERROR_MEMCPY_DEVICE_DEVICE ); 
	}
}

/*

Here Damping and Regularization is added to the ZZT before we take inverse. 

*/
void computeMatrixInverseCusolver (real *matrix, real *inverse, int n, 
	real *devPtr, real *pageLckPtr, 
	real lambda, real dampGamma, cudaStream_t curStream, cusolverDnHandle_t curHandle  )
{
	int *devInfo = (int *)pageLckPtr; 
	int cusolverWorkspaceSize;
	int blocks;

	//create Identity on the rhs. 
	cuda_memset( inverse, 0, sizeof(real) * n * n, ERROR_MEMSET ); 

	//Create Identity matrix here. for the RHS
	blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	ker_identity <<< blocks, BLOCK_SIZE >>> 
		(inverse, n); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 
	

	copy_device( devPtr, matrix, sizeof(real) * n * n, ERROR_MEMCPY_DEVICE_DEVICE ); 

	//Add the Damping Term here along with Regularization term here. 
	blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	ker_add_regularization <<< blocks, BLOCK_SIZE >>> 
		( devPtr, n, lambda + dampGamma); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 

	//buffer size
	cusolverWorkspaceSize = 0; 
	cusolverCheckError( cusolverDnDpotrf_bufferSize( curHandle, CUBLAS_FILL_MODE_LOWER, n, devPtr, n, &cusolverWorkspaceSize ) ); 
	cudaDeviceSynchronize (); 

	//fprintf( stderr, "LL factorization.... %zu \n", cusolverWorkspaceSize ); 

	// LL' factorization. 
	*devInfo = -1; 
	cusolverCheckError( cusolverDnDpotrf( curHandle, CUBLAS_FILL_MODE_LOWER, n, devPtr, n, devPtr + n*n, cusolverWorkspaceSize, devInfo )); 
	cudaDeviceSynchronize (); 

	if( *devInfo != 0) {
		fprintf( stderr, "We have a problem with the LU factorization ... %d\n", *devInfo); 
		exit ( -1 ); 
	}
	// Inversion.
	*devInfo = -1; 
	cusolverCheckError( cusolverDnDpotrs( curHandle, CUBLAS_FILL_MODE_LOWER, n, n, devPtr, n, inverse, n, devInfo ) ); 
	cudaDeviceSynchronize (); 

	if( *devInfo != 0) {
		fprintf( stderr, "We have a problem with the AX = B solver... %d\n", *devInfo); 
		exit ( -1 ); 
	}
}

/*
	Solving the PLU X = I system, 
	
*/
void computeMatrixInverseTR (real *matrix, real *inverse, int n, 
	real *devPtr, real *hostPtr, real *pageLckPtr, 
	real lambda, real dampGamma )
{
	real *ZPtr = devPtr;
	real *pivotMat = ZPtr + n * n; 
	real *ZInvPtr = inverse; 

	int *pivotArray = (int *)(pivotMat + n*n); 
	int *nextDevPtr = pivotArray + n + (n % 32); 

	int *infoArray = (int *)pageLckPtr; 

	real **ZArr = (real **) nextDevPtr;
	real **ZInvArr = ZArr + 4; 

	real scale = 0; 

	int blocks; 
	real alpha = 0; 

	copy_device( devPtr, matrix, sizeof(real) * n * n, ERROR_MEMCPY_DEVICE_DEVICE ); 

	//Add the Damping Term here along with Regularization term here. 
	scale = lambda + dampGamma; 

	blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	ker_add_regularization <<< blocks, BLOCK_SIZE >>> 
		( devPtr, n, scale ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

	//In place LU factorization here. 
	infoArray[ 0 ] = 0; 
	copy_host_device( &ZPtr, ZArr, sizeof(real *), cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
	cublasCheckError( cublasDgetrfBatched( cublasHandle, n, ZArr, n, pivotArray, infoArray, 1 ) ); 

	if (infoArray[ 0 ] != 0) {
		fprintf( stderr, "ComputeMatrixInverse: Problem with LU Decomposition .... %d \n", infoArray[ 0 ]); 
		exit ( -1 ); 
	}

	// Perform the Linear System solver here. 
	// P LU X = I, X is the inverse we are looking for. 
	// LU X = P', since P is orthonormal matrix. 
	// L A = P'
	// U X = A. 

	// For the P' Matrix here. 
	cuda_memset( pivotMat, 0, sizeof(real) * n * n, ERROR_MEMSET ); 

	blocks = ( n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	ker_pivot_mat <<< blocks, BLOCK_SIZE >>> 
		( pivotArray, pivotMat, n ); 
	cudaDeviceSynchronize (); 
	cudaCheckError (); 

	// L A = P'
	alpha = 1; 
	cublasCheckError( cublasDtrsm( cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, 
												CUBLAS_DIAG_UNIT, n, n, &alpha, ZPtr, n, pivotMat, n ) ); 

	// U X = A
	cublasCheckError( cublasDtrsm( cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 
												CUBLAS_DIAG_NON_UNIT, n, n, &alpha, ZPtr, n, pivotMat, n ) ); 

	//copy the result back to the inverse array. 
	copy_device( inverse, pivotMat, sizeof(real) * n * n, ERROR_MEMCPY_DEVICE_DEVICE ); 
}



void computeMatrixInverse (real *matrix, real *inverse, int n, real *devPtr, real *hostPtr, real *pageLckPtr, real lambda, real dampGamma )
{
	//real *ZPtr = matrix;
	real *ZPtr = devPtr;
	real *ZInvPtr = inverse; 

	int *infoArray = (int *)pageLckPtr; 
	int *pivotArray = (int *)(devPtr + n*n); 
	int *nextDevPtr = pivotArray + n + (n % 32); 

	real **ZArr = (real **) nextDevPtr;
	real **ZInvArr = ZArr + 4; 

	real scale = 0; 
	int blocks; 

	copy_device( devPtr, matrix, sizeof(real) * n * n, ERROR_MEMCPY_DEVICE_DEVICE ); 

	//Add the Damping Term here along with Regularization term here. 
	scale = lambda + dampGamma; 

	blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	ker_add_regularization <<< blocks, BLOCK_SIZE >>> 
		( devPtr, n, scale ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 

/*
copy_host_device( hostPtr, devPtr, sizeof(real) * 10, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST ); 
for (int i = 0; i < 10; i ++ )
	fprintf( stderr, " %e ", hostPtr[ i ] ); 
fprintf( stderr, "\n"); 
*/

/*
real temp = 0, sum = 0; 
cublasCheckError( cublasDnrm2( cublasHandle, n * n, devPtr, 1, &temp ) ); 
cublasCheckError( cublasDasum( cublasHandle, n * n, devPtr, 1, &sum )); 
fprintf( stderr, "Dnrm2( ., 2) --> %e, sum: %e \n", temp, sum ) ;
*/



	//In place LU factorization here. 
	infoArray[ 0 ] = 0; 
	copy_host_device( &ZPtr, ZArr, sizeof(real *), cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
	cublasCheckError( cublasDgetrfBatched( cublasHandle, n, ZArr, n, pivotArray, infoArray, 1 ) ); 

	if (infoArray[ 0 ] != 0) {
		fprintf( stderr, "ComputeMatrixInverse: Problem with LU Decomposition .... %d \n", infoArray[ 0 ]); 
		exit ( -1 ); 
	}

	// out of place inverseion here. 
	copy_host_device( &ZInvPtr, ZInvArr, sizeof(real *), cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 
	infoArray[ 0 ] = 0; 
	cublasCheckError( cublasDgetriBatched( cublasHandle, n, (const real **)ZArr, n, pivotArray, ZInvArr, n, infoArray, 1 ) ); 

	if (infoArray[ 0 ] != 0) {
		fprintf( stderr, "ComputeMatrixInverse: Problem with Matrix Inversion.... %d\n", infoArray[ 0 ]); 
		exit ( -1 ); 
	}
}

void printNorms( CNN_MODEL *model, KFAC_CURVATURE_INFO *kfac_info )
{
	int *omegaOffsets = kfac_info->OmegaZOffsets; 
	int *lambdaOffsets = kfac_info->LambdaGOffsets; 

	real pageLckPtr = 0; 
	int zRows, zCols; 

	for (int l = 0; l < model->cLayers; l ++){
		CONV_LAYER *convLayer = &model->convLayer[ l ]; 

		zRows = model->batchSize * convLayer->outHeight * convLayer->outWidth; 
		zCols = convLayer->inChannels * convLayer->kSize * convLayer->kSize ;
		if (model->bias  != 0) zCols += 1; 

		cublasCheckError( cublasDnrm2( cublasHandle, zCols * zCols, 
									kfac_info->OmegaZZT + omegaOffsets[ l ], 1, &pageLckPtr )); 
		fprintf( stderr, "OmegaZ Layer: %d, Norm( ZZT, 2 ): %e\n", l, pageLckPtr); 

		cublasCheckError( cublasDnrm2( cublasHandle, zCols * zCols, 
									kfac_info->OmegaZInv + omegaOffsets[ l ], 1, &pageLckPtr )); 
		fprintf( stderr, "OmegaZ Layer: %d, Norm( inverse ZZT, 2 ): %e\n", l, pageLckPtr); 

		cublasCheckError( cublasDnrm2( cublasHandle, convLayer->outChannels * convLayer->outChannels, 
							kfac_info->LambdaGGT + lambdaOffsets[ l ], 1, &pageLckPtr )); 
		fprintf( stderr, "LambdaDelta Layer: %d, LambdaG Layer. Norm( ., 2 ): %f \n", l, pageLckPtr ); 

		cublasCheckError( cublasDnrm2( cublasHandle, convLayer->outChannels * convLayer->outChannels, 
							kfac_info->LambdaGInv + lambdaOffsets[ l ], 1, &pageLckPtr )); 
		fprintf( stderr, "LambdaDelta Layer: %d, LambdaG Layer inverses. Norm( ., 2 ): %f \n", l, pageLckPtr ); 
	}

	for (int l = 0; l < model->lLayers; l ++) {
		FC_LAYER *ll = &model->fcLayer[ l ]; 

		pageLckPtr = 0; 
		cublasCheckError( cublasDnrm2( cublasHandle, 
									(ll->in + ((model->bias != 0) ? 1 : 0)) * (ll->in +((model->bias != 0) ? 1 : 0)), 
									kfac_info->OmegaZZT + omegaOffsets[ model->cLayers + l ], 1, &pageLckPtr )); 
		fprintf( stderr, "OmegaZ Layer: %d, Norm( ZZT, 2 ): %f\n", l, pageLckPtr); 

		cublasCheckError( cublasDnrm2( cublasHandle, 
									(ll->in + ((model->bias != 0) ? 1 : 0)) * (ll->in +((model->bias != 0) ? 1 : 0)), 
									kfac_info->OmegaZInv + omegaOffsets[ model->cLayers + l ], 1, &pageLckPtr )); 
		fprintf( stderr, "OmegaZ Layer: %d, Norm( inverse ZZT, 2 ): %f\n", l, pageLckPtr); 

		pageLckPtr = 0; 
		cublasCheckError( cublasDnrm2( cublasHandle, ll->out * ll->out, 
							kfac_info->LambdaGGT + lambdaOffsets[ model->cLayers + l ], 1, &pageLckPtr )); 
		fprintf( stderr, "LambdaDelta Layer: %d, Done with Norm( ., 2 ): %f \n", l, pageLckPtr ); 

		cublasCheckError( cublasDnrm2( cublasHandle, ll->out * ll->out, 
							kfac_info->LambdaGInv + lambdaOffsets[ model->cLayers + l ], 1, &pageLckPtr )); 
		fprintf( stderr, "LambdaDelta Layer: %d, Done with inverses. Norm( ., 2 ): %f \n", l, pageLckPtr ); 
	}
}


void computeSpeedUpOmegaZ( KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model, DEVICE_DATASET *data,
   int samples, real *z, int *zOffsets,
   real *dampedInput, real *dampedZ, int *zztOffsets, int iterCount)
{  
   if (iterCount == 0) { 
      copy_device( dampedInput, data->currentBatch, sizeof(real) * data->features * samples,
                     ERROR_MEMCPY_DEVICE_DEVICE );
      
      for (int i = 0; i < model->cLayers; i ++ ) {  
         CONV_LAYER *convLayer = &model->convLayer[ i ];         
			POOL_LAYER *poolLayer = &model->poolLayer[ i ];
         
         if (poolLayer->type == NO_POOL) { 
            copy_device( dampedZ + zztOffsets[ i + 1 ], z + zOffsets[ i + 1 ] + convLayer->outputOffset,     
               sizeof(real) * convLayer->outHeight * convLayer->outWidth * convLayer->outChannels * samples, 
               ERROR_MEMCPY_DEVICE_DEVICE );
         }  else { 
            copy_device( dampedZ + zztOffsets[ i + 1 ], z + zOffsets[ i + 1 ] + convLayer->outputOffset,     
               sizeof(real) * poolLayer->outHeight * poolLayer->outWidth * convLayer->outChannels * samples, 
               ERROR_MEMCPY_DEVICE_DEVICE );
         }
      }
      
      for (int l = 0; l < model->lLayers; l ++) {
         FC_LAYER *ll = &model->fcLayer[ l ];
         
         copy_device( dampedZ + zztOffsets[ l + 1 + model->cLayers ],
                        z + zOffsets[ l + 1 + model->cLayers ], 
                        sizeof(real) * ll->out * samples, ERROR_MEMCPY_DEVICE_DEVICE );
      }
   } else { 
      //All the succeeding mini-batches after the very first batch... 
      // Here we do the updateRunningStats on each of the Z's 
      updateRunningStats( dampedInput, data->currentBatch, data->features * samples,
                              sqrt( kfac_info->stats_decay ) );
      
      for (int i = 0; i < model->cLayers; i ++ ) {  
         CONV_LAYER *convLayer = &model->convLayer[ i ];
         POOL_LAYER *poolLayer = &model->poolLayer[ i ];
         
         if (poolLayer->type == NO_POOL) { 
            updateRunningStats( dampedZ + zztOffsets[ i + 1 ], 
                                 z + zOffsets[ i + 1 ] + convLayer->outputOffset, 
                                 convLayer->outHeight * convLayer->outWidth * convLayer->outChannels * samples,                
                                 sqrt( kfac_info->stats_decay ));
         }  else { 
            updateRunningStats( dampedZ + zztOffsets[ i + 1 ], 
                                 z + zOffsets[ i + 1 ] + convLayer->outputOffset, 
                                 poolLayer->outHeight * poolLayer->outWidth * convLayer->outChannels * samples,                
                                 sqrt( kfac_info->stats_decay ));
         }
      }
      
      for (int l = 0; l < model->lLayers; l ++) {
         FC_LAYER *ll = &model->fcLayer[ l ];
         
         updateRunningStats( dampedZ + zztOffsets[ l + 1 + model->cLayers ],
                              z + zOffsets[ l + 1 + model->cLayers ], 
                              ll->out * samples, sqrt( kfac_info->stats_decay ));
      }
   }
}


void computeOmegaZ( KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model, 
	DEVICE_DATASET *data, int samples, real *dampedInput, real *dampedZ, real *omegaZInv, 
	int *zOffsets, int *zztOffsets, real *devPtr, real *pageLckPtr, 
	cublasHandle_t blasHandle, cudaStream_t curStream, cusolverDnHandle_t curHandle)
{

	real *dataset = NULL; 

	real *expandedMatrix; 
	real *zMatrix; 
	real *scaledZ; 
	real alpha, beta; 
	const real **matPtr; 
	real **invMatPtr; 
	real *tmp; 

	real *nextDevPtr;
	real *zzt; 

	int inputOffset; 

	int zRows, zCols; 
	int blocks; 
	int *omegaOffsets = kfac_info->OmegaZOffsets; 

	/*
		Convolution Layers **
		n * h * w X inC

		Expanded form as 
		n * h * w X inC * k * k

		result of Z^T Z --> inC * k * k X inC * k * k
	*/
	for (int l = 0; l < model->cLayers; l ++){

		CONV_LAYER *convLayer = &model->convLayer[ l ]; 
		POOL_LAYER *poolLayer = &model->poolLayer[ l ]; 

      if (dampedInput != NULL) {
         if (l == 0) dataset = dampedInput;
         else dataset = dampedZ + zztOffsets[ l ];
      }  else {
         if (l == 0) dataset = data->currentBatch;
         else {
            CONV_LAYER *prevLayer = &model->convLayer[ l - 1 ];
            dataset = dampedZ + zOffsets[ l ] + prevLayer->outputOffset;
         }
      }

		zRows = samples * convLayer->outHeight * convLayer->outWidth; 
		zCols = convLayer->inChannels * convLayer->kSize * convLayer->kSize ;
		if (model->bias != 0)
			zCols += 1;

		zzt = devPtr; 
		expandedMatrix = zzt + zCols * zCols; 
		nextDevPtr = expandedMatrix + zRows * zCols;

		//Expanded Form here. 
		getBatchImageCols( dataset, samples, convLayer->inChannels, 
			convLayer->height, convLayer->width, convLayer->kSize, convLayer->padding, 
			convLayer->stride, expandedMatrix); 

		if (model->bias == 0) { 
			alpha = 1./(real)(convLayer->outHeight * convLayer->outWidth); 
			cublasCheckError( cublasDscal( blasHandle, zRows * zCols, &alpha, expandedMatrix, 1 ) ); 
		} else { 
			alpha = 1./(real)(convLayer->outHeight * convLayer->outWidth); 
			cublasCheckError( cublasDscal( blasHandle, zRows * (zCols-1), &alpha, expandedMatrix, 1 ) ); 

			blocks  = ( zRows  + BLOCK_SIZE - 1) / BLOCK_SIZE; 
			kerInitOneVector <<< blocks, BLOCK_SIZE >>>
				(expandedMatrix + zRows * (zCols - 1), zRows); 
			cudaThreadSynchronize (); 
			cudaCheckError (); 
		}

		//compute Z^T * Z
		// (inC * k * k + 1 X n * h * w) X (n * h * w X inC * k * k + 1) 
		alpha = 1.; beta = 0; 
		cublasCheckError( cublasDgemm( blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
					zCols, // m lda( Z^T )	
					zCols, //n columns of Z^T, rows of Z
					zRows, // k
					&alpha, expandedMatrix, zRows, expandedMatrix, zRows,
					&beta, zzt, zCols) ); 

		//Scale  
		alpha = 1./(real)samples;
		cublasCheckError( cublasDscal( blasHandle, zCols * zCols, &alpha, zzt, 1 ) ); 

		computeMatrixInverseCusolver( zzt,  omegaZInv + omegaOffsets[ l ], 
										zCols, nextDevPtr, pageLckPtr, 
									kfac_info->regLambda, kfac_info->dampGamma, curStream , curHandle ) ; 
	}

	//Inverses for the Linear Layers here. 
	for (int l = 0; l < model->lLayers; l ++) {

		FC_LAYER *ll = &model->fcLayer[ l ]; 
		if (l == 0) {
         if (dampedInput == NULL) {
            CONV_LAYER *c =  &( model->convLayer[ model->cLayers - 1 ] );
            POOL_LAYER *p = &( model->poolLayer[ model->cLayers - 1 ] );

            dataset = dampedZ + zOffsets[ model->cLayers + l ] + c->outputOffset;
         } else {
            dataset = dampedZ + zztOffsets[ model->cLayers + l ];
         }

			zMatrix = devPtr; 
			if (model->bias != 0) {
				zRows = ll->in + 1;
				scaledZ = zMatrix + zRows * samples;
				nextDevPtr = scaledZ + zRows * samples; 
			} else {
				zRows = ll->in; 
				scaledZ = zMatrix + zRows * samples;
				nextDevPtr = scaledZ + zRows * samples; 
			}

		} else {

         if (dampedInput == NULL)
            dataset = dampedZ + zOffsets[ model->cLayers + l ] ;
         else
            dataset = dampedZ + zztOffsets[ model->cLayers + l ] ;

			zMatrix = devPtr; 
			if (model->bias != 0) {
				zRows = ll->in + 1; 
				scaledZ = zMatrix + zRows * samples;
				nextDevPtr = scaledZ + zRows * samples; 
			} else {
				zRows = ll->in; 
				scaledZ = zMatrix + zRows * samples;
				nextDevPtr = scaledZ + zRows * samples; 
			}
		}

		//compute Z^T * Z
		if (model->bias != 0) {
			cudaMemcpy2D( 	zMatrix, sizeof(real) * zRows, 
							dataset , sizeof(real) * ll->in, 
							sizeof(real) * ll->in , sizeof(real) * samples, 
							cudaMemcpyDeviceToDevice ) ;
			cudaCheckError (); 

			blocks = (samples + BLOCK_SIZE - 1) / BLOCK_SIZE; 
			ker_init_last_row <<< blocks, BLOCK_SIZE >>> 
				( zMatrix, zRows, samples, 1 ); 
			cudaThreadSynchronize (); 
			cudaCheckError (); 

		} else { 
			copy_device( zMatrix, dataset, sizeof(real) * zRows * samples, 	
								ERROR_MEMCPY_DEVICE_DEVICE ); 
		}

		copy_device( scaledZ, zMatrix, sizeof(real) * zRows * samples, ERROR_MEMCPY_DEVICE_DEVICE ); 

		//Scale
		alpha = 1./ (real)(samples); 
		cublasCheckError( cublasDscal( blasHandle, zRows * samples, &alpha, scaledZ, 1));

		//Z * Z^T
		alpha = 1.; beta = 0; 
		cublasCheckError( cublasDgemm( blasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
									zRows, zRows, samples,
									&alpha, zMatrix, zRows, scaledZ, zRows, 
									&beta, nextDevPtr, zRows ) ); 

		//Inverse Computation. 
      computeMatrixInverseCusolver( nextDevPtr,
                           omegaZInv + omegaOffsets[ model->cLayers + l ],
                              zRows, nextDevPtr + zRows * zRows, pageLckPtr,
                           kfac_info->regLambda, kfac_info->dampGamma, curStream, curHandle ) ;
	}

}

void computeKFACInverses( KFAC_CURVATURE_INFO *kfac_info,
                              KFAC_THREAD_INFO *kfacThreadInfo,
                              CNN_MODEL *model,
                           int batchNo, real *devPtr, real *pageLckPtr, int currentIteration, int miniBatchNo )
{

#ifdef DEBUG_KFAC_THREAD
   fprintf( stderr, "KFAC-COMPUTEKFACINVERSES.... batchNo: %d, currentIteration: %d \n", batchNo, currentIteration );
#endif

   if ((miniBatchNo % kfac_info->inverseFreq) != 0) {
      //Check if we have already signalled here. 
      if ( kfacThreadInfo->signalled  == 0) {
#ifdef DEBUG_KFAC_THREAD
         fprintf( stderr, "KFAC-THREAD.. NO SIGNAL to the thread.. \n");
#endif
         return ;
      }

      //Signalled ... check how many minibatches have passed. 
      kfacThreadInfo->signalled ++ ;
      if ((kfacThreadInfo->signalled - kfacThreadInfo->batchNo) >= 2) {
#ifdef DEBUG_KFAC_THREAD
         fprintf( stderr, "KFAC-THREAD... Expired mini-batches after signalling... waiting... \n"); 
#endif
         // We have to wait here.  semaphores here. 
         sem_wait( &kfacThreadInfo->resSemaphore );

      } else {
#ifdef DEBUG_KFAC_THREAD
       fprintf( stderr, "KFAC-THREAD... using the STALE data for now... \n"); 
#endif
         return;
      }

      if (kfacThreadInfo->workComplete == 1) {

         kfacThreadInfo->workComplete = 0;
         kfacThreadInfo->signalled = 0;
         kfacThreadInfo->batchNo = 0;

			// Copy The resulst back to the master, Slave completed the computation here. 
         copyResultToMaster (kfacThreadInfo, kfac_info, model);
      } else {
         fprintf( stderr, " ... WE HAVE A PROBLEM WITH THE THREADED IMPLEMENTATION..... \n\n\n");
         exit( -1 );
      }

      return;
   }


   // Need to compute the KFAC INVERSES HERE ... 
   prepSlaveDevice( kfacThreadInfo, kfac_info, model );

   //SIGNAL
   pthread_mutex_lock( &kfacThreadInfo->initiateMutex );

      pthread_cond_signal( &kfacThreadInfo->initiateCVariable );

   pthread_mutex_unlock( &kfacThreadInfo->initiateMutex );

   kfacThreadInfo->signalled = miniBatchNo;
   kfacThreadInfo->batchNo = miniBatchNo;
}


void computeSpeedUpLambdaDelta( KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model,
   int samples, real *dx, real *dampedLambda, int *zOffsets, int *zztOffsets, int iterCount )
{
   real *delta ;
   if (iterCount == 0) {

      for (int i = 0; i < model->cLayers; i ++ ) {
         CONV_LAYER *convLayer = &model->convLayer[ i ];
         POOL_LAYER *poolLayer = &model->poolLayer[ i ];

         //Delta dimensions -- n * h * w X outChannels
			if (convLayer->batchNorm == PERFORM_NO_BATCH_NORM) 
				delta = dx + zOffsets[ i + 1 ] + convLayer->activationOffset; 
			else 
				delta = dx + zOffsets[ i + 1 ] + convLayer->batchNormOffset; 

         copy_device( dampedLambda + zztOffsets[ i + 1 ], delta,
                        sizeof(real) * convLayer->outChannels * convLayer->outHeight * convLayer->outWidth * samples,
                        ERROR_MEMCPY_DEVICE_DEVICE );
      }

      for (int l = 0; l < model->lLayers; l ++) {
         FC_LAYER *ll = &model->fcLayer[ l ];
         delta = dx + zOffsets[ l + 1 + model->cLayers ];

         copy_device( dampedLambda + zztOffsets[ l + 1 + model->cLayers ], delta,
                        sizeof(real) * ll->out * samples, ERROR_MEMCPY_DEVICE_DEVICE );
      }
   } else {
      //All the succeeding mini-batches after the very first batch... 
      // Here we do the updateRunningStats on each of the Z's 

      for (int i = 0; i < model->cLayers; i ++ ) {
         CONV_LAYER *convLayer = &model->convLayer[ i ];
         POOL_LAYER *poolLayer = &model->poolLayer[ i ];

			if (convLayer->batchNorm == PERFORM_NO_BATCH_NORM) 
				delta = dx + zOffsets[ i + 1 ] + convLayer->activationOffset; 
			else 
				delta = dx + zOffsets[ i + 1 ] + convLayer->batchNormOffset; 

         updateRunningStats( dampedLambda + zztOffsets[ i + 1 ],
                                 delta,
                                 convLayer->outHeight * convLayer->outWidth * convLayer->outChannels * samples,
                                 sqrt( kfac_info->stats_decay ));
      }

      for (int l = 0; l < model->lLayers; l ++) {
         FC_LAYER *ll = &model->fcLayer[ l ];
         delta = dx + zOffsets[ l + 1 + model->cLayers ];

         updateRunningStats( dampedLambda + zztOffsets[ l + 1 + model->cLayers ],
                              delta, ll->out * samples, sqrt( kfac_info->stats_decay ));
      }
   }
}



//  There is no Bias for this term here. 
void computeLambdaDelta( KFAC_CURVATURE_INFO *kfac_info, CNN_MODEL *model, 
	int samples, real *dx, real *dampedLambda, real *lambdaGInv, 
	int *zOffsets, int *zztOffsets, real *devPtr, real *pageLckPtr, 
	cublasHandle_t blasHandle, cudaStream_t curStream, cusolverDnHandle_t curHandle ) 
{
	real *nextDevPtr = devPtr;
	real *delta; 
	real *nextDevPtr2;

	const real **matPtr; 
	real **invMatPtr; 
	real *tmp; 

	real alpha, beta; 

	int *lambdaOffsets = kfac_info->LambdaGOffsets; 
	int *omegaOffsets = kfac_info->OmegaZOffsets; 

	for (int l = 0; l < model->cLayers; l ++) {
		CONV_LAYER *convLayer = &model->convLayer[ l ]; 

		//Delta dimensions -- n * h * w X outChannels
		if (dampedLambda != NULL) { 
			delta = dampedLambda + zztOffsets[ l + 1 ];	
		} else { 
			if (convLayer->batchNorm == PERFORM_NO_BATCH_NORM) 
				delta = dx + zOffsets[ l + 1 ] + convLayer->activationOffset; 
			else 
				delta = dx + zOffsets[ l + 1 ] + convLayer->batchNormOffset; 
		} 

/*
		alpha = convLayer->outHeight * convLayer->outWidth * samples;
		cublasCheckError( cublasDscal( blasHandle, 
			convLayer->outChannels * convLayer->outHeight * convLayer->outWidth * samples, &alpha, 
													delta, 1 ) ); 

		nextDevPtr2 = nextDevPtr + convLayer->outChannels * convLayer->outChannels; 
		copy_device( nextDevPtr2, delta, sizeof(real) * convLayer->outHeight * convLayer->outWidth * samples * convLayer->outChannels, 
							ERROR_MEMCPY_DEVICE_DEVICE ); 

		alpha = 1./(real)(samples * convLayer->outWidth * convLayer->outHeight);
		cublasCheckError( cublasDscal( blasHandle, 
				convLayer->outChannels * samples * convLayer->outWidth * convLayer->outHeight, 
				&alpha, nextDevPtr2, 1 ) ); 
*/

		// G^T * G --> outCHannels * outCHannels are the dimensions of the results
		alpha = 1.; beta = 0.; 
		cublasCheckError( cublasDgemm( blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
										convLayer->outChannels, convLayer->outChannels, 
										samples * convLayer->outHeight * convLayer->outWidth, 
										&alpha, delta, samples * convLayer->outHeight * convLayer->outWidth, 
										delta, samples * convLayer->outHeight * convLayer->outWidth, 
										&beta, nextDevPtr, convLayer->outChannels ) ); 

		alpha = convLayer->outHeight * convLayer->outWidth * samples;
		cublasCheckError( cublasDscal( blasHandle, convLayer->outChannels * convLayer->outChannels, &alpha, nextDevPtr, 1 ) ); 

		

		//Inverse of G^T * G Here. 
		nextDevPtr2 = nextDevPtr + convLayer->outChannels * convLayer->outChannels; 
		computeMatrixInverseCusolver( nextDevPtr, lambdaGInv + lambdaOffsets[ l ], 
										convLayer->outChannels, nextDevPtr2, pageLckPtr, 
										kfac_info->regLambda, kfac_info->dampGamma, curStream, curHandle); 
	}


	for (int l = 0; l < model->lLayers; l ++) {
		FC_LAYER *fcLayer = &model->fcLayer[ l ]; 

		// delta --> shape is fcLayer->out * n
		if (dampedLambda != NULL) 
			delta = dampedLambda + zztOffsets[ model->cLayers + l + 1 ]; 
		else
			delta = dx + zOffsets[ model->cLayers + l + 1 ]; 

		// delta * delta ^T
		alpha = 1; beta = 0; 
		cublasCheckError( cublasDgemm( blasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
									fcLayer->out, fcLayer->out, samples, 
									&alpha, delta, fcLayer->out, delta, fcLayer->out, 
									&beta, nextDevPtr, fcLayer->out )); 

		//TODO --- Since we are using Size_Average = True. 
		//				dx terms are scaled by the number of data points... 
		// 		 which changes the alpha here. 
		//alpha = (real)samples / (real)(samples * samples); 
		alpha = (real) samples; 
		cublasCheckError( cublasDscal( blasHandle, fcLayer->out * fcLayer->out, &alpha, nextDevPtr, 1 ) ); 

		//Inverse of ( delta * delta^T )
		computeMatrixInverseCusolver( nextDevPtr, 
									lambdaGInv + lambdaOffsets[ model->cLayers + l ], 
									fcLayer->out, nextDevPtr + fcLayer->out * fcLayer->out, pageLckPtr, 
									kfac_info->regLambda, kfac_info->dampGamma, curStream, curHandle ); 	
	}
}
