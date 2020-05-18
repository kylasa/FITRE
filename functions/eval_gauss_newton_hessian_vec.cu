
#include <functions/eval_hessian_vec.h>

#include <core/datadefs.h>
#include <nn/nn_decl.h>
#include <core/structdefs.h>
#include <core/errors.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>

#include <functions/dev_mat_mat_scale.h>
#include <functions/dev_initializations.h>
#include <functions/dev_mat_vec_addition.h>
#include <functions/dev_mat_vec_scale.h>
#include <functions/dev_layer_error.h>
#include <functions/dev_hessian_helpers.h>

//This uses the sampled dataset. 
void gaussNewtonHessianVec ( NN_MODEL *model, DEVICE_DATASET *data, 
			real *z, real *dx, real *vec, real *weights, SCRATCH_AREA *scratch, 
			DATASET_SIZE allData)
{
	
	// local variables here.
	int *layerSizes = model->layerSizes; 
	int *zOffsets = model->sZOffsets; 
	if (allData == FULL_DATASET) zOffsets = model->zOffsets; 
	int *rZOffsets = model->sRZOffsets;
	if (allData == FULL_DATASET) rZOffsets = model->rZOffsets;
	int *bOffsets = model->bOffsets; 
	int *wOffsets = model->wOffsets;
	
	int numLayers = model->numLayers; 
	int numFeatures = data->features;

	int n = data->sampleSize; 
	real *trainX = data->sampledTrainX;
	real *trainY = data->sampledTrainY;
	if (allData == FULL_DATASET) {
		trainX = data->trainSetX;
		trainY = data->trainSetY;
		n = data->trainSizeX; 
	}

	// derivative parameters here. 	
	real *RdW = scratch->nextDevPtr; 			// wOffsets
	real *Rdz = RdW + model->pSize; 			// zoffsets

	real *Rdx = Rdz + model->sampledZSize; 	// REVERSE Z Offsets
	if (allData == FULL_DATASET) Rdx = Rdz + model->zSize; 				// REVERSE Z Offsets

	real *Rz = Rdx + model->sampledRSize; 		// zoffsets
	if (allData == FULL_DATASET) Rz = Rdx + model->rFullSize; 		// zoffsets

	real *Rx = Rz + model->sampledZSize; 		// zoffsets
	if (allData == FULL_DATASET) Rx = Rz + model->zSize; 		// zoffsets

	real *oneVector = Rx + model->sampledZSize;
	if (allData == FULL_DATASET) oneVector = Rx + model->zSize;

	real *nextDevPtr = oneVector + 2 * numFeatures;

	//tmp 
	real *rxi; 
	real *rzi; 
	real alpha = 1, beta = 0;
	real *VW = vec; 

	int numElements, numBlocks;

	//testing
	cuda_memset( RdW, 0, model->pSize * sizeof(real), ERROR_MEMSET ); 

	//initializations here. 
	//Rz[0] = 0;
	cuda_memset( Rz + zOffsets[0], 0, layerSizes[0] * n * sizeof(real), ERROR_MEMSET ); 

	//Forward pass for the second derivatives. 
	for (int i = 0; i < numLayers; i ++) {

		rxi = Rx + zOffsets [i+1];
		rzi = Rz + zOffsets [i+1];

		//bxsfun (VW* z, vb);
		//VW(layerSizes[i+1], layerSizes[i]) * z(layerSizes[i], n)
		cublasCheckError( cublasDgemm ( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
					layerSizes[ i+1 ], n, layerSizes[ i ], 
					&alpha, VW + wOffsets[ i ], layerSizes[ i+1 ], 
					(i == 0) ? (trainX) : ( z + zOffsets[ i ] ), layerSizes[ i ], 
					&beta, rxi, layerSizes[ i+1 ] ) );  	

		// + Vb(layerSizes[i+1], 1)
		numElements = layerSizes[ i+1 ] * n; 
		numBlocks = numElements / BLOCK_SIZE + 
								(( numElements % BLOCK_SIZE ) == 0 ? 0 : 1); 
		kerUtilsAddColumnToMatrix <<<numBlocks, BLOCK_SIZE >>> 
			( rxi, layerSizes[ i+1 ], n, VW + bOffsets[i] ); 
		cudaThreadSynchronize (); 
		cudaCheckError (); 

		if (i == 0){
			; //Do nothing since Rzi is 0.
		} else {
			//W(i)(layerSizes[i+1], layerSizes[i]) * Rz(i)(layerSizes[i], n)
			cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 	
				layerSizes[ i+1 ], n, layerSizes[ i ], 
				&alpha, weights + wOffsets[ i ], layerSizes[ i+1 ], 
				Rz + zOffsets[ i ], layerSizes[ i ], 
				&beta, nextDevPtr, layerSizes[ i+1 ] ) ); 	

			cublasCheckError( 
				cublasDaxpy( cublasHandle, layerSizes[ i+1 ] * n,
									&alpha, nextDevPtr, 1, 
								rxi, 1 ) );
		}

		switch( model->actFuns[ i ] ){
			case ACT_LOGISTIC: 

				numElements = layerSizes[ i+1 ] * n; 
				copy_device( rzi, rxi, numElements * sizeof(real), ERROR_MEMCPY_DEVICE_DEVICE ); 

				numBlocks = numElements / BLOCK_SIZE + 
									(( numElements % BLOCK_SIZE == 0 ) ? 0 : 1); 
				kerNNBackPropLogisticErrors <<< numBlocks, BLOCK_SIZE >>> 
					(rzi, z + zOffsets[ i+1 ], numElements); 
				cudaThreadSynchronize (); 
				cudaCheckError (); 
				break;

			case ACT_TANH:

				numElements = layerSizes[ i+1 ] * n; 
				copy_device( rzi, rxi, numElements * sizeof(real), ERROR_MEMCPY_DEVICE_DEVICE ); 

				numBlocks = numElements / BLOCK_SIZE + 
									(( numElements % BLOCK_SIZE == 0 ) ? 0 : 1); 
				kerNNBackPropTanHErrors <<< numBlocks, BLOCK_SIZE >>> 
					(rzi, z + zOffsets[ i+1 ], numElements); 
				cudaThreadSynchronize (); 
				cudaCheckError (); 
				break;

			case ACT_LINEAR: 
				numElements = layerSizes[ i+1 ] * n; 
				copy_device( rzi, rxi, numElements * sizeof(real), ERROR_MEMCPY_DEVICE_DEVICE ); 

				break;

			case ACT_SOFTMAX: 
				//rzi = z[i+1] .* rxi
				numElements = layerSizes[ i+1 ] * n; 
				numBlocks = numElements / BLOCK_SIZE + 
									(( numElements % BLOCK_SIZE == 0 ) ? 0 : 1); 
				kerUtilsMatMatScale <<< numBlocks, BLOCK_SIZE >>> 
					(z + zOffsets[ i+1 ], rxi, numElements, rzi); 
				cudaThreadSynchronize (); 	
				cudaCheckError ();

				//compute sum along cols = nextDevPtr
				// use oneVector * matrix and store the columns sums in nextDev
   			//numElements = model->layerSizes[ numLayers - 1]; 
   			numElements = layerSizes[ i+1 ]; 
   			numBlocks = numElements / BLOCK_SIZE + 
               				(( numElements % BLOCK_SIZE  == 0) ? 0 : 1 );  
   			kerInitOneVector <<< numBlocks, BLOCK_SIZE>>> 
      								( oneVector, numElements );  
   			cudaThreadSynchronize (); 
   			cudaCheckError (); 

				//sum (rzi, 1) = sum along columns here. 
         	cublasCheckError( 
            	cublasDgemv( cublasHandle, CUBLAS_OP_T, 
                        layerSizes[i+1], n, &alpha, rzi, layerSizes[ i+1 ], 
                        oneVector, 1, &beta, nextDevPtr, 1) );    

				kerUtilsMatRowVecScale <<< numBlocks, BLOCK_SIZE >>> 
					( z + zOffsets[ i+1 ], layerSizes [i+1], n, nextDevPtr, nextDevPtr + n); 
				cudaThreadSynchronize (); 
				cudaCheckError ();

				//rzi = rzi - bsxfun( @times, z{i+1}, sum(rzi, 1) );
				alpha = -1; 
				cublasCheckError( cublasDaxpy( cublasHandle, layerSizes[ i+1 ] * n, 
											&alpha, (nextDevPtr + n) , 1, 
											rzi, 1 ) );
				alpha = 1;
				
				break;

			default: 
				fprintf( stderr, "hessianVec: unknown activation function at Layer: %d \n", i ); 
				exit ( -1 ); 
		}

	} //end of for loop

	//At the end of all layers... compute the error terms
	switch( model->actFuns[ numLayers - 1] ) {

		// Rdx{numlayers} = - Rz{numlayers+1};
		case ACT_LOGISTIC:
		case ACT_SOFTMAX: 
			// Rdx{numlayers} = - Rz{numlayers+1};
			// CAREFUL WITH THE SIZES OF RDX here.... 
			// TODO TODO TODO
			// Rdx is the reverse of the Z scale. 
			copy_device( Rdx + rZOffsets[ numLayers - 1], Rz + zOffsets[ numLayers ], 
								sizeof( real ) * n * layerSizes[ numLayers ], ERROR_MEMCPY_DEVICE_DEVICE ); 
			alpha = -1; 
			cublasCheckError( cublasDscal( cublasHandle, n * layerSizes[numLayers], 
										&alpha, Rdx + rZOffsets[numLayers - 1], 1 ) );
			alpha = 1; 
			break;

		case ACT_LINEAR: 
		default:
			fprintf( stderr, "Unknow activation function here ... \n"); 
			exit( -1 ); 
	}

	//RdW{numlayers} = Rdx{numlayers} * z{numlayers}' ;
	//						Rdx(layerSizes[numLayers], n) 
	//						z(layerSizes[numLayers-1], n)

	//Rdx( layerSizes[numLayers], n) * z( layerSizes[ numLayers -1 ], n)
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
									layerSizes[ numLayers ], layerSizes[ numLayers - 1], n, 
									&alpha, Rdx + rZOffsets[ numLayers - 1 ], layerSizes[ numLayers ], 
									z + zOffsets[ numLayers - 1 ], layerSizes[ numLayers - 1 ], 
									&beta, RdW + wOffsets[numLayers - 1], layerSizes[ numLayers ] ) ); 
	
	//update Rdb... 
	// Rdb{numlayers} = sum(Rdx{numlayers},2);
   numElements = n; 
   numBlocks = numElements / BLOCK_SIZE + 
               				(( numElements % BLOCK_SIZE  == 0) ? 0 : 1 );  
   kerInitOneVector <<< numBlocks, BLOCK_SIZE>>> 
      ( oneVector, numElements );  
   cudaThreadSynchronize (); 
   cudaCheckError (); 

	//sum Rdx
	cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, 
								layerSizes[ numLayers ], n, 
								&alpha, Rdx + rZOffsets[ numLayers - 1 ], layerSizes[ numLayers ], 
								oneVector, 1, &beta, RdW + bOffsets[ numLayers - 1 ], 1 ) ); 

	//udpate Rdz
	// Rdz{numlayers} = VW{numlayers}'*dx{numlayers} + W{numlayers}'*Rdx{numlayers};
	/*
						VW( layerSizes(numLayers), layerSizes(numLayers-1) )
						dx( layerSizes( numLayers ), n )
						W( layerSizes(numLayers), layerSizes(numLayers-1) )
						Rdx( layerSizes( numLayers ), n )	
	*/
	//testing
	cuda_memset( Rdz + zOffsets[ numLayers - 1 ], 0, layerSizes[ numLayers - 1 ] * n * sizeof(real), 
						ERROR_MEMSET ); 
	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
								layerSizes[ numLayers - 1], n, layerSizes[ numLayers ], 
								&alpha, VW + wOffsets[ numLayers - 1], layerSizes[ numLayers ], 
								dx + rZOffsets[ numLayers - 1 ], layerSizes[ numLayers ], 
								&beta, Rdz + zOffsets[ numLayers - 1], layerSizes[ numLayers - 1] ) );

	cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
								layerSizes[ numLayers - 1 ], n, layerSizes[ numLayers ], 
								&alpha, weights + wOffsets[ numLayers - 1], layerSizes[ numLayers ], 
								Rdx + rZOffsets[ numLayers - 1 ], layerSizes[ numLayers ], 
								&beta, nextDevPtr, layerSizes[ numLayers - 1 ] )); 

	cublasCheckError( cublasDaxpy( cublasHandle, layerSizes[ numLayers - 1 ] * n, &alpha, 
								nextDevPtr, 1 , Rdz + zOffsets[ numLayers - 1 ], 1 )); 


	//backward propagation... 
	for (int i = numLayers - 2; i >= 0; i --) {

		switch( model->actFuns[ i ] ) {
			case ACT_LOGISTIC: 
				//Rdx{i} = (1 - z{i+1}).*z{i+1}.*Rdz{i+1} + Rx{i+1}.*(1 - 2*z{i+1}).*dx{i};
				numElements = layerSizes[ i+1 ] * n; 
				numBlocks = numElements / BLOCK_SIZE + 
								( (numElements % BLOCK_SIZE == 0) ? 0 : 1 ); 	
				eval_gauss_newton_backprop <<< numBlocks, BLOCK_SIZE >>> 
						( z + zOffsets[ i+1 ], Rdz + zOffsets[ i+1 ], Rdx + rZOffsets[ i ], numElements ); 
				cudaThreadSynchronize (); 
				cudaCheckError ();

				break; 

			case ACT_LINEAR: 
				// Rdx[i] = Rdz[i + 1]
				copy_device ( Rdx + rZOffsets[ i ], Rdz + zOffsets[ i + 1], 
									sizeof (real) * n * layerSizes[ i+1 ], ERROR_MEMCPY_DEVICE_DEVICE ); 
				break; 

			case ACT_TANH: 
				;
				break; 

			default: 
				fprintf( stderr, "Unknown layer type in hessian vec... back prop\n" ); 
				exit ( -1 ); 
		}

		//update RdW, Rdb, Rdz
      //RdW{i} = Rdx{i} * z{i}' + dx{i} * Rz{i}';
		/*
							Rdx(layerSizes[i+1], n) 
							z(layerSizes[i], n)
		*/
		//testing
		cuda_memset( RdW + wOffsets[ i ], 0, layerSizes[ i+1 ] * layerSizes[ i ] * sizeof(real), ERROR_MEMSET ); 
		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
								layerSizes[ i+1 ], layerSizes[ i ], n, 
								&alpha, Rdx + rZOffsets[ i ], layerSizes[ i+1 ], 
								//z + zOffsets[ i ], layerSizes[ i ], 
								//SUDHIR TODO FIX.... 
								(i == 0) ? (trainX) : (z + zOffsets[ i ]), layerSizes[ i ], 
								&beta, RdW + wOffsets[ i ], layerSizes[ i+1 ] ) ); 

      //     Rdb{i} = sum(Rdx{i},2);
   	numElements = n; 
   	numBlocks = numElements / BLOCK_SIZE + 
               				(( numElements % BLOCK_SIZE  == 0) ? 0 : 1 );  
   	kerInitOneVector <<< numBlocks, BLOCK_SIZE>>> 
      	( oneVector, numElements );  
   	cudaThreadSynchronize (); 
   	cudaCheckError (); 

		alpha = 1;
      cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, 
                        layerSizes[i+1], n, &alpha, Rdx + rZOffsets[i], layerSizes[ i+1 ], 
                        oneVector, 1, &beta, RdW + bOffsets[ i ], 1) );    

      //    Rdz{i} = W{i}'*Rdx{i};
		/*
						W( layerSizes(i+1), layerSizes(i) )
						Rdx( layerSizes( i+1 ), n )	
		*/
		//testing
		cuda_memset( Rdz + zOffsets[ i ], 0, layerSizes[ i ] * n * sizeof(real), 
							ERROR_MEMSET ); 

		cublasCheckError( cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
								layerSizes[ i ], n, layerSizes[ i+1 ], 
								&alpha, weights + wOffsets[ i ], layerSizes[ i+1 ], 
								Rdx + rZOffsets[ i ], layerSizes[ i+1 ], 
								&beta, Rdz + zOffsets[ i ], layerSizes[ i ] ) ); 
	}

	//Done with hessian vec computation... return the result vector here. 
	// [RdW, Rdb] is the result.

	//HV = HV / n. 
	alpha = -(1. / ((real)n)); 
	cublasCheckError( cublasDscal( cublasHandle, model->pSize, &alpha, RdW, 1 ) );
}
