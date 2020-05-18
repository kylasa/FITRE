
#include <utilities/csr_helper.h>

#include <cuda.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <device/cuda_utils.h>
#include <device/device_defines.h>
#include <device/handles.h>



void convertGradientSampleToCSR 
		(SparseDataset *spSamplingMatrix, int sampleSize, int cols, real *devPtr) {

   //make sure that the data is sorted here. 
   size_t pBufferSizeInBytes = 0;  
   void* pBuffer = (void *)devPtr; 

   //Sampled Dataset Here. 
   cusparseCheckError( 
         cusparseXcoosort_bufferSizeExt( 
            cusparseHandle, sampleSize, cols, spSamplingMatrix->nnz, 
            spSamplingMatrix->rowPtr, spSamplingMatrix->colPtr, &pBufferSizeInBytes ) );  

   cusparseCheckError( 
      cusparseCreateIdentityPermutation( cusparseHandle, spSamplingMatrix->nnz, spSamplingMatrix->P) ); 
   
   cusparseCheckError( 
      cusparseXcoosortByRow( cusparseHandle, sampleSize, cols, spSamplingMatrix->nnz, 
            spSamplingMatrix->rowPtr, spSamplingMatrix->colPtr, spSamplingMatrix->P, pBuffer ) ); 

   cusparseCheckError( 
      cusparseDgthr( cusparseHandle, spSamplingMatrix->nnz, spSamplingMatrix->valPtr, 
            spSamplingMatrix->sortedVals, spSamplingMatrix->P, CUSPARSE_INDEX_BASE_ZERO ) );  
   //convert to csr format. 
   cusparseCheckError( 
         cusparseXcoo2csr( cusparseHandle, spSamplingMatrix->rowPtr, spSamplingMatrix->nnz, sampleSize,    
            spSamplingMatrix->rowCsrPtr, CUSPARSE_INDEX_BASE_ZERO )
      );        

   //fprintf( stderr, "Converting gradient to CSR .... \n"); 
}

