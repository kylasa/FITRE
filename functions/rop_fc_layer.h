
#ifndef __H_ROP_FC_LAYER__
#define __H_ROP_FC_LAYER__

#include <core/datadefs.h>

void applyROpLayerActivation (int actFunction, 
         real *W, int wRows, int wCols,  
         real *b, int bRows, 
         real *z, int zRows, int zCols, 
         real *z1, int z1Rows, int z1Cols, 
         real *VW, real * Vb, real *rz, 
         real *output, int offset, real *devPtr, real *hostPtr);

#endif
