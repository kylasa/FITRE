
#ifndef __H_EXTRACT_COLUMNS__
#define __H_EXTRACT_COLUMNS__

#include <core/datadefs.h>
#include <device/device_defines.h>

GLOBAL void kerExtractColumns( 
   real *tgt, real *src, int *indices, int rows, int numElements);

#endif
