#ifndef _H_UTILS__
#define _H_UTILS__

#include <stdio.h>
#include <stdlib.h>


#include "core/datadefs.h"

void allocate_memory( void **, size_t);
void release_memory( void ** );


real Get_Time ();
real Get_Timing_Info( real t_start );




#endif
