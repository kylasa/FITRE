
#include <drivers/norm_driver.h>

#include <device/cuda_utils.h>
#include <device/reduce.h>
#include <device/handles.h>
#include <device/gen_random.h>

#include <functions/dev_initializations.h>

#include <utilities/utils.h>
#include <utilities/reduce_helper.h>
#include <utilities/norm_inf.h>

#include <core/datadefs.h>
#include <core/errors.h>


void testNorm(SCRATCH_AREA *s)
{
	real *hostPtr = s->hostWorkspace; 
	real *devPtr = s->devWorkspace; 
	real *page = s->pageLckWorkspace; 

	real start, total; 
	real idx;
	int count = 842320;

   getRandomVector( count, NULL, devPtr, RAND_NORMAL ); 
		
	start = Get_Time (); 
	norm_inf_host( devPtr, hostPtr,count, page, &idx, devPtr + count); 
	total = Get_Timing_Info (start); 
	fprintf( stderr, "Inf Norm (HOST) is %10.2f, in %f (msecs) \n", *page, (total * 1000.)); 

	start = Get_Time (); 
	norm_inf( devPtr, hostPtr,count, page, &idx, devPtr + count); 
	total = Get_Timing_Info (start); 
	fprintf( stderr, "Inf Norm (DEVICE) is %10.2f, in %f (msecs) \n", *page, (total * 1000.)); 
}
