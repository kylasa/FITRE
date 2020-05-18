
#ifndef __H_WARPSUM__
#define __H_WARPSUM__

#include <core/datadefs.h>
#include <device/device_defines.h>

DEVICE  __inline__ double  my_shfl(double x, int lane)
{
        // Split the double number into 2 32b registers.
        int lo, hi; 
        asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x));

        // Shuffle the two 32b registers.
        lo = __shfl_xor(lo, lane);
        hi = __shfl_xor(hi, lane);

        // Recreate the 64b number.
        //asm volatile( "mov.b64 %0, {%1,%2};" : "=d(x)" : "r"(lo), "r"(hi));
        //return x;
        return __hiloint2double( hi, lo);
}

DEVICE  __inline__ double warpSum( double x ) 
{
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
                x += my_shfl( x, offset);
        return x;
}


#endif
