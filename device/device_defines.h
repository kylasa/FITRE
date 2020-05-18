#ifndef __H_DEVICE_DEFS__
#define __H_DEVICE_DEFS__

#define GLOBAL __global__
#define DEVICE __device__

#define WARP_SIZE 	32
//#define BLOCK_SIZE 	1024

extern int BLOCK_SIZE; 

extern int DEVICE_NUM_BLOCKS;

extern int __THREADS_PER_SAMPLE__; 
#endif
