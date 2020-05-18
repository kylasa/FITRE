#ifndef _H_MEM_SIZES__
#define _H_MEM_SIZES__

#define __GIGA_BYTE_SIZE__				((1024 * 1024 * 1024))

#define HOST_WORKSPACE_SIZE			((size_t(1) << 30) * 12) // 4 Gb
//#define DEVICE_WORKSPACE_SIZE			((size_t(1) << 33)) // 8 Gb
#define DEVICE_WORKSPACE_SIZE			((size_t(1) << 30) * 12) // 8 Gb
#define PAGE_LOCKED_WORKSPACE_SIZE	((size_t(1) << 20)) // 1 MB

#define DEBUG_SCRATCH_SIZE				((size_t(1) << 24))

#endif
