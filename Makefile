#
#	DEBUG FLAGS for additional debug information 
#	STATS				: collect timing info. and other stats for plots.
#	DEBUG_FIXED		: This will fix the sampling to first sz columns and (1....1) vec for hessianVec operation
#	DEBUG_GRADIENT	: debug information for gradient computation and also model evaluation
#	DEBUG_HESSIAN	: debug information for hessian vec computation.
#	DEBUG_TRUST		: debug information for TRUST REGION problem
#	DEBUG_CG			: debug information for CG Nonlinear Solver. 
#
#DEBUGFLAGS = -DSTATS -DDEBUG_TRUST -DDEBUG_CG
#DEBUGFLAGS = -DSTATS -DDEBUG_CNN -DDEBUG_DETAILED -DDEBUG_BATCH_NORM -DDEBUG_BATCH -DDEBUG_TRUST_REGION -DDEBUG_ROP
#DEBUGFLAGS = -DSTATS -DDEBUG_CNN 
#DEBUGFLAGS = -DSTATS -DDEBUG_CNN
#DEBUGFLAGS = -DSTATS
#DEBUGFLAGS = -DSTATS -DDEBUG_TRUST -DDEBUG_CG
#DEBUGFLAGS = -DDEBUG_TRUST_REGION

CUDA		= /apps/gilbreth/cuda-toolkit/cuda-9.0.176

CC	 	= g++
NVCC	 	= nvcc
LIBS    	= -L$(CUDA)/lib64 -lm -lcuda -lcudart -lcublas -lcusparse -lcusolver -lcurand -lpthread -m64 -Xcompiler -fopenmp
NVCCFLAGS  	= -I. -I$(CUDA)/include -arch=sm_60
CFLAGS  	= -I. -I$(CUDA)/include -Wall -funroll-loops -fstrict-aliasing -O3 $(DEBUGFLAGS)

DEFS    	= $(CFLAGS) $(DEBUGFLAGS)
NVCCDEFS 	= $(NVCCFLAGS)
FLAG    	= $(DEFS) $(INCS) $(LIBS) $(DEBUGFLAGS)
NVCCFLAG 	= $(NVCCDEFS) $(LIBS) $(DEBUGFLAGS)

CSRC = $(wildcard nn/*.c) \
			$(wildcard device/*.c) \
			$(wildcard functions/*.c) \
			$(wildcard augmentation/*.c) \
			$(wildcard solvers/*.c) \
			$(wildcard utilities/*.c) \
			$(wildcard drivers/*.c)

CUSRC = $(wildcard nn/*.cu) \
			$(wildcard device/*.cu) \
			$(wildcard functions/*.cu) \
			$(wildcard augmentation/*.cu) \
			$(wildcard solvers/*.cu) \
			$(wildcard utilities/*.cu) \
			$(wildcard drivers/*.cu)

OBJ	=	$(CSRC:.c=.o) 
OBJ_CU = $(CUSRC:.cu=.o)

%.o:%.cu
	$(NVCC) $(NVCCFLAG) -o $@ -c $<

all:   beta

beta: $(OBJ) $(OBJ_CU)
	$(NVCC) $(OBJ) $(OBJ_CU) -o NewtonTRSolver $(NVCCFLAG) 

clean:
	rm -f ./core/*.o
	rm -f ./device/*.o
	rm -f ./drivers/*.o
	rm -f ./functions/*.o
	rm -f ./nn/*.o
	rm -f ./solvers/*.o
	rm -f ./utilities/*.o
	rm -f ./augmentation/*.o
	rm NewtonTRSolver
