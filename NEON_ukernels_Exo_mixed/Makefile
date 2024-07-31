CC=/home/adcastel/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc
CC=gcc
#CFLAGS= -O0 -g3 -march=rv64gcv_zfh_xtheadc -mabi=lp64d
CFLAGS= -O3 -march=rv64gcv
CFLAGS= -O3 -march=armv8.2-a+simd+fp+fp16fml 
#KPATH=kernels/RVV/

#ifeq ($(SWAPAB), 1)
#	kernels=kernels/RVV/kernels_col_swapAB.o
#	VER=-DSWAPAB

#else
#	kernels=kernels/RVV/kernel_col.o
#	VER=
#endif
kernels=kernel_col.o
OBJECTS := exo_matrix.o $(kernels)
#/RVV/kernel_col.o
#OBJECTS := exo_matrix.o

all:main

main: $(OBJECTS)
	$(CC) $(CFLAGS) main.c -o test_uk $(OBJECTS) -I$(PWD) -L$(PWD) 

$(kernels): 
	$(CC) $(CFLAGS) -o $(kernels) -c $*.c -I$(PWD) -L$(PWD)

.c.o:
	$(CC) $(CFLAGS) -c $*.c $(VER) -I$(PWD) -L$(PWD)

clean:
	rm *.o test_uk
