
ifeq ($(ARCH),RVV)
    CC=/home/adcastel/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc
    #CFLAGS= -O0 -g3 -march=rv64gcv_zfh_xtheadc -mabi=lp64d
    CFLAGS= -O3 -march=rv64gcv
    A=RVV
else
    CC=gcc
    CFLAGS= -O3 -march=armv8.2-a+simd+fp+fp16fml 
    A=NEON
endif

ifeq ($(DTYPE),fp32)
    type=fp32
else
    type=fp16
endif

path=kernels/$(A)/$(MR)x$(NR)

kernels  = $(path)/kernels_NEON_$(MR)x$(NR)_$(type).o
mat      = $(path)/exo_matrix_NEON_$(type).o
OBJECTS := $(mat) $(kernels)

all:main

main: $(OBJECTS)
	$(CC) $(CFLAGS) main.c -o test_uk -D$(type) $(OBJECTS) -I$(PWD)/$(path) -L$(PWD) 

$(kernels): 
	$(CC) $(CFLAGS) -o $(kernels) -c $*.c -I$(PWD) -L$(PWD)
$(mat): 
	$(CC) $(CFLAGS) -o $(mat) -c $*.c -I$(PWD) -L$(PWD)

.c.o:
	$(CC) $(CFLAGS) -c $*.c  $(VER) -I$(PWD) -L$(PWD)

clean:
	rm *.o test_uk $(OBJECTS)
