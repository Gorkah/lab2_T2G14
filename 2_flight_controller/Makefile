CC=gcc
MPICC=mpicc
CFLAGS=-O3 -march=native -lm -std=c99

all: fc_seq fc_mpi

fc_seq: fc_seq.c auxiliar.c
	$(CC) fc_seq.c auxiliar.c -o fc_seq $(CFLAGS)

fc_mpi: fc_mpi.c auxiliar.c
	$(MPICC) fc_mpi.c auxiliar.c -o fc_mpi $(CFLAGS)

clean:
	rm -f fc_seq fc_mpi

