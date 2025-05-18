CC=gcc
MPICC=mpicc
CFLAGS=-O3 -march=native -lm -std=c99

all: seq mpi

seq:
	$(CC) fc_seq.c auxiliar.c -o fc_seq $(CFLAGS)

mpi:
	$(MPICC) fc_mpi.c auxiliar.c -o fc_mpi $(CFLAGS)

clean:
	rm -f fc_seq fc_mpi

