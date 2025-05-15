CC=mpicc
CFLAGS=-O3 -march=native -lm -std=c99

all: montecarlo

montecarlo: montecarlo.c
	$(CC) montecarlo.c -o montecarlo $(CFLAGS)

clean:
	rm -f montecarlo
