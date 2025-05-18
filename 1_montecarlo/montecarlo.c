#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <mpi.h>

// PCG Random Number Generator
typedef struct { uint64_t state; uint64_t inc; } pcg32_random_t;

double pcg32_random(pcg32_random_t* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    uint32_t ran_int = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return (double)ran_int / (double)UINT32_MAX;
}

int main(int argc, char *argv[]) {
    // Variables
    int rank, size;
    int d = 3; // Default dimension
    long NUM_SAMPLES = 1000000; // Default number of samples
    long SEED = time(NULL); // Default seed
    double ratio = 0.0, global_ratio = 0.0;
    double start_time, end_time, elapsed_time;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments
    if (argc == 4) {
        d = atoi(argv[1]);
        NUM_SAMPLES = atol(argv[2]);
        SEED = atol(argv[3]);
    }

    // Divide the total number of samples among processes
    long local_samples = NUM_SAMPLES / size;
    long extra_samples = NUM_SAMPLES % size;
    if (rank < extra_samples) {
        local_samples++;
    }

    // Initialize random generator
    pcg32_random_t rng;
    rng.state = SEED + rank;
    rng.inc = (rank << 16) | 0x3039;

    // Monte Carlo sampling
    long inside_sphere = 0;
    for (long i = 0; i < local_samples; i++) {
        double r_sq = 0.0;
        for (int j = 0; j < d; j++) {
            double coord = 2.0 * pcg32_random(&rng) - 1.0; // Random coordinate in [-1, 1]
            r_sq += coord * coord;
        }
        if (r_sq <= 1.0) {
            inside_sphere++;
        }
    }

    // Compute local ratio
    double local_ratio = (double)inside_sphere / (double)local_samples;

    // Reduce to compute global ratio
    MPI_Reduce(&local_ratio, &global_ratio, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Measure elapsed time
    start_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    // Print results
    if (rank == 0) {
        global_ratio /= size;
        printf("Monte Carlo sphere/cube ratio estimation\n");
        printf("N: %ld samples, d: %d, seed %ld, size: %d\n", NUM_SAMPLES, d, SEED, size);
        printf("Ratio = %.3e (%.3e) Err: %.2e\n", global_ratio, pow(M_PI, d / 2.0) / (tgamma(d / 2.0 + 1) * pow(2, d)), fabs(global_ratio - pow(M_PI, d / 2.0) / (tgamma(d / 2.0 + 1) * pow(2, d))));
        printf("Elapsed time: %.3f seconds\n", elapsed_time);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}