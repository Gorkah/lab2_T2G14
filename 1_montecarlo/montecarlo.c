#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <stdint.h>

// PCG32 Random Number Generator
typedef struct { uint64_t state; uint64_t inc; } pcg32_random_t;

double pcg32_random(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    uint32_t ran_int = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return (double)ran_int / (double)UINT32_MAX;
}

// Check if a point is inside the unit hypersphere
int is_inside_sphere(double* coords, int d) {
    double sum = 0.0;
    for (int i = 0; i < d; i++) {
        sum += coords[i] * coords[i];
    }
    return (sum <= 1.0) ? 1 : 0;
}

// Analytical calculation of the sphere/cube volume ratio
double analytical_ratio(int d) {
    double num = pow(M_PI, d / 2.0);
    double denom = pow(2.0, d) * tgamma(d / 2.0 + 1.0);
    return num / denom;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int d = 3;                // Default dimension
    long NUM_SAMPLES = 1000000; // Default number of samples
    long SEED = time(NULL);   // Default seed
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    if (argc >= 2) d = atoi(argv[1]);
    if (argc >= 3) NUM_SAMPLES = atol(argv[2]);
    if (argc >= 4) SEED = atol(argv[3]);
    
    // Samples per process
    long samples_per_process = NUM_SAMPLES / size;
    long remainder = NUM_SAMPLES % size;
    long my_samples = samples_per_process + (rank < remainder ? 1 : 0);
    
    // Initialize random number generator
    pcg32_random_t rng;
    rng.state = SEED + rank;
    rng.inc = (rank << 16) | 0x3039;
    
    // Start timing
    double start_time = MPI_Wtime();
    
    // Allocate array for random coordinates
    double* coords = (double*)malloc(d * sizeof(double));
    if (coords == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Finalize();
        return 1;
    }
    
    // Perform Monte Carlo simulation
    long points_inside = 0;
    
    for (long i = 0; i < my_samples; i++) {
        // Generate random point
        for (int j = 0; j < d; j++) {
            coords[j] = 2.0 * pcg32_random(&rng) - 1.0; // Values between -1 and 1
        }
        
        // Check if inside unit sphere
        if (is_inside_sphere(coords, d)) {
            points_inside++;
        }
    }
    
    // Gather results
    long total_points_inside = 0;
    MPI_Reduce(&points_inside, &total_points_inside, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Calculate total samples (in case of uneven distribution)
    long total_samples = 0;
    MPI_Reduce(&my_samples, &total_samples, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // End timing
    double elapsed_time = MPI_Wtime() - start_time;
    double max_time;
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Print results on rank 0
    if (rank == 0) {
        double mc_ratio = (double)total_points_inside / total_samples;
        double analytical = analytical_ratio(d);
        double error = fabs(mc_ratio - analytical);
        
        printf("Monte Carlo sphere/cube ratio estimation\n");
        printf("N: %ld samples, d: %d, seed %ld, size: %d\n", 
               total_samples, d, SEED, size);
        printf("Ratio = %.3e (%.3e) Err: %.2e\n", 
               mc_ratio, analytical, error);
        printf("Elapsed time: %.3f seconds\n", max_time);
    }
    
    // Cleanup
    free(coords);
    MPI_Finalize();
    
    return 0;
}
