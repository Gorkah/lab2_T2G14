#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "auxiliar.h"

/// Function to reads the planes from a file
void read_planes_seq(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    char line[MAX_LINE_LENGTH];
    int num_planes = 0;

    // Reading header
    fgets(line, sizeof(line), file);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Map: %lf, %lf : %d %d", x_max, y_max, N, M);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Number of Planes: %d", &num_planes);
    fgets(line, sizeof(line), file);

    // Reading plane data
    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        int idx;
        double x, y, vx, vy;
        if (sscanf(line, "%d %lf %lf %lf %lf",
                   &idx, &x, &y, &vx, &vy) == 5) {
            int index_i = get_index_i(x, *x_max, *N);
            int index_j = get_index_j(y, *y_max, *M);
            int index_map = get_index(index_i, index_j, *N, *M);
            int rank = 0;
            insert_plane(planes, idx, index_map, rank, x, y, vx, vy);
            index++;
        }
    }
    fclose(file);

    printf("Total planes read: %d\n", index);
    assert(num_planes == index);
}

int main(int argc, char **argv) {
    int debug = 0;                      // 0: no debug, 1: shows all planes information during checking
    int N = 0, M = 0;                   // Grid dimensions
    double x_max = 0.0, y_max = 0.0;    // Total grid size
    int max_steps;                      // Total simulation steps
    char* input_file;                   // Input file name
    int check;                          // 0: no check, 1: check the simulation is correct

    int mode = 0;
    if (argc == 5) {
        input_file = argv[1];
        max_steps = atoi(argv[2]);
        if (max_steps <= 0) {
            fprintf(stderr, "max_steps needs to be a positive integer\n");
            return 1;
        }
        mode = atoi(argv[3]);
        if (mode > 2 || mode < 0) {
            fprintf(stderr, "mode needs to be a value between 0 and 2\n");
            return 1;
        }
        check = atoi(argv[4]);
        if (check >= 2 || check < 0) {
            fprintf(stderr, "check needs to be a 0 or 1\n");
            return 1;
        }
    }
    else {
        fprintf(stderr, "Usage: %s <filename> <max_steps> <mode> <check>\n", argv[0]);
        return 1;
    }

    // Read the planes from the input file and copy it for checking.
    PlaneList owning_planes = {NULL, NULL};
    read_planes_seq(input_file, &owning_planes, &N, &M, &x_max, &y_max);
    PlaneList owning_planes_t0 = copy_plane_list(&owning_planes);

    int step = 0;

    clock_t start_time = clock();
    double comp_time = 0.;
    for (step = 1; step <= max_steps; step++) {
        clock_t start_comp = clock();

        // Simulation loop
        PlaneNode* current = owning_planes.head;
        while (current != NULL) {
            current->x += current->vx;
            current->y += current->vy;
            current = current->next;
        }

        filter_planes(&owning_planes, x_max, y_max);
        clock_t end_comp = clock();
        comp_time += ((double)(end_comp - start_comp)) / CLOCKS_PER_SEC;
    }

    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Flight controller simulation: #input %s mode: %d\n", input_file, mode);
    printf("Time: comp: %.2fs    total: %.2lfs\n", comp_time, time_taken);

    if (check == 1)
        check_planes_seq(&owning_planes_t0, &owning_planes, N, M, x_max, y_max, max_steps, debug);
    return 0;
}