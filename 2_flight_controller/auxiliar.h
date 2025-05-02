#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

# define MAX_LINE_LENGTH 256

/// Function to get the grid index i based on a given x coordinate
static int get_index_i(double x, double max_x, int N)
{
    return (int) (x / (max_x) * N);
}

/// Function to get the grid index j based on a given y coordinate
static int get_index_j(double y, double max_y, int M)
{
    return (int) (y / (max_y) * M);
}

/// Function to get the total grid index based on i and j coordinates
static int get_index(int i, int j, int N, int M)
{
    return j * N + i;
}

#include "mpi.h"
/// Function to get the rank from a given index based on the tile displacements array
static int get_rank_from_index(int index, const int* tile_displacements, int size)
{
    for (int rank = 0; rank < size; rank++) {
        if (index < tile_displacements[rank+1]) {
            return rank;
        }
    }

    printf("Error: index %d not found in tile_displacements: %d %d %d\n", index, tile_displacements[0], tile_displacements[1], tile_displacements[2]);
    return -1;
}

/// Function to get the rank from a given indices i,j based on the tile displacements array
static int get_rank_from_indices(int i, int j, int N, int M, const int* tile_displacements, int size)
{
    int index = get_index(i, j, N, M);
    if(index > N*M) {
        printf("Error: index %d,%d [%d,%d]\n", i, j, N, M);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return get_rank_from_index(index, tile_displacements, size);
}

/// Double linked list node for a plane
typedef struct PlaneNode {
    int index_plane;
    int index_map;
    int rank;
    double x;
    double y;
    double vx;
    double vy;
    struct PlaneNode* next;
    struct PlaneNode* prev;
} PlaneNode;

/// Double linked list for planes
typedef struct PlaneList {
    PlaneNode* head;
    PlaneNode* tail;
} PlaneList;

/// Function to outputs the planes (recommended only for checking with small number of planes)
void print_planes_par_debug(PlaneList* list);

/// Insert a new plane node at the end of the list
void insert_plane(struct PlaneList* list, int index_plane, int index_map, int rank, double x, double y, double vx, double vy);

/// Remove a plane node from the list
void remove_plane(PlaneList* list, PlaneNode* node);

/// Seek for plane with a given index
PlaneNode* seek_plane(PlaneList* list, int index_plane);

/// Print the list of planes
void print_planes_debug(PlaneList* list);

/// Remove from the list the planes that are outside of the map
void filter_planes(PlaneList* list, double x_max, double y_max);

/// Copy the plane list
PlaneList copy_plane_list(PlaneList* list);

/// Function that checks that the evolution of the planes is correct (sequential)
int check_planes_seq(PlaneList* owning_planes_t0, PlaneList* owning_planes, int N, int M, double x_max, double y_max, int max_steps, int debug);

/// Function that checks that the evolution of the planes is correct (MPI)
int check_planes_mpi(PlaneList* owning_planes_t0, PlaneList* owning_planes, int N, int M, double x_max, double y_max, int max_steps, int* tile_displacements, int size, int debug);

/// Function that checks that the evolution of the planes is correct (internal)
int check_planes_internal(PlaneList* owning_planes_t0, PlaneList* owning_planes, int N, int M, double x_max, double y_max, int max_steps, int* tile_displacements, int size, int debug);

/// Function to read the planes from a file (not optimized, only for checking with small number of planes)
void read_planes_seq_full(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max);

/// Function to outputs the planes to a file (sequential)
void output_planes_seq(const char* filename, PlaneList* list, int N, int M, double x_max, double y_max);

/// Function to outputs the planes to a file (not optimized, recommended only for checking with small number of planes)
void output_planes_par_debug(const char* filename, PlaneList* list, int N, int M, double x_max, double y_max);

