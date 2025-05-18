#include "auxiliar.h"

void output_planes_seq(const char* filename, PlaneList* list, int N, int M, double x_max, double y_max)
{
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file");
        return;
    }

    fprintf(file, "# Plane Data\n");
    fprintf(file, "# Map: %.6lf, %.6lf : %d %d\n", x_max, y_max, N, M);
    fprintf(file, "# Number of Planes: X\n");
    fprintf(file, "# idx x y vx vy\n");

    PlaneNode* current = list->head;
    while (current != NULL) {
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);

        fprintf(file, "%d %.8f %.8f %.8f %.8f %d %d %d %d\n", current->index_plane, current->x, current->y, current->vx, current->vy, index_i, index_j, current->index_map, current->rank);
        current = current->next;
    }
    fclose(file);
}

void filter_planes(PlaneList* list, double x_max, double y_max)
{
    PlaneNode* current = list->head;
    while (current != NULL) {
        if (current->x <= 1e-3 || current->x >= (x_max-1e-3) || current->y <=  1e-3 || current->y >= (y_max-1e-3)) {
            PlaneNode* next = current->next;
            if (current->prev != NULL) {
                current->prev->next = current->next;
            } else {
                list->head = current->next;
            }
            if (current->next != NULL) {
                current->next->prev = current->prev;
            } else {
                list->tail = current->prev;
            }
            free(current);
            current = next;
        } else {
            current = current->next;
        }
    }
}

void insert_plane(struct PlaneList* list, int index_plane, int index_map, int rank, double x, double y, double vx, double vy)
{
    struct PlaneNode* new_node = (struct PlaneNode*) malloc(sizeof(struct PlaneNode));
    new_node->index_plane = index_plane;
    new_node->index_map = index_map;
    new_node->rank = rank;
    new_node->x = x;
    new_node->y = y;
    new_node->vx = vx;
    new_node->vy = vy;
    new_node->next = NULL;

    if (list->head == NULL) {
        list->head = new_node;
        list->tail = new_node;
        new_node->prev = NULL;
    } else {
        list->tail->next = new_node;
        new_node->prev = list->tail;
        list->tail = new_node;
    }
}

void remove_plane(PlaneList* list, PlaneNode* node)
{
    if (node->prev == NULL) {
        list->head = node->next;
    } else {
        node->prev->next = node->next;
    }

    if (node->next == NULL) {
        list->tail = node->prev;
    } else {
        node->next->prev = node->prev;
    }

    free(node);
}

PlaneNode* seek_plane(PlaneList* list, int index_plane)
{
    PlaneNode* current = list->head;
    while (current != NULL) {
        if (current->index_plane == index_plane) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

PlaneList copy_plane_list(PlaneList* list) {
    PlaneList new_list = {NULL, NULL};
    PlaneNode* current = list->head;
    while (current != NULL) {
        insert_plane(&new_list, current->index_plane, current->index_map, current->rank, current->x, current->y, current->vx, current->vy);
        current = current->next;
    }
    return new_list;
}

void print_planes_debug(PlaneList* list)
{
    PlaneNode* current = list->head;
    while (current != NULL) {
        printf("Plane %d: (%.2f, %.2f) -> (%.2f, %.2f) : [%d, %d]\n", current->index_plane, current->x, current->y, current->vx, current->vy, current->index_map, current->rank);
        current = current->next;
    }
}

int check_planes_internal(PlaneList* owning_planes_t0, PlaneList* owning_planes, int N, int M, double x_max, double y_max, int max_steps, int* tile_displacements, int size, int debug)
{
    PlaneNode* current_t0 = owning_planes_t0->head;
    while (current_t0 != NULL) {
        current_t0->x = current_t0->x + current_t0->vx*(double)max_steps;
        current_t0->y = current_t0->y + current_t0->vy*(double)max_steps;
        int index_i = get_index_i(current_t0->x, x_max, N);
        int index_j = get_index_j(current_t0->y, y_max, M);
        current_t0->index_map = get_index(index_i, index_j, N, M);
        if (current_t0->index_map >= 0 && current_t0->index_map < N*M)
            current_t0->rank = tile_displacements == NULL || size <= 1 ? 0 : get_rank_from_index(current_t0->index_map, tile_displacements, size);
        current_t0 = current_t0->next;
    }

    int error = 0;
    current_t0 = owning_planes_t0->head;
    while (current_t0 != NULL) {
        double current_x = current_t0->x;
        double current_y = current_t0->y;

        PlaneNode* found = seek_plane(owning_planes, current_t0->index_plane);

        if ((current_x <= 1e-3 || current_x >= (x_max-1e-3) || current_y <= 1e-3 || current_y >= (y_max-1e-3)) && found == NULL) {
            if(debug)
                printf("Ok! Plane %d out of bounds at step %d\n", current_t0->index_plane, max_steps);
        }
        else if ((current_x <= 0 || current_x >= x_max || current_y <= 0 || current_y >= y_max) && found != NULL) {
            printf("Missing Plane %d! It should be out of bounds at step %d\n", current_t0->index_plane, max_steps);
            printf("   State:    (%.2f, %.2f) (%d, %d) %d %d, %d\n", found->x, found->y,
                   get_index_i(found->x, x_max, N),
                   get_index_j(found->x, y_max, M),
                   get_index(get_index_i(found->x, x_max, N), get_index_j(found->y, y_max, M), N, M),
                   found->index_map, found->rank);
            printf("   Expected: (%.2f, %.2f) (%d, %d) %d %d, %d\n", current_t0->x, current_t0->y,
                   get_index_i(current_t0->x, x_max, N),
                   get_index_j(current_t0->x, y_max, M),
                   get_index(get_index_i(current_t0->x, x_max, N), get_index_j(found->y, y_max, M), N, M),
                   current_t0->index_map, current_t0->rank);
            error++;
        }
        else if (found == NULL) {
            printf("Missing Plane %d! The plane should be inside the map, but it could not be found.\n", current_t0->index_plane);
            printf("   Expected: (%.2f, %.2f) %d, %d\n", current_t0->x, current_t0->y, current_t0->index_map, current_t0->rank);
            error++;
        }
        else if (fabs(current_x-found->x) > 1e-6 || fabs(current_y-found->y) > 1e-6 ) {
            printf("Missing Plane %d! It should be at (%.5f, %.5f) at step %d, but it is at (%.5f, %.5f)\n", current_t0->index_plane, current_x, current_y, max_steps, found->x, found->y);
            printf("   State:    (%.2f, %.2f) (%d, %d) %d %d, %d\n", found->x, found->y,
                   get_index_i(found->x, x_max, N),
                   get_index_j(found->x, y_max, M),
                   get_index(get_index_i(found->x, x_max, N), get_index_j(found->y, y_max, M), N, M),
                   found->index_map, found->rank);
            printf("   Expected: (%.2f, %.2f) (%d, %d) %d %d, %d\n", current_t0->x, current_t0->y,
                   get_index_i(current_t0->x, x_max, N),
                   get_index_j(current_t0->x, y_max, M),
                   get_index(get_index_i(current_t0->x, x_max, N), get_index_j(found->y, y_max, M), N, M),
                   current_t0->index_map, current_t0->rank);
            error++;
        }
        else if (current_t0->rank != found->rank) {
            printf("Missing Plane %d! It should be at rank %d at step %d, but it is at rank %d\n", current_t0->index_plane, current_t0->rank, max_steps, found->rank);
            printf("   State:    (%.2f, %.2f) (%d, %d) %d %d, %d\n", found->x, found->y,
                   get_index_i(found->x, x_max, N),
                   get_index_j(found->x, y_max, M),
                   get_index(get_index_i(found->x, x_max, N), get_index_j(found->y, y_max, M), N, M),
                   found->index_map, found->rank);
            printf("   Expected: (%.2f, %.2f) (%d, %d) %d %d, %d\n", current_t0->x, current_t0->y,
                   get_index_i(current_t0->x, x_max, N),
                   get_index_j(current_t0->x, y_max, M),
                   get_index(get_index_i(current_t0->x, x_max, N), get_index_j(found->y, y_max, M), N, M),
                   current_t0->index_map, current_t0->rank);
            error++;
        }
        else {
            if (debug)
                printf("Ok! Plane %d found at (%.2f, %.2f) at step %d\n", current_t0->index_plane, current_x, current_y, max_steps);
        }

        current_t0 = current_t0->next;
    }

    return error;
}

int check_planes_seq(PlaneList* owning_planes_t0, PlaneList* owning_planes, int N, int M, double x_max, double y_max, int max_steps, int debug)
{
    return check_planes_internal(owning_planes_t0, owning_planes, N, M, x_max, y_max, max_steps, NULL, 0, debug);
}

void read_planes_seq_full(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    char line[MAX_LINE_LENGTH];
    int num_planes = 0;

    // Read header
    fgets(line, sizeof(line), file);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Map: %lf, %lf : %d %d", &(*x_max), &(*y_max), N, M);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Number of Planes: %d", &num_planes);
    fgets(line, sizeof(line), file);

    // Read grid data
    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        int idx, index_i, index_j, index_map, rank;
        double x, y, vx, vy;
        if (sscanf(line, "%d %lf %lf %lf %lf %d %d %d %d",
                   &idx, &x, &y, &vx, &vy, &index_i, &index_j, &index_map, &rank) == 9) {
            insert_plane(planes, idx, index_map, rank, x, y, vx, vy);
            index++;
        }
    }

    fclose(file);
}

#include <mpi.h>
int check_planes_mpi(PlaneList* owning_planes_t0, PlaneList* owning_planes, int N, int M, double x_max, double y_max, int max_steps, int* tile_displacements, int size, int debug)
{
    char filename_comp[256];
    sprintf(filename_comp, "planes_comp.txt");
    char filename_check[256];
    sprintf(filename_check, "planes_check.txt");

    output_planes_par_debug(filename_comp, owning_planes, N, M, x_max, y_max);
    output_planes_par_debug(filename_check, owning_planes_t0, N, M, x_max, y_max);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int error = 0;
    if (rank == 0)
    {
        PlaneList r0_owning_planes = {NULL, NULL};
        read_planes_seq_full(filename_comp, &r0_owning_planes, &N, &M, &x_max, &y_max);
        PlaneList r0_owning_planes_t0 = {NULL, NULL};
        read_planes_seq_full(filename_check, &r0_owning_planes_t0, &N, &M, &x_max, &y_max);

        error = check_planes_internal(&r0_owning_planes_t0, &r0_owning_planes, N, M, x_max, y_max, max_steps, tile_displacements, size, debug);
    }
    MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return error;
}

void output_planes_par_debug(const char* filename, PlaneList* list, int N, int M, double x_max, double y_max)
{
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        FILE *file = fopen(filename, "w");
        if (!file) {
            perror("Error opening file");
            return;
        }

        fprintf(file, "# Plane Data\n");
        fprintf(file, "# Map: %.2lf, %.2lf : %d %d\n", x_max, y_max, N, M);
        fprintf(file, "# Number of Planes: X\n");
        fprintf(file, "# idx x y vx vy\n");
        fclose(file);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < size; i++)
    {
        if (rank == i)
        {
            FILE *file = fopen(filename, "a");
            if (!file) {
                perror("Error opening file");
                return;
            }

            PlaneNode* current = list->head;
            while (current != NULL) {
                int index_i = get_index_i(current->x, x_max, N);
                int index_j = get_index_j(current->y, y_max, M);

                fprintf(file, "%d %.8f %.8f %.8f %.8f %d %d %d %d\n", current->index_plane, current->x, current->y, current->vx, current->vy, index_i, index_j, current->index_map, current->rank);
                current = current->next;
            }
            fclose(file);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_planes_par_debug(PlaneList* list)
{
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int p = 0; p < size; p++)
    {
        if (rank == p) {
            PlaneNode* current = list->head;
            printf("# Rank %d\n", rank);

            while (current != NULL) {
                printf("  Plane %d: (%.2f, %.2f) -> (%.2f, %.2f) : [%d, %d]\n", current->index_plane, current->x, current->y, current->vx, current->vy, current->index_map, current->rank);
                current = current->next;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

}


