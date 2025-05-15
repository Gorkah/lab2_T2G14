#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <stddef.h>

#include "auxiliar.h"

/// Reading the planes from a file for MPI
void read_planes_mpi(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max, int rank, int size, int* tile_displacements)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    
    // Calculate tile displacements - distribute tiles evenly among ranks
    int total_tiles = (*N) * (*M);
    int base_tiles_per_rank = total_tiles / size;
    int remaining_tiles = total_tiles % size;
    
    tile_displacements[0] = 0;
    for (int i = 0; i < size; i++) {
        int tiles_for_this_rank = base_tiles_per_rank + (i < remaining_tiles ? 1 : 0);
        tile_displacements[i+1] = tile_displacements[i] + tiles_for_this_rank;
    }
    
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
            
            // Determine which rank this plane belongs to
            int plane_rank = get_rank_from_index(index_map, tile_displacements, size);
            
            // Add plane to list only if it belongs to current rank
            if (plane_rank == rank) {
                insert_plane(planes, idx, index_map, rank, x, y, vx, vy);
            }
            index++;
        }
    }
    fclose(file);

    // Use MPI_Allreduce to get the total number of planes read by all ranks
    int my_planes = 0;
    PlaneNode* current = planes->head;
    while (current != NULL) {
        my_planes++;
        current = current->next;
    }

    int total_planes = 0;
    MPI_Allreduce(&my_planes, &total_planes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total planes read: %d\n", total_planes);
        assert(num_planes == total_planes);
    }
}

/// Communicate planes using mainly Send/Recv calls with default data types
void communicate_planes_send(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    PlaneNode* current = list->head;
    PlaneNode* next = NULL;
    
    // Create temporary lists of planes to be sent to each rank
    PlaneList* send_lists = (PlaneList*)malloc(size * sizeof(PlaneList));
    for (int i = 0; i < size; i++) {
        send_lists[i].head = NULL;
        send_lists[i].tail = NULL;
    }
    
    // Check each plane and move it to the appropriate send list if needed
    while (current != NULL) {
        next = current->next;
        
        // Recalculate grid indices based on new position
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int index_map = get_index(index_i, index_j, N, M);
        
        // Check if the plane has moved to a different rank's territory
        int target_rank = get_rank_from_index(index_map, tile_displacements, size);
        
        if (target_rank != rank) {
            // Update index_map
            current->index_map = index_map;
            
            // Remove from current list
            remove_plane(list, current);
            
            // Add to send list for target_rank
            insert_plane(&send_lists[target_rank], current->index_plane, 
                         index_map, target_rank, current->x, current->y, 
                         current->vx, current->vy);
        }
        
        current = next;
    }
    
    // For each rank, count planes to send
    int* send_counts = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        send_counts[i] = 0;
        current = send_lists[i].head;
        while (current != NULL) {
            send_counts[i]++;
            current = current->next;
        }
    }
    
    // First exchange counts to know how many planes to expect
    int* recv_counts = (int*)malloc(size * sizeof(int));
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Now send planes (we pack each plane as an array of 5 doubles: index_map, x, y, vx, vy)
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            // Send planes to rank i
            if (send_counts[i] > 0) {
                // Pack planes into buffer
                double* send_buffer = (double*)malloc(send_counts[i] * 5 * sizeof(double));
                current = send_lists[i].head;
                int idx = 0;
                while (current != NULL) {
                    send_buffer[idx++] = (double)current->index_plane;
                    send_buffer[idx++] = current->x;
                    send_buffer[idx++] = current->y;
                    send_buffer[idx++] = current->vx;
                    send_buffer[idx++] = current->vy;
                    current = current->next;
                }
                
                // Send buffer
                MPI_Send(send_buffer, send_counts[i] * 5, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                free(send_buffer);
            }
            
            // Receive planes from rank i
            if (recv_counts[i] > 0) {
                // Allocate buffer for receiving
                double* recv_buffer = (double*)malloc(recv_counts[i] * 5 * sizeof(double));
                
                // Receive data
                MPI_Status status;
                MPI_Recv(recv_buffer, recv_counts[i] * 5, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                
                // Unpack planes and add to list
                for (int j = 0; j < recv_counts[i]; j++) {
                    int idx_plane = (int)recv_buffer[j * 5];
                    double x = recv_buffer[j * 5 + 1];
                    double y = recv_buffer[j * 5 + 2];
                    double vx = recv_buffer[j * 5 + 3];
                    double vy = recv_buffer[j * 5 + 4];
                    
                    // Calculate new map index
                    int idx_i = get_index_i(x, x_max, N);
                    int idx_j = get_index_j(y, y_max, M);
                    int idx_map = get_index(idx_i, idx_j, N, M);
                    
                    // Add to local list
                    insert_plane(list, idx_plane, idx_map, rank, x, y, vx, vy);
                }
                
                free(recv_buffer);
            }
        }
    }
    
    // Clean up temporary lists
    for (int i = 0; i < size; i++) {
        current = send_lists[i].head;
        while (current != NULL) {
            next = current->next;
            free(current);
            current = next;
        }
    }
    
    free(send_lists);
    free(send_counts);
    free(recv_counts);
}

/// Communicate planes using all to all calls with default data types
void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    // Create temporary lists of planes to be sent to each rank
    PlaneList* send_lists = (PlaneList*)malloc(size * sizeof(PlaneList));
    for (int i = 0; i < size; i++) {
        send_lists[i].head = NULL;
        send_lists[i].tail = NULL;
    }
    
    // Check each plane and classify it for the appropriate rank
    PlaneNode* current = list->head;
    PlaneNode* next = NULL;
    
    while (current != NULL) {
        next = current->next;
        
        // Recalculate grid indices based on new position
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int index_map = get_index(index_i, index_j, N, M);
        
        // Check if the plane has moved to a different rank's territory
        int target_rank = get_rank_from_index(index_map, tile_displacements, size);
        
        if (target_rank != rank) {
            // Update index_map
            current->index_map = index_map;
            
            // Remove from current list
            remove_plane(list, current);
            
            // Add to send list for target_rank
            insert_plane(&send_lists[target_rank], current->index_plane, 
                         index_map, target_rank, current->x, current->y, 
                         current->vx, current->vy);
        }
        
        current = next;
    }
    
    // Count planes to send to each rank
    int* send_counts = (int*)malloc(size * sizeof(int));
    int* send_displs = (int*)malloc(size * sizeof(int));
    int total_planes_to_send = 0;
    
    for (int i = 0; i < size; i++) {
        send_counts[i] = 0;
        current = send_lists[i].head;
        while (current != NULL) {
            send_counts[i]++;
            current = current->next;
        }
        send_displs[i] = total_planes_to_send;
        total_planes_to_send += send_counts[i];
    }
    
    // Exchange counts to know how many planes to expect from each rank
    int* recv_counts = (int*)malloc(size * sizeof(int));
    int* recv_displs = (int*)malloc(size * sizeof(int));
    
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements for receiving
    int total_planes_to_recv = 0;
    for (int i = 0; i < size; i++) {
        recv_displs[i] = total_planes_to_recv;
        total_planes_to_recv += recv_counts[i];
    }
    
    // Prepare send buffer - 5 doubles per plane: index_plane, x, y, vx, vy
    double* send_buffer = NULL;
    if (total_planes_to_send > 0) {
        send_buffer = (double*)malloc(total_planes_to_send * 5 * sizeof(double));
    }
    
    // Pack planes into send buffer
    for (int i = 0; i < size; i++) {
        current = send_lists[i].head;
        int buffer_idx = send_displs[i] * 5;
        
        while (current != NULL) {
            send_buffer[buffer_idx++] = (double)current->index_plane;
            send_buffer[buffer_idx++] = current->x;
            send_buffer[buffer_idx++] = current->y;
            send_buffer[buffer_idx++] = current->vx;
            send_buffer[buffer_idx++] = current->vy;
            current = current->next;
        }
    }
    
    // Prepare receive buffer
    double* recv_buffer = NULL;
    if (total_planes_to_recv > 0) {
        recv_buffer = (double*)malloc(total_planes_to_recv * 5 * sizeof(double));
    }
    
    // Convert counts to elements (each plane = 5 doubles)
    int* send_counts_elements = (int*)malloc(size * sizeof(int));
    int* send_displs_elements = (int*)malloc(size * sizeof(int));
    int* recv_counts_elements = (int*)malloc(size * sizeof(int));
    int* recv_displs_elements = (int*)malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        send_counts_elements[i] = send_counts[i] * 5;
        send_displs_elements[i] = send_displs[i] * 5;
        recv_counts_elements[i] = recv_counts[i] * 5;
        recv_displs_elements[i] = recv_displs[i] * 5;
    }
    
    // Perform all-to-all communication
    MPI_Alltoallv(send_buffer, send_counts_elements, send_displs_elements, MPI_DOUBLE,
                  recv_buffer, recv_counts_elements, recv_displs_elements, MPI_DOUBLE,
                  MPI_COMM_WORLD);
    
    // Process received planes
    for (int i = 0; i < total_planes_to_recv; i++) {
        int idx_plane = (int)recv_buffer[i * 5];
        double x = recv_buffer[i * 5 + 1];
        double y = recv_buffer[i * 5 + 2];
        double vx = recv_buffer[i * 5 + 3];
        double vy = recv_buffer[i * 5 + 4];
        
        // Calculate new map index
        int idx_i = get_index_i(x, x_max, N);
        int idx_j = get_index_j(y, y_max, M);
        int idx_map = get_index(idx_i, idx_j, N, M);
        
        // Add to local list
        insert_plane(list, idx_plane, idx_map, rank, x, y, vx, vy);
    }
    
    // Clean up
    for (int i = 0; i < size; i++) {
        current = send_lists[i].head;
        while (current != NULL) {
            next = current->next;
            free(current);
            current = next;
        }
    }
    
    free(send_lists);
    free(send_counts);
    free(send_displs);
    free(recv_counts);
    free(recv_displs);
    free(send_counts_elements);
    free(send_displs_elements);
    free(recv_counts_elements);
    free(recv_displs_elements);
    
    if (send_buffer) free(send_buffer);
    if (recv_buffer) free(recv_buffer);
}

typedef struct {
    int    index_plane;
    double x;
    double y;
    double vx;
    double vy;
} MinPlaneToSend;

/// Communicate planes using all to all calls with custom data types
void communicate_planes_struct_mpi(PlaneList* list,
                               int N, int M,
                               double x_max, double y_max,
                               int rank, int size,
                               int* tile_displacements)
{
    // Create MPI datatype for MinPlaneToSend
    MPI_Datatype plane_type;
    int blocklengths[5] = {1, 1, 1, 1, 1}; // One element for each field
    
    // Calculate offsets for each field in the MinPlaneToSend struct
    MPI_Aint offsets[5];
    offsets[0] = offsetof(MinPlaneToSend, index_plane);
    offsets[1] = offsetof(MinPlaneToSend, x);
    offsets[2] = offsetof(MinPlaneToSend, y);
    offsets[3] = offsetof(MinPlaneToSend, vx);
    offsets[4] = offsetof(MinPlaneToSend, vy);
    
    // Define MPI types for each field
    MPI_Datatype types[5] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    
    // Create and commit the custom MPI datatype
    MPI_Type_create_struct(5, blocklengths, offsets, types, &plane_type);
    MPI_Type_commit(&plane_type);
    
    // Create temporary lists of planes to be sent to each rank
    PlaneList* send_lists = (PlaneList*)malloc(size * sizeof(PlaneList));
    for (int i = 0; i < size; i++) {
        send_lists[i].head = NULL;
        send_lists[i].tail = NULL;
    }
    
    // Classify planes by destination rank
    PlaneNode* current = list->head;
    PlaneNode* next = NULL;
    
    while (current != NULL) {
        next = current->next;
        
        // Recalculate grid indices based on new position
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int index_map = get_index(index_i, index_j, N, M);
        
        // Check if the plane has moved to a different rank's territory
        int target_rank = get_rank_from_index(index_map, tile_displacements, size);
        
        if (target_rank != rank) {
            // Update index_map
            current->index_map = index_map;
            
            // Remove from current list
            remove_plane(list, current);
            
            // Add to send list for target_rank
            insert_plane(&send_lists[target_rank], current->index_plane, 
                         index_map, target_rank, current->x, current->y, 
                         current->vx, current->vy);
        }
        
        current = next;
    }
    
    // Count planes to send to each rank
    int* send_counts = (int*)malloc(size * sizeof(int));
    int* send_displs = (int*)malloc(size * sizeof(int));
    int total_planes_to_send = 0;
    
    for (int i = 0; i < size; i++) {
        send_counts[i] = 0;
        current = send_lists[i].head;
        while (current != NULL) {
            send_counts[i]++;
            current = current->next;
        }
        send_displs[i] = total_planes_to_send;
        total_planes_to_send += send_counts[i];
    }
    
    // Exchange counts to know how many planes to expect from each rank
    int* recv_counts = (int*)malloc(size * sizeof(int));
    int* recv_displs = (int*)malloc(size * sizeof(int));
    
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements for receiving
    int total_planes_to_recv = 0;
    for (int i = 0; i < size; i++) {
        recv_displs[i] = total_planes_to_recv;
        total_planes_to_recv += recv_counts[i];
    }
    
    // Prepare send buffer
    MinPlaneToSend* send_buffer = NULL;
    if (total_planes_to_send > 0) {
        send_buffer = (MinPlaneToSend*)malloc(total_planes_to_send * sizeof(MinPlaneToSend));
    }
    
    // Pack planes into send buffer
    for (int i = 0; i < size; i++) {
        current = send_lists[i].head;
        int buffer_idx = send_displs[i];
        
        while (current != NULL) {
            send_buffer[buffer_idx].index_plane = current->index_plane;
            send_buffer[buffer_idx].x = current->x;
            send_buffer[buffer_idx].y = current->y;
            send_buffer[buffer_idx].vx = current->vx;
            send_buffer[buffer_idx].vy = current->vy;
            buffer_idx++;
            current = current->next;
        }
    }
    
    // Prepare receive buffer
    MinPlaneToSend* recv_buffer = NULL;
    if (total_planes_to_recv > 0) {
        recv_buffer = (MinPlaneToSend*)malloc(total_planes_to_recv * sizeof(MinPlaneToSend));
    }
    
    // Perform all-to-all communication with custom datatype
    MPI_Alltoallv(send_buffer, send_counts, send_displs, plane_type,
                  recv_buffer, recv_counts, recv_displs, plane_type,
                  MPI_COMM_WORLD);
    
    // Process received planes
    for (int i = 0; i < total_planes_to_recv; i++) {
        int idx_plane = recv_buffer[i].index_plane;
        double x = recv_buffer[i].x;
        double y = recv_buffer[i].y;
        double vx = recv_buffer[i].vx;
        double vy = recv_buffer[i].vy;
        
        // Calculate new map index
        int idx_i = get_index_i(x, x_max, N);
        int idx_j = get_index_j(y, y_max, M);
        int idx_map = get_index(idx_i, idx_j, N, M);
        
        // Add to local list
        insert_plane(list, idx_plane, idx_map, rank, x, y, vx, vy);
    }
    
    // Clean up
    for (int i = 0; i < size; i++) {
        current = send_lists[i].head;
        while (current != NULL) {
            next = current->next;
            free(current);
            current = next;
        }
    }
    
    MPI_Type_free(&plane_type);
    
    free(send_lists);
    free(send_counts);
    free(send_displs);
    free(recv_counts);
    free(recv_displs);
    
    if (send_buffer) free(send_buffer);
    if (recv_buffer) free(recv_buffer);
}

int main(int argc, char **argv) {
    int debug = 0;                      // 0: no debug, 1: shows all planes information during checking
    int N = 0, M = 0;                   // Grid dimensions
    double x_max = 0.0, y_max = 0.0;    // Total grid size
    int max_steps;                      // Total simulation steps
    char* input_file;                   // Input file name
    int check;                          // 0: no check, 1: check the simulation is correct

    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* tile_displacements = (int*)malloc((size+1) * sizeof(int));
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

    PlaneList owning_planes = {NULL, NULL};
    read_planes_mpi(input_file, &owning_planes, &N, &M, &x_max, &y_max, rank, size, tile_displacements);
    PlaneList owning_planes_t0 = copy_plane_list(&owning_planes);

    //print_planes_par_debug(&owning_planes);

    double time_sim = 0., time_comm = 0, time_total=0;

    double start_time = MPI_Wtime();
    int step = 0;
    for (step = 1; step <= max_steps; step++) {
        double start = MPI_Wtime();
        PlaneNode* current = owning_planes.head;
        while (current != NULL) {
            current->x += current->vx;
            current->y += current->vy;
            current = current->next;
        }
        filter_planes(&owning_planes, x_max, y_max);
        time_sim += MPI_Wtime() - start;

        start = MPI_Wtime();
        if (mode == 0)
            communicate_planes_send(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else if (mode == 1)
            communicate_planes_alltoall(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else
            communicate_planes_struct_mpi(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        time_comm += MPI_Wtime() - start;
    }
    time_total = MPI_Wtime() - start_time;

    // Calculate max time across all ranks for consistent reporting
    double max_time_sim = 0.0;
    double max_time_comm = 0.0;
    double max_time_total = 0.0;
    
    MPI_Reduce(&time_sim, &max_time_sim, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_comm, &max_time_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_total, &max_time_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("Flight controller simulation: #input %s mode: %d size: %d\n", input_file, mode, size);
        printf("Time simulation:     %.2fs\n", time_sim);
        printf("Time communication:  %.2fs\n", time_comm);
        printf("Time total:          %.2fs\n", time_total);
    }

    if (check ==1)
        check_planes_mpi(&owning_planes_t0, &owning_planes, N, M, x_max, y_max, max_steps, tile_displacements, size, debug);

    MPI_Finalize();
    return 0;
}
