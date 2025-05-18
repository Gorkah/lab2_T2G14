#include <stdio.h>

#include <assert.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

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
    
    // Compute tile displacements - distribute tiles to ranks as evenly as possible
    int total_tiles = (*N) * (*M);
    int base_tiles_per_rank = total_tiles / size;
    int extra_tiles = total_tiles % size;
    
    tile_displacements[0] = 0;
    for (int i = 0; i < size; i++) {
        int tiles_for_rank = base_tiles_per_rank + (i < extra_tiles ? 1 : 0);
        tile_displacements[i+1] = tile_displacements[i] + tiles_for_rank;
    }
    
    // Reading plane data
    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        int idx;
        double x, y, vx, vy;
        if (sscanf(line, "%d %lf %lf %lf %lf", &idx, &x, &y, &vx, &vy) == 5) {
            int index_i = get_index_i(x, *x_max, *N);
            int index_j = get_index_j(y, *y_max, *M);
            int index_map = get_index(index_i, index_j, *N, *M);
            
            // Check if this plane belongs to the current rank
            int plane_rank = get_rank_from_index(index_map, tile_displacements, size);
            
            if (plane_rank == rank) {
                insert_plane(planes, idx, index_map, rank, x, y, vx, vy);
            }
            
            index++;
        }
    }
    fclose(file);

    // Verify all processes have read the same total number of planes
    int total_planes_read;
    MPI_Allreduce(&index, &total_planes_read, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Total planes read: %d\n", index);
        if (total_planes_read != index * size) {
            fprintf(stderr, "Error: Not all processes read the same number of planes\n");
        }
    }
}

/// Communicate planes using mainly Send/Recv calls with default data types
void communicate_planes_send(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    // Define array to hold planes that need to be sent to each rank
    PlaneList to_send[size];
    for (int i = 0; i < size; i++) {
        to_send[i].head = NULL;
        to_send[i].tail = NULL;
    }
    
    // Check each plane's position and determine if it needs to be sent to another rank
    PlaneNode* current = list->head;
    while (current != NULL) {
        PlaneNode* next = current->next; // Save next pointer before potentially removing this node
        
        // Calculate new grid indices based on plane's current position
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int index_map = get_index(index_i, index_j, N, M);
        
        // Check if plane has moved to a different region
        int new_rank = get_rank_from_index(index_map, tile_displacements, size);
        
        if (new_rank != rank) {
            // Add to list of planes to send to the new rank
            insert_plane(&to_send[new_rank], current->index_plane, index_map, rank, 
                         current->x, current->y, current->vx, current->vy);
            
            // Remove from current list
            remove_plane(list, current);
        } else {
            // Update the plane's map index if it stayed in the same rank but moved
            current->index_map = index_map;
        }
        
        current = next;
    }
    
    // Array to store how many planes to send to each rank
    int send_counts[size];
    for (int i = 0; i < size; i++) {
        // Count planes to send to rank i
        int count = 0;
        PlaneNode* node = to_send[i].head;
        while (node != NULL) {
            count++;
            node = node->next;
        }
        send_counts[i] = count;
    }
    
    // Use a collective operation to share how many planes each rank will send
    int recv_counts[size];
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Send and receive planes
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            // Send planes to rank i
            if (send_counts[i] > 0) {
                PlaneNode* node = to_send[i].head;
                int tag = rank * size + i;
                
                while (node != NULL) {
                    // Pack data into array: [index_plane, index_map, x, y, vx, vy]
                    double plane_data[6];
                    plane_data[0] = (double)node->index_plane;
                    plane_data[1] = (double)node->index_map;
                    plane_data[2] = node->x;
                    plane_data[3] = node->y;
                    plane_data[4] = node->vx;
                    plane_data[5] = node->vy;
                    
                    // Send the plane data
                    MPI_Send(plane_data, 6, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
                    
                    node = node->next;
                }
            }
            
            // Receive planes from rank i
            if (recv_counts[i] > 0) {
                int tag = i * size + rank;
                
                for (int j = 0; j < recv_counts[i]; j++) {
                    double plane_data[6];
                    MPI_Status status;
                    
                    // Receive the plane data
                    MPI_Recv(plane_data, 6, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
                    
                    // Extract data and add plane to list
                    int index_plane = (int)plane_data[0];
                    int index_map = (int)plane_data[1];
                    double x = plane_data[2];
                    double y = plane_data[3];
                    double vx = plane_data[4];
                    double vy = plane_data[5];
                    
                    insert_plane(list, index_plane, index_map, rank, x, y, vx, vy);
                }
            }
        }
    }
    
    // Clean up the temporary lists
    for (int i = 0; i < size; i++) {
        // Free memory for all nodes in to_send[i]
        PlaneNode* node = to_send[i].head;
        while (node != NULL) {
            PlaneNode* temp = node;
            node = node->next;
            free(temp);
        }
    }
}

/// Communicate planes using all to all calls with default data types
void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    // First, sort planes into buckets for each rank
    PlaneList to_send[size];
    for (int i = 0; i < size; i++) {
        to_send[i].head = NULL;
        to_send[i].tail = NULL;
    }
    
    // Check each plane's position and determine if it needs to be sent to another rank
    PlaneNode* current = list->head;
    while (current != NULL) {
        PlaneNode* next = current->next; // Save next pointer before potentially removing
        
        // Calculate new grid indices based on plane's current position
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int index_map = get_index(index_i, index_j, N, M);
        
        // Check if plane has moved to a different region
        int new_rank = get_rank_from_index(index_map, tile_displacements, size);
        
        if (new_rank != rank) {
            // Add to list of planes to send to the new rank
            insert_plane(&to_send[new_rank], current->index_plane, index_map, rank, 
                         current->x, current->y, current->vx, current->vy);
            
            // Remove from current list
            remove_plane(list, current);
        } else {
            // Update the plane's map index if it stayed in the same rank but moved
            current->index_map = index_map;
        }
        
        current = next;
    }
    
    // Step 1: Count how many planes to send to each rank
    int send_counts[size];
    for (int i = 0; i < size; i++) {
        // Count planes to send to rank i
        int count = 0;
        PlaneNode* node = to_send[i].head;
        while (node != NULL) {
            count++;
            node = node->next;
        }
        send_counts[i] = count;
    }
    
    // Step 2: Exchange counts with all other ranks
    int recv_counts[size];
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Step 3: Calculate send/recv displacements
    int send_displs[size], recv_displs[size];
    int total_send = 0, total_recv = 0;
    
    for (int i = 0; i < size; i++) {
        send_displs[i] = total_send;
        total_send += send_counts[i] * 6; // 6 doubles per plane
        
        recv_displs[i] = total_recv;
        total_recv += recv_counts[i] * 6; // 6 doubles per plane
    }
    
    // Step 4: Pack planes into sendbuf
    double *sendbuf = NULL;
    if (total_send > 0) {
        sendbuf = (double*)malloc(total_send * sizeof(double));
    }
    
    for (int i = 0; i < size; i++) {
        int offset = send_displs[i];
        PlaneNode* node = to_send[i].head;
        while (node != NULL) {
            // Pack plane data
            sendbuf[offset++] = (double)node->index_plane;
            sendbuf[offset++] = (double)node->index_map;
            sendbuf[offset++] = node->x;
            sendbuf[offset++] = node->y;
            sendbuf[offset++] = node->vx;
            sendbuf[offset++] = node->vy;
            
            node = node->next;
        }
    }
    
    // Step 5: Allocate receive buffer
    double *recvbuf = NULL;
    if (total_recv > 0) {
        recvbuf = (double*)malloc(total_recv * sizeof(double));
    }
    
    // Step 6: Perform the all-to-all communication
    // We need to convert the counts from plane count to double count (6 doubles per plane)
    int send_counts_doubles[size], recv_counts_doubles[size];
    int send_displs_doubles[size], recv_displs_doubles[size];
    
    for (int i = 0; i < size; i++) {
        send_counts_doubles[i] = send_counts[i] * 6;
        recv_counts_doubles[i] = recv_counts[i] * 6;
        send_displs_doubles[i] = send_displs[i];
        recv_displs_doubles[i] = recv_displs[i];
    }
    
    MPI_Alltoallv(sendbuf, send_counts_doubles, send_displs_doubles, MPI_DOUBLE,
                  recvbuf, recv_counts_doubles, recv_displs_doubles, MPI_DOUBLE,
                  MPI_COMM_WORLD);
    
    // Step 7: Unpack received planes and add to local list
    for (int i = 0; i < size; i++) {
        if (recv_counts[i] > 0) {
            int offset = recv_displs[i];
            for (int j = 0; j < recv_counts[i]; j++) {
                int index_plane = (int)recvbuf[offset++];
                int index_map = (int)recvbuf[offset++];
                double x = recvbuf[offset++];
                double y = recvbuf[offset++];
                double vx = recvbuf[offset++];
                double vy = recvbuf[offset++];
                
                insert_plane(list, index_plane, index_map, rank, x, y, vx, vy);
            }
        }
    }
    
    // Step 8: Free allocated memory
    if (sendbuf) free(sendbuf);
    if (recvbuf) free(recvbuf);
    
    // Clean up the temporary lists
    for (int i = 0; i < size; i++) {
        // Free memory for all nodes in to_send[i]
        PlaneNode* node = to_send[i].head;
        while (node != NULL) {
            PlaneNode* temp = node;
            node = node->next;
            free(temp);
        }
    }
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
    // First, sort planes into buckets for each rank
    PlaneList to_send[size];
    for (int i = 0; i < size; i++) {
        to_send[i].head = NULL;
        to_send[i].tail = NULL;
    }
    
    // Check each plane's position and determine if it needs to be sent to another rank
    PlaneNode* current = list->head;
    while (current != NULL) {
        PlaneNode* next = current->next; // Save next pointer before potentially removing
        
        // Calculate new grid indices based on plane's current position
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int index_map = get_index(index_i, index_j, N, M);
        
        // Check if plane has moved to a different region
        int new_rank = get_rank_from_index(index_map, tile_displacements, size);
        
        if (new_rank != rank) {
            // Add to list of planes to send to the new rank
            insert_plane(&to_send[new_rank], current->index_plane, index_map, rank, 
                         current->x, current->y, current->vx, current->vy);
            
            // Remove from current list
            remove_plane(list, current);
        } else {
            // Update the plane's map index if it stayed in the same rank but moved
            current->index_map = index_map;
        }
        
        current = next;
    }
    
    // Step 1: Count how many planes to send to each rank
    int send_counts[size];
    for (int i = 0; i < size; i++) {
        // Count planes to send to rank i
        int count = 0;
        PlaneNode* node = to_send[i].head;
        while (node != NULL) {
            count++;
            node = node->next;
        }
        send_counts[i] = count;
    }
    
    // Step 2: Exchange counts with all other ranks
    int recv_counts[size];
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Step 3: Create MPI datatype for MinPlaneToSend
    MPI_Datatype plane_type;
    int blocklengths[5] = {1, 1, 1, 1, 1}; // Each field is a single value
    MPI_Aint offsets[5];
    MPI_Datatype types[5] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    
    // Calculate offsets using offsetof for proper alignment
    offsets[0] = offsetof(MinPlaneToSend, index_plane);
    offsets[1] = offsetof(MinPlaneToSend, x);
    offsets[2] = offsetof(MinPlaneToSend, y);
    offsets[3] = offsetof(MinPlaneToSend, vx);
    offsets[4] = offsetof(MinPlaneToSend, vy);
    
    // Create the custom datatype
    MPI_Type_create_struct(5, blocklengths, offsets, types, &plane_type);
    MPI_Type_commit(&plane_type);
    
    // Step 4: Calculate send/recv displacements
    int send_displs[size], recv_displs[size];
    int total_send = 0, total_recv = 0;
    
    for (int i = 0; i < size; i++) {
        send_displs[i] = total_send;
        total_send += send_counts[i];
        
        recv_displs[i] = total_recv;
        total_recv += recv_counts[i];
    }
    
    // Step 5: Pack planes into sendbuf
    MinPlaneToSend *sendbuf = NULL;
    if (total_send > 0) {
        sendbuf = (MinPlaneToSend*)malloc(total_send * sizeof(MinPlaneToSend));
    }
    
    for (int i = 0; i < size; i++) {
        int offset = send_displs[i];
        PlaneNode* node = to_send[i].head;
        while (node != NULL) {
            // Pack plane data into the struct
            sendbuf[offset].index_plane = node->index_plane;
            sendbuf[offset].x = node->x;
            sendbuf[offset].y = node->y;
            sendbuf[offset].vx = node->vx;
            sendbuf[offset].vy = node->vy;
            offset++;
            
            node = node->next;
        }
    }
    
    // Step 6: Allocate receive buffer
    MinPlaneToSend *recvbuf = NULL;
    if (total_recv > 0) {
        recvbuf = (MinPlaneToSend*)malloc(total_recv * sizeof(MinPlaneToSend));
    }
    
    // Step 7: Perform the all-to-all communication with custom datatype
    MPI_Alltoallv(sendbuf, send_counts, send_displs, plane_type,
                  recvbuf, recv_counts, recv_displs, plane_type,
                  MPI_COMM_WORLD);
    
    // Step 8: Unpack received planes and add to local list
    for (int i = 0; i < size; i++) {
        if (recv_counts[i] > 0) {
            int offset = recv_displs[i];
            for (int j = 0; j < recv_counts[i]; j++) {
                // Calculate new map index based on current position
                int index_i = get_index_i(recvbuf[offset].x, x_max, N);
                int index_j = get_index_j(recvbuf[offset].y, y_max, M);
                int index_map = get_index(index_i, index_j, N, M);
                
                insert_plane(list, recvbuf[offset].index_plane, index_map, rank,
                           recvbuf[offset].x, recvbuf[offset].y,
                           recvbuf[offset].vx, recvbuf[offset].vy);
                offset++;
            }
        }
    }
    
    // Step 9: Free allocated memory and MPI datatype
    if (sendbuf) free(sendbuf);
    if (recvbuf) free(recvbuf);
    MPI_Type_free(&plane_type);
    
    // Clean up the temporary lists
    for (int i = 0; i < size; i++) {
        // Free memory for all nodes in to_send[i]
        PlaneNode* node = to_send[i].head;
        while (node != NULL) {
            PlaneNode* temp = node;
            node = node->next;
            free(temp);
        }
    }
}

int main(int argc, char **argv) {
    int debug = 0;                      // 0: no debug, 1: shows all planes information during checking
    int N = 0, M = 0;                   // Grid dimensions
    double x_max = 0.0, y_max = 0.0;    // Total grid size
    int max_steps;                      // Total simulation steps
    char* input_file;                   // Input file name
    int check;                          // 0: no check, 1: check the simulation is correct

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int tile_displacements[size+1];
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

    // Gather timing information from all ranks - we want the maximum values
    double max_time_sim, max_time_comm, max_time_total;
    MPI_Reduce(&time_sim, &max_time_sim, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_comm, &max_time_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_total, &max_time_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("Flight controller simulation: #input %s mode: %d size: %d\n", input_file, mode, size);
        printf("Time simulation:     %.2fs\n", max_time_sim);
        printf("Time communication:  %.2fs\n", max_time_comm);
        printf("Time total:          %.2fs\n", max_time_total);
    }

    if (check ==1)
        check_planes_mpi(&owning_planes_t0, &owning_planes, N, M, x_max, y_max, max_steps, tile_displacements, size, debug);

    MPI_Finalize();
    return 0;
}
