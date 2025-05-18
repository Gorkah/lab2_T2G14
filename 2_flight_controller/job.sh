#!/bin/bash

#SBATCH --job-name=fc_mpi
#SBATCH --output=fc_mpi_%j.out
#SBATCH --error=fc_mpi_%j.err
#SBATCH --ntasks=20
#SBATCH --time=00:10:00

# Load required modules
module load gcc/13.3.0
module load openmpi/4.1.1

# Compile the code
make clean
make mpi

# Run the code with different communication modes
echo "Running with Send/Recv communication"
srun -n 20 ./fc_mpi input_planes_10kk.txt 25 0 0

echo "Running with Alltoall communication"
srun -n 20 ./fc_mpi input_planes_10kk.txt 25 1 0

echo "Running with Struct MPI communication"
srun -n 20 ./fc_mpi input_planes_10kk.txt 25 2 0
