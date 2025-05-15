#!/bin/bash
#SBATCH --job-name=flight_controller_mpi
#SBATCH --output=fc_mpi_%j.out
#SBATCH --error=fc_mpi_%j.err
#SBATCH --ntasks=5
#SBATCH --time=00:10:00

# Load required modules
module load gcc/13.3.0
module load openmpi/4.1.1

# Run the program with communication mode 1 (alltoall)
mpirun -n $SLURM_NTASKS ./fc_mpi input_planes_test.txt 10 1 1
