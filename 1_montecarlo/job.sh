#!/bin/bash
#SBATCH --job-name=montecarlo_mpi
#SBATCH --output=montecarlo_%j.out
#SBATCH --error=montecarlo_%j.err
#SBATCH --ntasks=8
#SBATCH --time=00:10:00

# Load required modules
module load gcc/13.3.0
module load openmpi/4.1.1

# Run the program
mpirun -n $SLURM_NTASKS ./montecarlo 3 1000000 42
