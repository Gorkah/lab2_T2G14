#!/bin/bash

#SBATCH --job-name=fc_scaling
#SBATCH --output=fc_scaling_%j.out
#SBATCH --error=fc_scaling_%j.err
#SBATCH --ntasks=80
#SBATCH --time=00:20:00

# Load required modules
module load gcc/13.3.0
module load openmpi/4.1.1

# Compile the code
make clean
make mpi

# Run the code with increasing number of ranks using the best communication strategy (mode 2)
echo "Running with 20 ranks"
srun -n 20 ./fc_mpi input_planes_10kk.txt 25 2 0

echo "Running with 40 ranks"
srun -n 40 ./fc_mpi input_planes_10kk.txt 25 2 0

echo "Running with 60 ranks"
srun -n 60 ./fc_mpi input_planes_10kk.txt 25 2 0

echo "Running with 80 ranks"
srun -n 80 ./fc_mpi input_planes_10kk.txt 25 2 0
