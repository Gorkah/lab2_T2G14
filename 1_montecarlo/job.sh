#!/bin/bash
#SBATCH --job-name=montecarlo_scaling
#SBATCH --output=output_%j.txt       # Archivo de salida (%j = ID del trabajo)
#SBATCH --ntasks=192                 # Número máximo de procesadores
#SBATCH --time=01:00:00              # Tiempo máximo (1 hora)
#SBATCH --partition=compute          # Partición de nodos de cómputo

# Cargar módulos necesarios
module load gcc/13.3.0
module load openmpi/4.1.1

# Parámetros fijos
DIMENSION=10
SEED=42

# Escalabilidad fuerte
NUM_SAMPLES_STRONG=100000000
echo "Strong Scaling Results" > strong_scaling_results.txt
for size in 1 2 4 8 16 32 64 128 192; do
    echo "Running with $size processors (strong scaling)" >> strong_scaling_results.txt
    mpirun -n $size ./montecarlo $DIMENSION $NUM_SAMPLES_STRONG $SEED >> strong_scaling_results.txt
done

# Escalabilidad débil
SAMPLES_PER_PROCESS=100000000
echo "Weak Scaling Results" > weak_scaling_results.txt
for size in 1 2 4 8 16 32 64 128 192; do
    NUM_SAMPLES_WEAK=$((SAMPLES_PER_PROCESS * size))
    echo "Running with $size processors (weak scaling)" >> weak_scaling_results.txt
    mpirun -n $size ./montecarlo $DIMENSION $NUM_SAMPLES_WEAK $SEED >> weak_scaling_results.txt
done