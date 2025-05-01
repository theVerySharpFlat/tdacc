#!/bin/bash
#SBATCH -N 1
#SBATCH -C intel
#SBATCH --exclusive
#SBATCH --ntasks=24
#SBATCH --mem-per-cpu=8G
#SBATCH -t 02:30:00
#SBATCH --output=tdacc-run-mpi-%j.log

OMP_NUM_THREADS=1 time srun -n24 ./build/tdacc 10 10 2 712
OMP_NUM_THREADS=2 time srun -n12 ./build/tdacc 10 10 2 712
OMP_NUM_THREADS=4 time srun -n6 ./build/tdacc 10 10 2 712
OMP_NUM_THREADS=6 time srun -n4 ./build/tdacc 10 10 2 712
OMP_NUM_THREADS=12 time srun -n2 ./build/tdacc 10 10 2 712
