#!/bin/bash
#SBATCH -N 1
#SBATCH -C intel
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH -t 02:30:00
#SBATCH --output=tdacc-run-%j.log

OMP_NUM_THREADS=24 time srun -n1 ./build/tdacc 20 10 2 712

OMP_NUM_THREADS=24 time srun -n1 ./build/tdacc 10 10 2 712
OMP_NUM_THREADS=16 time srun -n1 ./build/tdacc 10 10 2 712
OMP_NUM_THREADS=8 time srun -n1 ./build/tdacc 10 10 2 712
OMP_NUM_THREADS=4 time srun -n1 ./build/tdacc 10 10 2 712
OMP_NUM_THREADS=1 time srun -n1 ./build/tdacc 10 10 2 712
