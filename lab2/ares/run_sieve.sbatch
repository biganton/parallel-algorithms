#!/bin/bash
#SBATCH --job-name=sieve
#SBATCH --account=plgar2026-cpu
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=ares/sieve_%j.out

module load scipy-bundle/2021.10-intel-2021b
export SLURM_OVERLAP=1

N=${1:-1000000000}
cd "$SLURM_SUBMIT_DIR"

for p in 1 2 4 8 16 32; do
    mpiexec -n $p src/sieve_mpi.py $N
    mpiexec -n $p src/sieve_shared.py $N
done
