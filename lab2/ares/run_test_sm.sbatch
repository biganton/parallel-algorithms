#!/bin/bash
#SBATCH --job-name=test_sm
#SBATCH --account=plgar2026-cpu
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=00:05:00
#SBATCH --output=ares/test_sm_%j.out

module load scipy-bundle/2021.10-intel-2021b
export SLURM_OVERLAP=1
cd "$SLURM_SUBMIT_DIR"

mpiexec src/try.py
mpiexec src/test_sm.py
