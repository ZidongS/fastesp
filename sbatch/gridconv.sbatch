#!/bin/bash
#SBATCH --job-name=fastesp-gridconv
#SBATCH --partition=5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gridconv-%j.out
#SBATCH --error=logs/gridconv-%j.err

# Does Multiwfn converge towards the exact analytic ESP when its Becke grid is
# refined?  If yes, the 1.8e-7 a.u. gap in the head-to-head comparison is
# Multiwfn's quadrature error, not ours.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
ulimit -s unlimited || true
export PYTHONUNBUFFERED=1
python3 -c "
import numpy as np
np.save('results/gpu_300.npy', np.loadtxt('results/esp_cpp.txt', skiprows=1)[:,3])
"
python3 benchmarks/mwfn_grid_convergence.py data/frag54.molden.input data/pts_bohr.txt results/gpu_300.npy
