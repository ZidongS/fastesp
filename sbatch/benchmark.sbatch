#!/bin/bash
#SBATCH --job-name=fastesp-bench
#SBATCH --partition=5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:5090:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/bench-%j.out
#SBATCH --error=logs/bench-%j.err

# Head-to-head benchmark on a dedicated 5090 node:
#   Multiwfn (CPU, 1 and 8 threads)  vs  this work (exact analytic ESP on GPU)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results/bench

ulimit -s unlimited || true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=== host $(hostname)  cpus=$(nproc)  $(date -Is) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

echo "=== toolchain ==="
g++ --version | head -1
/usr/local/cuda/bin/nvcc --version | tail -2

echo "=== build ==="
mkdir -p bin
g++ -O3 -fopenmp -march=native -o bin/build_pairs fastesp/native/build_pairs.cpp
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 -o bin/esp_gpu fastesp/native/esp_gpu.cu

echo "=== self-check: analytic integrals vs quadrature ==="
python3 benchmarks/check_integrals.py

echo "=== basis/density cross-check vs Multiwfn (same wavefunction?) ==="
python3 benchmarks/check_density_mwfn.py

echo "=== benchmark ==="
python3 benchmarks/benchmark.py --work results/bench "$@"

echo "=== done $(date -Is) ==="
