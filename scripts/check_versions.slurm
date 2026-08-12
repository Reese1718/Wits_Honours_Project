#!/bin/bash

#SBATCH --job-name=check_versions
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=versions_%j.out
#SBATCH --error=versions_%j.err

echo "=========================================="
echo "ENVIRONMENT VERSION CHECK"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Date: $(date)"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"

# Initialise Lmod
source /etc/profile.d/lmod.sh

# Explicit cluster module path
export MODULEPATH=/etc/lmod/modules:/usr/share/lmod/lmod/modulefiles

# Load the cluster Anaconda environment
module load anaconda/anaconda-2025-12-2

echo ""
echo "=== PYTHON VERSION ==="
conda run -n wits_env python --version

echo ""
echo "=== PYTORCH VERSION ==="
conda run -n wits_env python -c "import torch; print(torch.__version__)"

echo ""
echo "=== RECBOLE VERSION ==="
conda run -n wits_env python -c "import recbole; print(recbole.__version__)"

echo ""
echo "=== CUDA INFORMATION ==="
conda run -n wits_env python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

echo ""
echo "=========================================="
echo "VERSION CHECK COMPLETE"
echo "=========================================="


