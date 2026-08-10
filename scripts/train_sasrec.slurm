#!/bin/bash

#SBATCH --job-name=sasrec
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=sasrec_%j.out
#SBATCH --error=sasrec_%j.err

echo "=================================================="
echo "SASRec TRAINING JOB"
echo "=================================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Working Directory: $PWD"
echo "=================================================="

cd "$SLURM_SUBMIT_DIR"

# Load Lmod
source /etc/profile.d/lmod.sh

# Explicitly set the cluster module paths
export MODULEPATH=/etc/lmod/modules:/usr/share/lmod/lmod/modulefiles

# Load Anaconda
module load anaconda/anaconda-2025-12-2

# Verify that Anaconda/Conda loaded correctly
echo ""
echo "=== ANACONDA CHECK ==="
type conda
conda --version

# Make the cluster datasets visible inside Apptainer
export APPTAINER_BINDPATH="/datasets:/datasets"

echo ""
echo "=== PYTHON VERSION ==="
conda run -n wits_env python --version

echo ""
echo "=== STARTING SASRec ==="

conda run -n wits_env python pipeline/training/train_sasrec.py

echo ""
echo "=================================================="
echo "SASRec JOB COMPLETE"
echo "=================================================="
