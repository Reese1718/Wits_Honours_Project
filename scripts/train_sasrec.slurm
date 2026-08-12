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

# ------------------------------------------------------
# Check configuration argument
# ------------------------------------------------------

if [ -z "$1" ]; then
    echo "ERROR: No SASRec configuration supplied."
    echo "Usage: sbatch train_sasrec.slurm <config_file>"
    exit 1
fi

CONFIG_FILE="$1"

echo ""
echo "=== SASRec CONFIGURATION ==="
echo "Config file: $CONFIG_FILE"

# ------------------------------------------------------
# Load Lmod
# ------------------------------------------------------

source /etc/profile.d/lmod.sh

export MODULEPATH=/etc/lmod/modules:/usr/share/lmod/lmod/modulefiles

# ------------------------------------------------------
# Load Anaconda
# ------------------------------------------------------

module load anaconda/anaconda-2025-12-2

# ------------------------------------------------------
# Verify Conda
# ------------------------------------------------------

echo ""
echo "=== ANACONDA CHECK ==="
type conda
conda --version

# ------------------------------------------------------
# Make cluster datasets visible inside Apptainer
# ------------------------------------------------------

export APPTAINER_BINDPATH="/datasets:/datasets"

# ------------------------------------------------------
# Python version
# ------------------------------------------------------

echo ""
echo "=== PYTHON VERSION ==="
conda run -n wits_env python --version

# ------------------------------------------------------
# Start SASRec
# ------------------------------------------------------

echo ""
echo "=== STARTING SASRec ==="
echo "Configuration: $CONFIG_FILE"

conda run -n wits_env python \
    pipeline/training/train_sasrec.py \
    "$CONFIG_FILE"

# ------------------------------------------------------
# Finished
# ------------------------------------------------------

echo ""
echo "=================================================="
echo "SASRec JOB COMPLETE"
echo "=================================================="