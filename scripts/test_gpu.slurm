#!/bin/bash

#SBATCH --job-name=test_gpu
#SBATCH --partition=biggpu
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err

echo "=== GPU TEST JOB ==="
echo "Date: "$(date)
echo "Username: "$USER
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Node: "$HOSTNAME
echo "Working Directory: "$PWD
echo "Submitted From: "$SLURM_SUBMIT_HOST
echo "Submitted Directory: "$SLURM_SUBMIT_DIR
echo "===================="

cd "$SLURM_SUBMIT_DIR"

source /etc/profile.d/lmod.sh
module load anaconda/2025-12-2

echo "=== PYTHON VERSION ==="
conda run -n wits_env python --version

echo "=== RUNNING GPU TEST ==="
conda run -n wits_env python test_gpu.py

echo "=== GPU TEST COMPLETE ==="