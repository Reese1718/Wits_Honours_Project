#!/bin/bash
#SBATCH --job-name=qlora_prep
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home-mscluster/dnaidoo3/Wits_Honours_Project/logs/qlora_prepare_%j.out
#SBATCH --error=/home-mscluster/dnaidoo3/Wits_Honours_Project/logs/qlora_prepare_%j.err

set -euo pipefail

cd /home-mscluster/dnaidoo3/Wits_Honours_Project

source .venv/bin/activate

python pipeline/llm/09_tuning_workflow.py prepare-stage1
