#!/usr/bin/env bash
#SBATCH --job-name=llama-vg-r5x3
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster107
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --exclusive
#SBATCH --output=/home-mscluster/dnaidoo3/Wits_Honours_Project/logs/%x-%j.out
#SBATCH --error=/home-mscluster/dnaidoo3/Wits_Honours_Project/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/home-mscluster/dnaidoo3/Wits_Honours_Project"
VENV_ROOT="${PROJECT_ROOT}/.venv"
DATA_ROOT="/datasets/dnaidoo3/Video_Games"
CONFIG_FILE="${PROJECT_ROOT}/pipeline/configs/llm/llama_video_games_recent5_three_qlora.yaml"
TRAIN_SCRIPT="${PROJECT_ROOT}/pipeline/llm/04_train_qlora.py"

cd "${PROJECT_ROOT}"
source "${VENV_ROOT}/bin/activate"

export HF_HOME="${DATA_ROOT}/huggingface_cache"
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONUNBUFFERED="1"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

mkdir -p "${HF_HOME}"

for required_file in \
    "${CONFIG_FILE}" \
    "${TRAIN_SCRIPT}" \
    "${DATA_ROOT}/processed/llm/sft/train_recent5_three_per_user.jsonl" \
    "${DATA_ROOT}/processed/llm/sft/validation_recent5.jsonl"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "ERROR: Required file is missing or empty: ${required_file}" >&2
        exit 1
    fi
done

echo "============================================================"
echo "LLAMA VIDEO GAMES QLORA TRAINING"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "Started:     $(date --iso-8601=seconds)"
echo "Config:      ${CONFIG_FILE}"
echo "HF cache:    ${HF_HOME}"

nvidia-smi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.device_count() >= 2, f'Expected 2 GPUs, found {torch.cuda.device_count()}'; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

MAIN_PROCESS_PORT=$((20000 + SLURM_JOB_ID % 20000))

accelerate launch \
    --multi_gpu \
    --mixed_precision fp16 \
    --num_machines 1 \
    --num_processes 2 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    "${TRAIN_SCRIPT}" \
    "${CONFIG_FILE}"

echo "Finished: $(date --iso-8601=seconds)"

