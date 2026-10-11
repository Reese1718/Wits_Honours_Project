#!/usr/bin/env bash
#SBATCH --job-name=vg-r5x3-tune
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster107
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/home-mscluster/dnaidoo3/Wits_Honours_Project"
VENV_ROOT="${PROJECT_ROOT}/.venv"
DATA_ROOT="/datasets/dnaidoo3/Video_Games"
TRAIN_SCRIPT="${PROJECT_ROOT}/pipeline/llm/04_train_qlora.py"
CONFIG_ARGUMENT="${1:?Usage: sbatch $0 <config.yaml>}"

if [[ "${CONFIG_ARGUMENT}" = /* ]]; then
    CONFIG_FILE="${CONFIG_ARGUMENT}"
else
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_ARGUMENT}"
fi

cd "${PROJECT_ROOT}"
source "${VENV_ROOT}/bin/activate"

export HF_HOME="${DATA_ROOT}/huggingface_cache"
export PYTHONUNBUFFERED="1"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

mkdir -p "${HF_HOME}"

for required_file in "${CONFIG_FILE}" "${TRAIN_SCRIPT}"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "ERROR: Required file is missing or empty: ${required_file}" >&2
        exit 1
    fi
done

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.device_count() == 2, f'Expected exactly 2 allocated GPUs, found {torch.cuda.device_count()}'; print('Allocated GPUs:', [torch.cuda.get_device_name(i) for i in range(2)])"

MAIN_PROCESS_PORT=$((20000 + SLURM_JOB_ID % 20000))

accelerate launch \
    --multi_gpu \
    --mixed_precision fp16 \
    --num_machines 1 \
    --num_processes 2 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    "${TRAIN_SCRIPT}" \
    "${CONFIG_FILE}"
