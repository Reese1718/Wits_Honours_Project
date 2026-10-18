#!/usr/bin/env bash
#SBATCH --job-name=vg-r5x3-s2-3090
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=05:00:00
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
export CUDA_VISIBLE_DEVICES="0"

mkdir -p "${HF_HOME}"

for required_file in "${CONFIG_FILE}" "${TRAIN_SCRIPT}"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "ERROR: Required file is missing or empty: ${required_file}" >&2
        exit 1
    fi
done

python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, (
    f"Expected exactly 1 visible GPU, found {torch.cuda.device_count()}"
)

properties = torch.cuda.get_device_properties(0)
print("Visible GPU:", properties.name)
print(f"Total VRAM: {properties.total_memory / 1024**3:.2f} GiB")
PY

# Lock the canonical output directory for the entire Accelerate process.
# Dependencies serialize planned chunks; flock also blocks accidental overlap.
OUTPUT_DIRECTORY=$(python - "${CONFIG_FILE}" <<'PYCONFIG'
import sys
from pathlib import Path
import yaml

with open(sys.argv[1]) as file:
    config = yaml.safe_load(file)
if config.get("tuning", {}).get("stage") != 2:
    raise SystemExit("This launcher accepts Stage-2 configs only.")
print(Path(config["training"]["output_dir"]).expanduser().resolve())
PYCONFIG
)
mkdir -p "${OUTPUT_DIRECTORY}"
exec 9>"${OUTPUT_DIRECTORY}/.stage2.lock"
if ! flock -n 9; then
    echo "ERROR: Another Stage-2 chunk holds the output-directory lock." >&2
    exit 1
fi

MAIN_PROCESS_PORT=$((20000 + SLURM_JOB_ID % 20000))

accelerate launch \
    --mixed_precision fp16 \
    --num_machines 1 \
    --num_processes 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    "${TRAIN_SCRIPT}" \
    "${CONFIG_FILE}"
