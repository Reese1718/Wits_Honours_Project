#!/usr/bin/env bash
#SBATCH --job-name=vg-stage2-select
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=30:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="/home-mscluster/dnaidoo3/Wits_Honours_Project"
cd "${PROJECT_ROOT}"
source .venv/bin/activate
export HF_HOME=/datasets/dnaidoo3/Video_Games/huggingface_cache
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

CONFIG="${1:?config.yaml required}"
NUM_SHARDS="${2:?num_shards required}"
RUN_ID="${3:?unique run ID required}"
BATCH_SIZE="${4:-4}"
EXPECTED_USERS="${5:-109258}"
MODE="${6:-worker}"
ARGS=("${CONFIG}" --num-shards "${NUM_SHARDS}" --shard-run "${RUN_ID}"
      --generation-batch-size "${BATCH_SIZE}" --expected-users "${EXPECTED_USERS}")
if [[ "${MODE}" == aggregate ]]; then
    python pipeline/llm/05_select_best_checkpoint.py "${ARGS[@]}" --aggregate
elif [[ "${MODE}" == worker ]]; then
    export CUDA_VISIBLE_DEVICES=0
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.device_count() == 1; print('GPU:', torch.cuda.get_device_name(0))"
    python pipeline/llm/05_select_best_checkpoint.py "${ARGS[@]}" \
        --shard-index "${SLURM_ARRAY_TASK_ID:?must run as a Slurm array}"
else
    echo "Unknown mode: ${MODE}" >&2
    exit 2
fi
