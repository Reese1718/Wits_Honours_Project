#!/usr/bin/env bash
#SBATCH --job-name=fashion-qlora
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --exclude=mscluster74
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd /home-mscluster/dnaidoo3/Wits_Honours_Project
source .venv/bin/activate
export HF_HOME=/datasets/dnaidoo3/Fashion/huggingface_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
ACTION="${1:?stage1, train, select or generate required}"
CONFIG="${2:?Fashion config required}"
# Validate dataset, execution settings and actual populations before loading weights.
python pipeline/llm/fashion_qlora_workflow.py check-config "${CONFIG}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.device_count() == 1; print('GPU:', torch.cuda.get_device_name(0))"
OUTPUT_DIRECTORY=$(python - "${CONFIG}" <<'PYCONFIG'
import sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(config['training']['output_dir']).expanduser().resolve())
PYCONFIG
)
mkdir -p "${OUTPUT_DIRECTORY}"
# Same output-directory lock used by the proven Video Games Stage-2 launcher.
exec 9>"${OUTPUT_DIRECTORY}/.stage2.lock"
if ! flock -n 9; then
    echo "Another Fashion job holds the output-directory lock" >&2
    exit 1
fi
train() {
    accelerate launch --mixed_precision fp16 --num_machines 1 --num_processes 1 \
        --main_process_port "$((20000 + SLURM_JOB_ID % 20000))" \
        pipeline/llm/04_train_qlora.py "${CONFIG}"
}
case "${ACTION}" in
    stage1) train; python pipeline/llm/05_select_best_checkpoint.py "${CONFIG}" ;;
    train) train ;;
    select) python pipeline/llm/05_select_best_checkpoint.py "${CONFIG}" ;;
    generate) python pipeline/llm/06_generate_test_recommendations.py "${CONFIG}" ;;
    *) echo "Unknown Fashion GPU action: ${ACTION}" >&2; exit 2 ;;
esac
