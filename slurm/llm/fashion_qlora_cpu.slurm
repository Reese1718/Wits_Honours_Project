#!/usr/bin/env bash
#SBATCH --job-name=fashion-qlora-cpu
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --exclude=mscluster74
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd /home-mscluster/dnaidoo3/Wits_Honours_Project
source .venv/bin/activate
export HF_HOME=/datasets/dnaidoo3/Fashion/huggingface_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
ACTION="${1:?prepare, rank, evaluate or compare-export required}"
shift
case "${ACTION}" in
    prepare) python pipeline/llm/fashion_qlora_workflow.py prepare "$@" ;;
    rank) python pipeline/llm/fashion_qlora_workflow.py rank "$@" ;;
    evaluate)
        python pipeline/llm/fashion_qlora_workflow.py check-config "$1"
        python pipeline/llm/07_evaluate_recommendations.py "$1" ;;
    compare-export)
        python pipeline/llm/fashion_qlora_workflow.py check-config "$1"
        python pipeline/llm/fashion_qlora_workflow.py check-config "$2"
        python pipeline/llm/08_compare_results.py "$1" "$2" \
            --primary-metric NDCG@10 --output-dir results/qlora/fashion_comparison
        python pipeline/llm/export_qlora_statistics.py "$1" "$2" ;;
    *) echo "Unknown Fashion CPU action: ${ACTION}" >&2; exit 2 ;;
esac
