#!/usr/bin/env bash
#SBATCH --job-name=check-blackwell
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --exclusive
#SBATCH --output=logs/check_blackwell_%j.out
#SBATCH --error=logs/check_blackwell_%j.err

set -euo pipefail

PROJECT_ROOT="/home-mscluster/dnaidoo3/Wits_Honours_Project"

cd "${PROJECT_ROOT}"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="0"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
assert torch.cuda.device_count() == 1, (
    f"Expected one visible GPU, found {torch.cuda.device_count()}"
)

print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()

print("FP16 CUDA matmul: PASS")
print("Result dtype:", y.dtype)
PY

python - <<'PY'
import bitsandbytes as bnb
import torch

print("bitsandbytes import: PASS")
print("bitsandbytes version:", getattr(bnb, "__version__", "unknown"))
print("CUDA available:", torch.cuda.is_available())
PY
