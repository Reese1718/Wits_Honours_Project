```bash
#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err

module load anaconda/anaconda-2025-12-2

source /opt/anaconda/etc/profile.d/conda.sh
conda activate wits_env

echo "===== SYSTEM INFORMATION ====="
hostname
echo "Python:"
python --version

echo ""
echo "===== PYTORCH INFORMATION ====="
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo ""
    echo "===== GPU DETECTED ====="
    python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
else
    echo ""
    echo "===== NO GPU DETECTED ====="
fi
```
