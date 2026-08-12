#!/bin/bash

#SBATCH --job-name=inspect_recbole
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=recbole_inspect_%j.out
#SBATCH --error=recbole_inspect_%j.err

cd "$SLURM_SUBMIT_DIR"

source /etc/profile.d/lmod.sh
export MODULEPATH=/etc/lmod/modules:/usr/share/lmod/lmod/modulefiles

module load anaconda/anaconda-2025-12-2

echo "=== RECBOLE TRAINER LOCATION ==="

conda run -n wits_env python -c "
import inspect
import recbole.trainer.trainer
print(inspect.getfile(recbole.trainer.trainer))
"

echo ""
echo "=== TORCH.LOAD REFERENCES ==="

conda run -n wits_env python -c "
import inspect
from recbole.trainer import Trainer

source = inspect.getsource(Trainer.evaluate)

for i, line in enumerate(source.splitlines(), 1):
    if 'torch.load' in line or 'checkpoint' in line:
        print(f'{i}: {line}')
"
