# Video Games recent5-three QLoRA tuning

This workflow tunes Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct without
changing the temporal split, recent-5 prompt history, maximum three training
examples per user, or held-out test set. Stage 1 uses the same seeded 100,000
example training subset and the same seeded 5,000-user recommendation
validation subset for every candidate. Stage 2 retrains the best two candidates
per model on all 292,890 training examples and selects checkpoints on all
109,258 validation users.

The held-out test split is not read until the two model-family winners have
been fixed by full-validation NDCG@10.

## Stage-1 design

| Candidate | Learning rate | Rank | Alpha | Dropout | Epochs |
|---|---:|---:|---:|---:|---:|
| `lr5e5_r8_a16_d0_e2` | 5e-5 | 8 | 16 | 0.00 | 2 |
| `lr1e4_r8_a16_d005_e3` | 1e-4 | 8 | 16 | 0.05 | 3 |
| `lr2e4_r16_a32_d01_e2` | 2e-4 | 16 | 32 | 0.10 | 2 |
| `lr5e5_r16_a32_d005_e3` | 5e-5 | 16 | 32 | 0.05 | 3 |
| `lr1e4_r32_a64_d01_e2` | 1e-4 | 32 | 64 | 0.10 | 2 |
| `lr2e4_r32_a64_d0_e3` | 2e-4 | 32 | 64 | 0.00 | 3 |

The learning-rate range tests conservative through aggressive adaptation.
Rank tests adapter capacity, while alpha remains `2 x rank` so the LoRA scaling
relationship stays comparable. Dropout tests no, moderate, and stronger adapter
regularisation. Two versus three epochs tests underfitting against overfitting.
Every value for learning rate, rank, and dropout occurs twice, and each epoch
count occurs three times. This covers the requested ranges without a Cartesian
product.

NF4, double quantization, float16, the cosine scheduler, batch sizes, gradient
accumulation, gradient checkpointing, maximum sequence length, history
exclusion, constrained decoding, temporal splits, and seed remain fixed.

## Prepare Stage 1

Run this once after cloning or updating the repository:

```bash
cd /home-mscluster/dnaidoo3/Wits_Honours_Project
source .venv/bin/activate
python pipeline/llm/09_tuning_workflow.py prepare-stage1
```

This creates deterministic configs under
`pipeline/configs/llm/tuning/generated/stage1/`.

## Stage-1 training submissions

Submit only the jobs the available allocation can support. These commands do
not run automatically:

```bash
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr5e5_r8_a16_d0_e2.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr1e4_r8_a16_d005_e3.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr2e4_r16_a32_d01_e2.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr5e5_r16_a32_d005_e3.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr1e4_r32_a64_d01_e2.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr2e4_r32_a64_d0_e3.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr5e5_r8_a16_d0_e2.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr1e4_r8_a16_d005_e3.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr2e4_r16_a32_d01_e2.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr5e5_r16_a32_d005_e3.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr1e4_r32_a64_d01_e2.yaml
sbatch slurm/llm/tune_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr2e4_r32_a64_d0_e3.yaml
```

The launcher defaults to `gpuexpress`, two allocated GPUs, two Accelerate
processes, and FP16. If the site rejects the requested GPU count or eight-hour
limit, resubmit the affected command with cluster-approved overrides rather
than changing an experiment config.

## Stage-1 recommendation selection

After each matching training job has completed successfully, submit:

```bash
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr5e5_r8_a16_d0_e2.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr1e4_r8_a16_d005_e3.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr2e4_r16_a32_d01_e2.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr5e5_r16_a32_d005_e3.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr1e4_r32_a64_d01_e2.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/qwen_lr2e4_r32_a64_d0_e3.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr5e5_r8_a16_d0_e2.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr1e4_r8_a16_d005_e3.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr2e4_r16_a32_d01_e2.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr5e5_r16_a32_d005_e3.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr1e4_r32_a64_d01_e2.yaml
sbatch slurm/llm/select_video_games_recent5_three.slurm pipeline/configs/llm/tuning/generated/stage1/llama_lr2e4_r32_a64_d0_e3.yaml
```

Each run evaluates every retained epoch checkpoint with catalogue-constrained
decoding. It excludes the complete real history and ranks checkpoints by
NDCG@10, then Hit@10 and MRR@10. All candidates must report the same validation
sample SHA-256 before aggregation succeeds. Ranking also checks the generated
config list, so a missing or extra candidate config stops aggregation.

Rank Stage 1 and promote the top two candidates per model:

```bash
python pipeline/llm/09_tuning_workflow.py rank \
  --config-dir pipeline/configs/llm/tuning/generated/stage1 \
  --output-dir /datasets/dnaidoo3/Video_Games/processed/llm/checkpoints/tuning/results/stage1

python pipeline/llm/09_tuning_workflow.py promote-stage2 \
  --ranking /datasets/dnaidoo3/Video_Games/processed/llm/checkpoints/tuning/results/stage1/tuning_results.json \
  --top-per-model 2 \
  --output-dir pipeline/configs/llm/tuning/generated/stage2
```

## Stage 2

The promotion command writes four paths to `stage2_configs.txt`, ordered as the
top two Qwen configs followed by the top two Llama configs. Train each on the
complete recent5-three training dataset:

```bash
STAGE2_LIST=pipeline/configs/llm/tuning/generated/stage2/stage2_configs.txt
sbatch --partition=biggpu --time=1-00:00:00 slurm/llm/tune_video_games_recent5_three.slurm "$(sed -n '1p' "${STAGE2_LIST}")"
sbatch --partition=biggpu --time=1-00:00:00 slurm/llm/tune_video_games_recent5_three.slurm "$(sed -n '2p' "${STAGE2_LIST}")"
sbatch --partition=biggpu --time=1-00:00:00 slurm/llm/tune_video_games_recent5_three.slurm "$(sed -n '3p' "${STAGE2_LIST}")"
sbatch --partition=biggpu --time=1-00:00:00 slurm/llm/tune_video_games_recent5_three.slurm "$(sed -n '4p' "${STAGE2_LIST}")"
```

After those jobs finish, evaluate each retained checkpoint on the full
validation set. These jobs can be expensive, so they use `biggpu` overrides:

```bash
sbatch --partition=biggpu --time=1-00:00:00 slurm/llm/select_video_games_recent5_three.slurm "$(sed -n '1p' "${STAGE2_LIST}")"
sbatch --partition=biggpu --time=1-00:00:00 slurm/llm/select_video_games_recent5_three.slurm "$(sed -n '2p' "${STAGE2_LIST}")"
sbatch --partition=biggpu --time=1-00:00:00 slurm/llm/select_video_games_recent5_three.slurm "$(sed -n '3p' "${STAGE2_LIST}")"
sbatch --partition=biggpu --time=1-00:00:00 slurm/llm/select_video_games_recent5_three.slurm "$(sed -n '4p' "${STAGE2_LIST}")"
```

Rank only after all four full-validation jobs succeed:

```bash
python pipeline/llm/09_tuning_workflow.py rank \
  --config-dir pipeline/configs/llm/tuning/generated/stage2 \
  --output-dir /datasets/dnaidoo3/Video_Games/processed/llm/checkpoints/tuning/results/stage2 \
  --require-full-validation
```

## Final validation to test sequence

`winning_configs.txt` contains the validation-selected Qwen winner followed by
the validation-selected Llama winner. These are the only configurations sent
to the held-out test set:

```bash
WINNER_LIST=/datasets/dnaidoo3/Video_Games/processed/llm/checkpoints/tuning/results/stage2/winning_configs.txt
sbatch slurm/llm/evaluate_video_games_recent5_three.slurm "$(sed -n '1p' "${WINNER_LIST}")"
sbatch slurm/llm/evaluate_video_games_recent5_three.slurm "$(sed -n '2p' "${WINNER_LIST}")"
```

Do not use either test result to revisit the sweep, choose a different
checkpoint, or alter hyperparameters.

## Compare with sequential baselines

After both held-out evaluations finish, provide the existing baseline JSON in
the format accepted by `08_compare_results.py`, then run:

```bash
python pipeline/llm/08_compare_results.py \
  "$(sed -n '1p' "${WINNER_LIST}")" \
  "$(sed -n '2p' "${WINNER_LIST}")" \
  --primary-metric NDCG@10 \
  --baseline-file experiments/video_games_baselines.json \
  --output-dir /datasets/dnaidoo3/Video_Games/processed/llm/checkpoints/tuning/results/final_comparison
```

The repository does not currently contain
`experiments/video_games_baselines.json`; populate it from the completed
SASRec, BERT4Rec, and GRU4Rec evaluations without altering their values.
