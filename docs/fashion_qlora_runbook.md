# Fashion QLoRA recent5-three runbook

The scientific implementation is the final Video Games pipeline, Steps 01-08.
Fashion adds separate configs, a manifest, launchers and a compact statistics
export. No trained weights or tuning results are provided by this change.

## Protocol and actual preparation evidence

Verified inputs:

- `/datasets/dnaidoo3/Fashion/processed/Fashion.inter`
- `/datasets/dnaidoo3/Fashion/exports/Fashion_metadata_lookup.json`

Local copies of these files were audited on 2026-10-23. `Fashion.inter` SHA256:
`edf75e2e9f1ad8c2a0672c8e639bef8e18accd86b9c8be9bb3fbfd2984020fbc`.
Preparation checks existing splits against fresh chronological splits without
replacing existing authoritative files. New splits are created only if absent.

| Quantity | Validated count |
| --- | ---: |
| Users | 3,135 |
| Original interactions | 21,441 |
| Training interactions, items 1..n-2 | 15,171 |
| Uncapped possible SFT examples | 12,036 |
| Recent-five, at most three SFT examples/user | **7,894** |
| Validation examples/users | 3,135 |
| Test users | 3,135 |
| Catalogue items | 12,493 |

The old 12,036-example file is **not** recent5-three. New preparation retains at
most the three most recent training prefix/target examples per user. Histories
contain the latest five preceding items, oldest first. Validation target is n-1;
test target is n and its real history includes n-1. Targets never enter prompts.
Steps 02, 05 and 06 share `common.build_recommendation_messages` and format each
history item as `1. ITEM_ID | Title`. Model-specific templates/EOS remain Qwen
`<|im_end|>` and Llama `<|eot_id|>`.

There is one catalogue item without a usable title. Preparation explicitly allows
and reports the existing shared `Unknown title` fallback; users/items are not
removed. Local validation passed 211,647 checks, zero errors, one title warning.

Preparation creates, without overwriting the old train/validation SFT files:

```text
/datasets/dnaidoo3/Fashion/processed/llm/sft/
  train_recent5_three_per_user.jsonl
  validation_recent5.jsonl
  sft_summary_recent5_three.json
  validation_report_recent5_three.json
```

The counts above describe the local verified input. Cluster preparation validates
and prints its actual counts. It fails on conflicting existing splits/partial
SFT outputs rather than changing targets or silently mixing datasets.

## Stage 1 and Stage 2

Models: Qwen/Qwen2.5-3B-Instruct and meta-llama/Llama-3.2-3B-Instruct.
Both use the unchanged Step-04 FP16/NF4 + double-quantization safeguards, frozen
4-bit base, FP32 trainables, completion-only loss, gradient checkpointing,
sequence length 512, batch 2, accumulation 32, effective batch 64, seed 42.

Exactly three one-epoch Stage-1 candidates per model:

| ID | Learning rate | Rank | Alpha | Dropout |
| --- | ---: | ---: | ---: | ---: |
| lr5e5_r8_a16_d005_e1 | 0.00005 | 8 | 16 | 0.05 |
| lr1e4_r8_a16_d005_e1 | 0.0001 | 8 | 16 | 0.05 |
| lr1e4_r16_a32_d005_e1 | 0.0001 | 16 | 32 | 0.05 |

`stage1_train_samples: null` means all available recent5-three records; it is not
a fixed count. Validation loss uses up to 1,000 SFT records, with the same
Step-04 deterministic sampling policy as Video Games (seed 42). Recommendation
validation uses 1,000 users, sampled with seed **20261008**; every candidate and
both models must use exactly the same population hash. Stage-1 candidate ranking
is NDCG@10, Hit@10, MRR@10, then validation loss. The shared ranking code rejects
mixed datasets/workflows and differing evaluation populations.

Promote exactly one winner per model from real completed Stage-1 results.
Stage 2 uses the full training file, one epoch, and all validation-loss examples.
Full checkpoint selection uses every validation user, ranking checkpoints by
NDCG@10, Hit@10, MRR@10. Fashion defaults to single-GPU selection/generation;
generic Video Games sharding remains available and unchanged.

With 7,894 examples, one epoch is approximately `ceil(7894/64) = 124` optimizer
steps. Stage-2 saving defaults to every **50 steps**, with the existing final
checkpoint callback, `save_total_limit: 2`, and epoch validation. This retains a
periodic checkpoint (step 100) and the final checkpoint (approximately step 124).
Promotion counts the actual training file and reduces the interval if needed
for a smaller dataset; fewer than two total steps is rejected rather than
claiming a periodic-plus-final checkpoint can exist. The current Stage-2
completion/resume/identity/verified-checkpoint machinery is reused unchanged.

Selection and generation use 10 beams, batch 4, top 10, recent-five prompts,
complete-history exclusions and full-catalogue constraints. Final generation
retains max_new_tokens=24 and length_penalty=1.0, and reads Step 05's selected
checkpoint artifact. Test users/targets never select hyperparameters/checkpoints.

## Paths and execution

Source configs:
`pipeline/configs/llm/{qwen,llama}_fashion_recent5_three_qlora.yaml`.
Manifest: `pipeline/configs/llm/tuning/fashion_recent5_three_sweep.yaml`.

Generated configs (git-ignored; regenerate on the cluster):

```text
pipeline/configs/llm/tuning/generated/fashion/
  stage1_fast_3090/{model}_{candidate}.yaml
  stage1_fast_3090_configs.txt
  fast_promoted/stage2_fast_3090/{model}_{winning_candidate}_full.yaml
```

Model run outputs are isolated from Video Games and old Fashion runs:

```text
/datasets/dnaidoo3/Fashion/processed/llm/checkpoints/tuning/
  {model}/stage1_fast_3090/all_samples/{candidate}/
  {model}/stage2_fast_3090/{winning_candidate}/
```

Each Stage-2 output contains the existing run/trainer summaries,
`validation_selection/checkpoint_selection.json`, per-checkpoint metric JSONs,
and `test_predictions/{test_predictions.jsonl,generation_summary.json,
test_metrics.json,per_user_metrics.jsonl}`.

GPU jobs: bigbatch, one node/task, 8 CPUs, 64G, four hours, exclusive node,
CUDA_VISIBLE_DEVICES=0, CUDA preflight, no GRES, no pinned good node, exclude
mscluster74. Training launches one Accelerate FP16 process. The output lock
prevents overlapping jobs and Stage-2 state resumes through Step 04 normally.
CPU jobs: batch, 2 CPUs, 8G, one hour; no visible GPU. All jobs use the project
.venv and HF_HOME=/datasets/dnaidoo3/Fashion/huggingface_cache.

## Exact cluster commands

After committing/pushing the source changes locally:

```bash
cd /home-mscluster/dnaidoo3/Wits_Honours_Project
git pull --ff-only origin main
source .venv/bin/activate

# Preparation must succeed before the generated configs are used.
sbatch --wait slurm/llm/fashion_qlora_cpu.slurm prepare && \
  python pipeline/llm/fashion_qlora_workflow.py submit-stage1 --submit
```

The Stage-1 submitter launches **six GPU jobs**. Each trains one candidate then
runs Step 05 on its validation sample. One CPU ranking job depends on all six
via `afterok`. No held-out generation/evaluation is submitted in this phase.
The helper prints each job ID, including `rank: <JOB_ID>`. Without `--submit`,
it only prints its plan and never invokes sbatch.

```bash
# Inspect the printed ranking job ID; continue only after COMPLETED / 0:0.
sacct -X -j RANK_JOB_ID --format=JobID,State,ExitCode

# Promotion is lightweight and refuses missing/incomplete ranking evidence.
python pipeline/llm/fashion_qlora_workflow.py promote

# Review the actual promoted winners/configs; print the dependency plan first.
python pipeline/llm/fashion_qlora_workflow.py submit-final
python pipeline/llm/fashion_qlora_workflow.py submit-final --submit
```

The final submitter discovers the two actual promoted configs without guessing
winner IDs. Per model it chains `train -> select -> generate -> evaluate` with
`afterok`, then chains CPU comparison/export after **both** evaluations succeed:
8 model jobs plus 1 final CPU job. All Stage-2 jobs preserve the output lock,
resume validity checks and completion guard. A timed-out/failed training job
blocks its downstream scientific stages; inspect the failure, then resubmit the
same final plan to resume its Stage-2 training. No checkpoint is hard-coded.

Ranking evidence:
`pipeline/configs/llm/tuning/results/fashion_stage1_fast_3090/tuning_results.json`.

Monitoring:

```bash
squeue -u "$USER" -o '%.18i %.24j %.12T %.12M %R'
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,NodeList
tail -f slurm-fashion-qwen_train-JOB_ID.out
```

For explicit stage reruns with an actual promoted config:

```bash
CONFIG=path/to/actual/promoted/fashion_config.yaml
sbatch slurm/llm/fashion_qlora_gpu.slurm train "$CONFIG"
sbatch slurm/llm/fashion_qlora_gpu.slurm select "$CONFIG"
sbatch slurm/llm/fashion_qlora_gpu.slurm generate "$CONFIG"
sbatch slurm/llm/fashion_qlora_cpu.slurm evaluate "$CONFIG"
```

These individual commands are alternatives to the automatic chain: run each
only after its prerequisite succeeds. Completed Step-06 output resumes/skips
through its existing mechanism; do not overwrite successful final predictions.

## Evaluation, comparison, statistics export

Unchanged Step 07 evaluates every test user at K=1,5,10: Hit, Recall, NDCG, MRR
and CatalogueCoverage. It writes standard aggregate and per-user JSON(L).
Unchanged Step 08 compares only the two Fashion configs, using NDCG@10 as primary
metric and its existing paired per-user comparison implementation. Comparison
artifacts are under `results/qlora/fashion_comparison/`.

The final CPU job exports `results/statistics/fashion/`:

```text
user_groups.csv
{qwen,llama}_qlora/
  per_user_metrics.jsonl
  test_metrics.json
  generation_summary.json
  validation_selection/checkpoint_selection.json
  validation_selection/checkpoint-*_metrics.json
  run_summary.json
  train_results.json
  validation_results.json
  all_results.json
  config.yaml
```

Grouping uses the **complete original test history**, not the recent-five model
prompt: original_history_length=len(test history)+1; sparse <=7, longer_history
>7. Every CSV record_index/user_id is checked against the full test population
and both Step-07 per-user files. Export also requires complete validation for
all checkpoints, complete test counts, consistent model/checkpoint identities,
and the required metric cutoffs. It refuses partial/misaligned results.

Only explicitly allowlisted files are copied. Weights, raw validation prediction
dumps and raw test_predictions.jsonl are excluded. A complete existing bundle is
never silently overwritten: use another output directory for a revised export.
To export manually after both evaluations:

```bash
python pipeline/llm/export_qlora_statistics.py \
  path/to/actual/qwen_promoted.yaml path/to/actual/llama_promoted.yaml
```

The default repository .gitignore ignores generated configs, datasets and JSONL.
Regenerate configs/data on the cluster. If transporting final statistics via Git,
explicitly add only the intended compact bundle files; do not force-add weights
or raw prediction dumps. No final statistics exist until real jobs complete.

## Lightweight local verification

```bash
python pipeline/llm/fashion_qlora_workflow.py prepare --data-root datasets/Fashion
python pipeline/llm/fashion_qlora_workflow.py submit-stage1
python -m unittest discover -s pipeline/llm -p 'test_*.py' -v
python -m py_compile pipeline/llm/09_tuning_workflow.py \
  pipeline/llm/fashion_qlora_workflow.py pipeline/llm/export_qlora_statistics.py \
  pipeline/llm/test_fashion_qlora_workflow.py
bash -n slurm/llm/fashion_qlora_gpu.slurm
bash -n slurm/llm/fashion_qlora_cpu.slurm
git diff --check
```
