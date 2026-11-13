# Repository cleanup — 13 November 2026

The cleanup removes terminal exports and logs while retaining experiment code,
configuration, documentation, datasets, intermediate inputs, and run metadata.
No training, evaluation, package installation, or job submission was performed.
Changes remain local and uncommitted.

## Removed artifacts

- 12 tracked Slurm training logs under `logs/`.
- Two tracked TensorBoard event files under `log_tensorboard/`.
- Three terminal comparison exports under `results/qlora/fashion_comparison/`.
- 77 tracked terminal tables, paired-test exports, interpretation text files,
  and figures from the Fashion and Video Games context-comparison directories.
- 23 ignored Python bytecode caches with corresponding source files retained.
- Saved outputs in 48 cells across six GAN comparison notebooks. Cell sources,
  markdown, and other notebook metadata were preserved.

Two context-comparison figures were already deleted before cleanup; those local
deletions were preserved. They are not included in the 94 tracked files removed
by this cleanup.

All 117 removed files and the six original notebooks were backed up outside the
repository to:

```text
C:\Users\deree\AppData\Local\Temp\Wits_Honours_Project-cleanup-20261113T093847Z
```

`plan.json` records the initial Git status, paths, sizes, and SHA-256 hashes of
removed files. `originals/` preserves repository-relative paths. Untracked
caches were moved there; tracked artifacts were copied and verified before
removal. This location is temporary storage; move the backup to durable storage
if it must survive system temporary-file cleanup.

## Generated inputs retained

- All `datasets/` files, including raw data, metadata lookups, interaction files,
  augmentation responses/histories, temporal splits, and SFT files. These are
  inputs to subsequent stages even when produced by earlier stages.
- `results/statistics/{fashion,video_games}/user_groups.csv`: read by baseline
  audits and training orchestration.
- `results/statistics/*/recent5_baselines/`: complete exported bundles consumed
  by completion/status validation. Their metric files are part of that contract.
- `results/statistics/*/{qwen,llama}_qlora/`: retained run configurations,
  summaries, checkpoint-selection records, metrics, and per-user results support
  reproduction and downstream comparisons. Metrics are not disposable when they
  serve as comparison inputs.
- `pipeline/configs/llm/tuning/results/qwen_stage1_fast_3090/`: the measured
  Stage-1 ranking feeds promotion; generated candidate/promoted configs and config
  lists are also retained.
- Context-comparison `run_manifest.json`, `verified_evaluation_protocol.json`,
  `configuration_comparison.json`, and `effective_context_audit.json`: retained
  provenance rather than terminal display exports. References to removed
  terminal tables/figures describe historical outputs, not runnable inputs.
- `reports/methodology_evidence/` and `reports/recent5_baseline_audits/`: retained
  provenance and dataset/protocol audit metadata.
- The existing virtual environment and all setup requirements.

## Files needing review

- `pipeline/configs/video_games_augmentation/bert4rec_video_games_llama.yaml:108`:
  pre-existing merge-conflict markers make this YAML invalid. It was preserved
  unchanged; this configuration needs repair before its training entry point can
  use it.
- `saved/SASRec-Aug-10-2026_17-28-27.pth`: the only local original SASRec checkpoint
  and a source of effective historical settings. It was not established to be a
  disposable checkpoint.
- `pipeline/llm/__pycache__/stage2_recovery.cpython-312.pyc` and
  `pipeline/llm/__pycache__/test_stage2_recovery.cpython-312.pyc`: their matching
  source files are absent. Retained because they could be the only local copies
  of that implementation.
- `datasets/Video Games/exports/Video_Games_metadata_lookup.backup.json`: role and
  equivalence to the active lookup are unverified; retained.
- Empty tracked placeholders: `scripts/setup_cluster.sh`, `scripts/evaluate.sh`,
  `scripts/train_bert4rec.slurm`, `scripts/train_gru4rec.slurm`, and
  `pipeline/evaluation/{compare_models,evaluate_models,generate_tables}.py`.
  These were already empty; they are not functional entry points.

## Static dependency checks and existing limitations

Python modules are parsed without importing training code. Notebook structure and
cell sources are checked against the backups. Referenced repository scripts,
base configurations, generated config lists, local dataset paths, and retained
input files are checked statically; external Drive/HPC paths are not validated
as existing on this machine.

Verification passed for 58 Python modules, 64 YAML files, four manifest base-config
references, 13 generated config-list entries, 24 job script/config references,
and all six notebook source/metadata comparisons. One additional YAML file has
the pre-existing conflict noted above. All 16 distinct configured LLM input files
exist after translating the recorded Drive/HPC dataset prefix to the local
dataset root. The 364 retained files checked against the pre-cleanup inventory
were unchanged (content hashes for code/config/metadata, size and modification
time for dataset files). All 117 artifact backups passed SHA-256 verification.
`git diff --check` passed; the Git index remains untouched. Detailed verification
results are in the backup's `verification.json`.

The GAN comparison notebooks already require external original significance
bundles (`preparation/checkpoint_configs.json`, per-user results, and overlap
flags) and GAN-augmented recommender checkpoints not available locally. They are
not made self-contained by this cleanup. Some configurations likewise use
absolute Drive/HPC paths, and the Movies/TV dataset is not present locally.

`logs/.gitkeep` keeps the log directory available in a fresh checkout: Slurm opens
its output/error files before the job body can create directories. Python output
writers use parent-directory creation or shared helpers; comparison outputs are
created when their stages run. The removed historical context-comparison
figures/tables have no generating entry point in this checkout; recreating those
specific exports requires their original external analysis workflow.
