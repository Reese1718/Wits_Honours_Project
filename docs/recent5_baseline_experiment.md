# Recent-five sequential baseline experiment

This is a **context-window ablation**, alongside the original 50-context baselines.
It does not replace them, retune them, or change QLoRA results. The six new YAMLs
are byte-for-byte parent copies except `MAX_ITEM_LIST_LENGTH: 50` becomes `5`:

| Model | Fashion config | Video Games config |
| --- | --- | --- |
| SASRec | `pipeline/configs/sasrec_fashion_recent5.yaml` | `pipeline/configs/sasrec_games_recent5.yaml` |
| BERT4Rec | `pipeline/configs/bert4rec_fashion_recent5.yaml` | `pipeline/configs/bert4rec_games_recent5.yaml` |
| GRU4Rec | `pipeline/configs/gru4rec_fashion_recent5.yaml` | `pipeline/configs/gru4rec_games_recent5.yaml` |

Each parent has the same filename without `_recent5`. Parent fingerprints are
frozen in `recent5_baselines.py` (LF-normalized for Windows/Linux portability).
Architecture settings, learning rate, epochs, batch sizes, early stopping,
masking probability, loss, sampling, temporal ordering, metrics/top-k and seed 42
remain model/dataset specific. In particular, Fashion BERT4Rec keeps mask ratio
0.1 and LR 0.0001; Video Games BERT4Rec keeps ratio 0.2 and LR 0.00001.

## Data and target identity

No new `.inter` files, copies, or symlinks are created. RecBole's
`SequentialDataset.data_augmentation` uses `MAX_ITEM_LIST_LENGTH` to truncate each
preceding history while retaining its target. The existing chronological
`TO / LS: valid_and_test / full` policy still chooses the final two targets.
This is not a new data split and does not limit training examples to three/user.

CPU-only audits on the original local sources, on 2026-10-25, established:

| Dataset | Users/validation/test | Items | Original interactions | Training examples |
| --- | ---: | ---: | ---: | ---: |
| Fashion | 3,135 | 12,493 | 21,441 | 12,036 |
| Video Games | 109,258 | 54,595 | 945,660 | 617,886 |

Evidence: `reports/recent5_baseline_audits/fashion_bert4rec.json` and
`reports/recent5_baseline_audits/video_games_sasrec.json`. Full native population
audits were run with BERT4Rec for Fashion and SASRec for Video Games; tests cover
all three models' native data paths and effective configurations on synthetic
data. Every production job repeats the full audit for its actual model/runtime.

The audits compare 50 vs 5 training target ordering/fingerprints, validation/test
targets and latest-five histories. They check original source hashes, full
user/item populations, original chronological targets and, when available, the
existing QLoRA split files and user_groups.csv. All local authoritative targets
and groups matched. Any mismatch aborts; targets are never silently repaired.
No models, training, or test-score evaluation were run during these audits.

Colab source discovery defaults to the unchanged parent data root:
`/content/drive/MyDrive/Wits_Honours_Project/datasets/`, preferring the
existing `.inter` under `processed/` or the dataset directory. The local
`datasets/Video Games/processed/Video Games.inter` spelling is also supported.
Conflicting sources fail. An explicit `--inter-file` can identify the exact file.

The source YAMLs retain their Colab path. The runtime adapter sets RecBole's
**resolved** data_path to
the existing interaction file's directory, as the evaluation notebooks do, and
resolves its existing filename directly on mounted Drive. `--data-root datasets`
remains available for local CPU audits.

## Reused execution and required BERT safeguard

`recent5_recbole.py` runs the unchanged `train_sasrec.py`, `train_bert4rec.py`, or
`train_gru4rec.py` using `runpy`, with scoped I/O/instrumentation hooks. Their
native model initialization, seeding order (including BERT's second seed reset),
Trainer.fit, early stopping, best-checkpoint restoration and evaluation remain
in place. Logs, TensorBoard and checkpoints are isolated in the new run directory.

**Necessary execution exception:** RecBole 1.2.0 computes Fashion BERT4Rec's mask
storage width as `int(0.1 * 5) == 0`. This produces empty or ragged mask tensors.
The scoped adapter clamps that storage width to at least one in the transform
and model attribute when it would be zero. It does not change mask_ratio, random
mask probabilities, ft_ratio, loss, layer sizes or token-selection code. Tests
reproduce the native zero-buffer failure and verify the fix. The exception is
recorded in data_audit.json and run_summary.json. It does not affect the original
50-context BERT configuration or the other model configurations.

Native BERT4Rec also appends a prediction MASK within the configured sequence
length. At a full five-slot history it therefore uses four real preceding items
plus MASK. Its parent does the analogous operation with 50 slots. This existing
behavior is preserved; no sixth position or modified architecture is introduced.

Full-sort scores are captured **after the original Trainer applies its masks**.
No new seen-item filtering is added. RecBole's sequential loader in the tested
version supplies history_index=None; padding item 0 is masked. This native policy
is retained even though QLoRA separately excludes complete seen histories. The
experiment matches context limits, not every detail of the QLoRA evaluation stack.
The capture mirrors the existing evaluation notebooks' native-score export.

## Outputs and paired-statistics contract

For each `{dataset}/{model}`, where dataset is `fashion` or `video_games` and
model is `sasrec`, `bert4rec`, or `gru4rec`:

```text
results/recent5_baselines/{dataset}/{model}/
  checkpoints/                    # native best-validation checkpoint, never exported
  log/ and log_tensorboard/       # native logs
  config.yaml                    # exact recent5 YAML
  parent_config.yaml
  runtime_config.yaml            # resolved defaults, paths, device, settings
  selected_checkpoint.json       # path/hash, selected epoch, validation score
  data_audit.json
  test_metrics.json
  per_user_metrics.jsonl
  run_summary.json
  complete.json                  # published only after all validation succeeds

results/statistics/{dataset}/recent5_baselines/{model}/
  config.yaml, parent_config.yaml, runtime_config.yaml
  selected_checkpoint.json, data_audit.json
  test_metrics.json, per_user_metrics.jsonl, run_summary.json, complete.json
```

Native RecBole aggregate results are preserved, including their original
rounding. Per-user Hit@10, Recall@10, NDCG@10 and MRR@10 use unrounded values from
the same native torch.topk score batches; their means must reproduce native
aggregates within the original rounding tolerance. No second inference pass is
needed. IDs/targets must match the complete audited population exactly once.
NaNs, positive-infinite scores, invalid target scores and invalid per-user values
are rejected before publishing a complete result.

Per-user JSONL follows QLoRA's `record_index`, `user_id`, `target`, `metrics`
structure, with `target_rank`, `original_history_length`, and `group` added.
`metrics` contains CamelCase keys such as `NDCG@10`. Row order/index follows the
existing QLoRA test file when present, otherwise chronological-source users in
sorted user-ID order (the same Step-01 convention). Pair by user_id and target,
and verify populations before computing statistical differences.

Groups use **total original interaction length**, equivalently full test history
length + 1: sparse <=7, longer_history >7. Never group by the truncated sequence.
Existing shared user_groups.csv files are validated and never rewritten. Local
audits found Fashion groups 2,494/641 and Video Games groups 69,074/40,184.

The statistics exporter copies only its explicit nine-file allowlist. It never
copies weights, caches, raw tensors or checkpoint directories. Existing run or
statistics directories are refused rather than overwritten. Use a new explicit
`--output-root`/`--statistics-root` for an intentionally separate run. Failed runs
have no valid complete.json; inspect them before choosing a new run directory.
If training completed but export failed, retry only `export`, without retraining.
Do not delete or overwrite original baseline or QLoRA output directories.

## Exact Colab cells

Use a **GPU runtime** in Google Colab. Make sure the revised source files and
existing datasets, QLoRA split files and shared user_groups.csv files are present
in your Drive repository. Existing notebooks are unchanged.

The execution references are `colab_notebooks/Experiments/SasRec.ipynb`,
`Bert4Rec.ipynb` and `Gru4Rec.ipynb`: mount Drive, install notebook dependencies,
then invoke the original `pipeline/training/train_*.py` scripts. The recent5
adapter invokes those same scripts in a fresh Python process, adding the scoped
BERT safeguard, output isolation, target audits and per-user capture. Do not also
run the training scripts separately for the same experiment.

Mount Drive and enter the existing repository:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Wits_Honours_Project
```

Install into Colab's active Python environment; no separate environment activation
is needed. This uses the original notebooks' requirements.txt installation,
GRU4Rec's NumPy/TensorBoard/RecBole pins, and the kmeans-pytorch/ray dependencies
used in the existing RecBole comparison/evaluation notebooks:

```python
%pip install -r requirements.txt "recbole==1.2.0" "numpy==1.26.4" "tensorboard==2.18.0" kmeans-pytorch ray
```

Restart the Colab session after installation, then rerun the mount/change-directory
cell. The older SASRec notebook contains alternative NumPy 2 repair cells; do not
combine those with this existing NumPy 1.26.4 setup or rerun its source-editing
repair cells. Compatibility aliases are already in the unchanged training scripts.
The notebooks do not pin a single shared PyTorch version; runtime versions are
recorded in each run_summary.json, rather than claiming an exact historical
environment match.

Validate the six configs and confirm the GPU without training:

```python
import torch
assert torch.cuda.is_available(), 'Select a GPU runtime in Colab'
print(torch.__version__, torch.cuda.get_device_name(0))
!python pipeline/evaluation/recent5_baselines.py validate
```

Run each command below in its **own Colab cell**, sequentially on one runtime.
Each command validates, audits, trains, restores the native best checkpoint,
evaluates and automatically exports its compact statistics. The existing
hyperparameters and stopping criteria are retained.

Fashion SASRec:

```python
!python -u pipeline/evaluation/recent5_baselines.py run --dataset fashion --model sasrec
```

Fashion BERT4Rec:

```python
!python -u pipeline/evaluation/recent5_baselines.py run --dataset fashion --model bert4rec
```

Fashion GRU4Rec:

```python
!python -u pipeline/evaluation/recent5_baselines.py run --dataset fashion --model gru4rec
```

Video Games SASRec:

```python
!python -u pipeline/evaluation/recent5_baselines.py run --dataset video_games --model sasrec
```

Video Games BERT4Rec:

```python
!python -u pipeline/evaluation/recent5_baselines.py run --dataset video_games --model bert4rec
```

Video Games GRU4Rec:

```python
!python -u pipeline/evaluation/recent5_baselines.py run --dataset video_games --model gru4rec
```

Training progress and native evaluation metrics appear in cell output; logs and
checkpoints persist under the Drive result directory. A disconnected/incomplete
run is not marked complete and is not automatically resumed. Inspect it before
choosing fresh output/statistics roots; do not overwrite a completed experiment.

Per-user statistics export is automatic. If a completed run's export failed,
retry only its export (example below), without training or evaluating again.
An already existing export is intentionally refused:

```python
!python pipeline/evaluation/recent5_baselines.py export --dataset fashion --model sasrec
```

Use the same command with `--dataset video_games` and/or `--model bert4rec` or
`--model gru4rec` for the other completed runs. Verify the artifacts and compact
exports for **all six** experiments:

```python
!python pipeline/evaluation/recent5_baselines.py status
```

Success prints six `COMPLETE` lines and exits zero after checking each bundle's
completion marker and file hashes. Missing/incomplete runs or exports fail the
check; this command never trains or submits work.

To inspect one full data audit without training, in a Colab cell:

```python
!python pipeline/evaluation/recent5_baselines.py audit --dataset fashion --model bert4rec
```

Or locally, using the original local datasets:

```bash
python pipeline/evaluation/recent5_baselines.py audit --dataset fashion --model bert4rec \
  --data-root datasets --audit-output reports/recent5_baseline_audits/fashion_bert4rec.json
```

Local lightweight checks:

```bash
python -m unittest discover -s pipeline/evaluation -p test_recent5_baselines.py -v
python -m py_compile pipeline/evaluation/recent5_baselines.py \
  pipeline/evaluation/recent5_recbole.py pipeline/evaluation/test_recent5_baselines.py
git diff --check
```

The repository ignores *.jsonl. After real runs finish, explicitly force-add only
the intended compact per_user_metrics.jsonl files when transporting statistics
through Git. Do not force-add checkpoints or complete run directories.
