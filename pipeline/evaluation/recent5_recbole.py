"""Scoped RecBole adapter: native datasets, original trainers and native scores.

No training loop, model architecture or evaluation filtering is reimplemented.
The original train_*.py entry points run under scoped I/O/instrumentation hooks.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import yaml

from recent5_baselines import (
    ROOT, MODELS, DATASETS, EXPORT_FILES, atomic_json, configs, evaluation_population,
    export_statistics, metrics_at_10, sha256, source_population, validate_config, validate_per_user,
)


def recbole_api():
    # Same NumPy compatibility aliases used by the original three entry points.
    for name, value in {"float_": np.float64, "complex_": np.complex128, "unicode_": np.str_}.items():
        if not hasattr(np, name): setattr(np, name, value)
    import torch
    import recbole.config
    import recbole.data
    import recbole.trainer
    from recbole.data.dataset import Dataset
    from recbole.data.dataloader import FullSortEvalDataLoader
    from recbole.data.transform import MaskItemSequence
    from recbole.model.sequential_recommender.bert4rec import BERT4Rec
    from recbole.utils import init_seed
    return dict(torch=torch, config_module=recbole.config, data_module=recbole.data,
                trainer_module=recbole.trainer, Config=recbole.config.Config,
                create_dataset=recbole.data.create_dataset, data_preparation=recbole.data.data_preparation,
                Trainer=recbole.trainer.Trainer, Dataset=Dataset, FullSortEvalDataLoader=FullSortEvalDataLoader,
                MaskItemSequence=MaskItemSequence, BERT4Rec=BERT4Rec, init_seed=init_seed)


@contextmanager
def input_and_mask_hooks(api, source, compatibility):
    """Resolve an existing filename without creating/copying a .inter dataset.

    Both 'Video Games.inter' and 'Video_Games.inter' source spellings exist.
    Only the original interaction file's I/O token changes.
    """
    original_load = api["Dataset"]._load_inter_feat
    original_transform_init = api["MaskItemSequence"].__init__
    original_transform_call = api["MaskItemSequence"].__call__
    original_model_init = api["BERT4Rec"].__init__
    original_multi_hot = api["BERT4Rec"].multi_hot_embed
    original_eval_init = api["FullSortEvalDataLoader"].__init__
    def recent5_bert(config):
        return config["model"] == "BERT4Rec" and config["MAX_ITEM_LIST_LENGTH"] == 5
    def load_inter(instance, token, dataset_path):
        if Path(dataset_path).resolve() != Path(source).parent.resolve():
            raise ValueError("Unexpected runtime data path")
        return original_load(instance, Path(source).stem, dataset_path)
    def transform_init(instance, config):
        original_transform_init(instance, config)
        instance._recent5_mask_safety = recent5_bert(config)
        if instance._recent5_mask_safety:
            if instance.mask_item_length == 0:
                compatibility["bert_zero_mask_buffer_minimum_one"] = True
            instance.mask_item_length = max(1, instance.mask_item_length)
            compatibility["bert_minimum_mask_storage_one"] = True
            compatibility["bert_forced_nonempty_training_masks"] = True
            compatibility["bert_position_zero_mask_index_sentinel"] = 5
            compatibility.setdefault("forced_mask_count", 0)
            compatibility.setdefault("forced_position_zero_mask_count", 0)
    def transform_call(instance, dataset, interaction):
        # Keep RecBole's random decisions, ft_ratio branch, truncation and
        # negatives intact. Only repair rows that contribute no native target.
        result = original_transform_call(instance, dataset, interaction)
        if not getattr(instance, "_recent5_mask_safety", False):
            return result
        indices = result[instance.MASK_INDEX]
        empty_rows = (~(indices > 0).any(dim=1)).nonzero(as_tuple=True)[0].tolist()
        if not empty_rows:
            return result
        sequences = result[instance.ITEM_SEQ].detach().cpu().tolist()
        n_items = dataset.num(instance.ITEM_ID)
        for row in empty_rows:
            sequence = sequences[row]
            # Native sequences are right padded. Stop at padding just as the
            # original transform does; never sample a target from a pad slot.
            positions = []
            for position, item in enumerate(sequence):
                if item == 0:
                    break
                positions.append(position)
            if not positions:
                raise ValueError("Recent5 BERT4Rec cannot supervise an empty history")
            position = positions[-1]  # deterministic latest real item
            negative = instance._neg_sample(sequence, n_items)
            result[instance.MASK_ITEM_SEQ][row, position] = n_items
            result[instance.POS_ITEMS][row, -1] = sequence[position]
            result[instance.NEG_ITEMS][row, -1] = negative
            # Index 0 is native padding in the loss. A singleton has no real
            # position >0: use the otherwise impossible index 5 as its marker,
            # decoded to position 0 only by the scoped training gather below.
            indices[row, -1] = position if position > 0 else instance.max_seq_length
            compatibility["forced_mask_count"] += 1
            if position == 0:
                compatibility["forced_position_zero_mask_count"] += 1
        return result
    def model_init(instance, config, dataset):
        original_model_init(instance, config, dataset)
        instance._recent5_mask_safety = recent5_bert(config)
        if instance._recent5_mask_safety:
            instance.mask_item_length = max(1, instance.mask_item_length)
    def eval_init(instance, *args, **kwargs):
        original_eval_init(instance, *args, **kwargs)
        # RecBole also transforms sequential full-sort evaluation batches.
        # Those masks are unused by prediction: retain its native draws and
        # never force evaluation masks or add them to the training counters.
        if isinstance(instance.transform, api["MaskItemSequence"]):
            instance.transform._recent5_mask_safety = False
    def multi_hot_embed(instance, masked_index, max_length):
        if getattr(instance, "_recent5_mask_safety", False):
            # Keep the positive sentinel in the interaction for native loss
            # weighting; decode a separate tensor for gathering hidden states.
            masked_index = api["torch"].where(masked_index == instance.max_seq_length,
                                              0, masked_index)
        return original_multi_hot(instance, masked_index, max_length)
    with ExitStack() as stack:
        stack.enter_context(patch.object(api["Dataset"], "_load_inter_feat", load_inter))
        stack.enter_context(patch.object(api["MaskItemSequence"], "__init__", transform_init))
        stack.enter_context(patch.object(api["MaskItemSequence"], "__call__", transform_call))
        stack.enter_context(patch.object(api["BERT4Rec"], "__init__", model_init))
        stack.enter_context(patch.object(api["BERT4Rec"], "multi_hot_embed", multi_hot_embed))
        stack.enter_context(patch.object(api["FullSortEvalDataLoader"], "__init__", eval_init))
        yield


def native_config(api, dataset, model, path, source, checkpoint_dir, cpu=False):
    # Suppress our orchestration CLI options: original scripts pass one filename.
    with patch.object(sys, "argv", ["recent5", str(path)]):
        config = api["Config"](model=MODELS[model], config_file_list=[str(path)],
                               config_dict={"use_gpu": False} if cpu else None)
    # RecBole appends dataset to data_path; set the final resolved directory as
    # existing evaluation notebooks do. No interaction contents are rewritten.
    config["data_path"] = str(Path(source).parent.resolve())
    config["checkpoint_dir"] = str(Path(checkpoint_dir).resolve())
    return config


def snapshot(loader, config):
    dataset = loader.dataset
    feat = dataset.inter_feat
    uid, iid, timestamp = (config[key] for key in ("USER_ID_FIELD", "ITEM_ID_FIELD", "TIME_FIELD"))
    checksum = hashlib.sha256()
    for field in (uid, iid, timestamp):
        checksum.update(feat[field].cpu().numpy().tobytes())
    users = dataset.id2token(uid, feat[uid].cpu().numpy())
    targets = dataset.id2token(iid, feat[iid].cpu().numpy())
    sequence = feat[iid + config["LIST_SUFFIX"]]
    lengths = feat[config["ITEM_LIST_LENGTH_FIELD"]].cpu().tolist()
    if len(users) != len(set(users)):
        raise ValueError("Leave-one-out evaluation must contain exactly one target per user")
    return dict(count=len(users), target_sha256=checksum.hexdigest(), targets=dict(zip(users, targets)),
                recent_context={str(user): sequence[index, max(0, length-5):length].cpu().tolist()
                                for index, (user, length) in enumerate(zip(users, lengths))})


def prepared_identity(train, config):
    feat = train.dataset.inter_feat
    checksum = hashlib.sha256()
    for field in (config["USER_ID_FIELD"], config["ITEM_ID_FIELD"], config["TIME_FIELD"]):
        checksum.update(feat[field].cpu().numpy().tobytes())
    return {"rows": len(train.dataset), "target_order_sha256": checksum.hexdigest()}


def audit_data(dataset, model, source, split_dir=None, groups_file=None, expected_users=None):
    parent, recent, _ = validate_config(dataset, model)
    source = Path(source).resolve()
    before = sha256(source)
    histories, items = source_population(source)
    expected_users = DATASETS[dataset][2] if expected_users is None else expected_users
    if len(histories) != expected_users:
        raise ValueError(f"Expected {expected_users} original users; found {len(histories)}")
    source_summary = Path(split_dir) / "split_summary.json" if split_dir else None
    if source_summary and source_summary.is_file():
        recorded = json.loads(source_summary.read_text())
        if recorded.get("source_sha256") != before:
            raise ValueError("Original interaction file differs from authoritative split source fingerprint")
    api = recbole_api()
    compatibility = {}
    snapshots = {}
    print(f"CPU-only native RecBole split audit: {dataset}/{model}", flush=True)
    with tempfile.TemporaryDirectory() as directory, input_and_mask_hooks(api, source, compatibility):
        for cap, path in ((50, parent), (5, recent)):
            print(f"  Preparing MAX_ITEM_LIST_LENGTH={cap}; no model/training", flush=True)
            config = native_config(api, dataset, model, path, source, directory, cpu=True)
            api["init_seed"](config["seed"], config["reproducibility"])
            data = api["create_dataset"](config)
            observed_users = set(map(str, data.field2id_token[config["USER_ID_FIELD"]][1:]))
            observed_items = set(map(str, data.field2id_token[config["ITEM_ID_FIELD"]][1:]))
            if observed_users != set(histories) or observed_items != items:
                raise ValueError("RecBole filtered/remapped away original users or items")
            train, valid, test = api["data_preparation"](config, data)
            for loader in (train, valid, test):
                lengths = loader.dataset.inter_feat[config["ITEM_LIST_LENGTH_FIELD"]]
                if int(lengths.max()) > cap: raise ValueError("Sequence cap not respected")
            snapshots[cap] = {
                "training": prepared_identity(train, config),
                "validation": snapshot(valid, config), "test": snapshot(test, config),
            }
            del train, valid, test, data
            gc.collect()
    if snapshots[50] != snapshots[5]:
        raise ValueError("Context cap changed training targets, validation/test targets, or latest-five contexts")
    population, evidence = evaluation_population(histories, snapshots[5]["validation"]["targets"],
        snapshots[5]["test"]["targets"], split_dir, groups_file)
    if sha256(source) != before: raise ValueError("Original dataset changed during audit")
    report = {
        "dataset": dataset, "model": MODELS[model], "recbole_version": importlib.metadata.version("recbole"),
        "source": str(source), "source_sha256": before, "source_unchanged": True,
        "users": len(histories), "items": len(items),
        "interactions": sum(map(len, histories.values())),
        "training_examples": snapshots[5]["training"]["rows"],
        "validation_users": snapshots[5]["validation"]["count"], "test_users": snapshots[5]["test"]["count"],
        "parent_config_sha256": sha256(parent), "recent_config_sha256": sha256(recent),
        "target_identity": {"training": snapshots[5]["training"],
            "validation_sha256": snapshots[5]["validation"]["target_sha256"],
            "test_sha256": snapshots[5]["test"]["target_sha256"]},
        "full_vs_recent_targets_equal": True, "recent_five_contexts_equal": True,
        "temporal_source_targets_equal": True, "evidence": evidence,
        "original_history_groups": {"sparse": sum(row["group"] == "sparse" for row in population.values()),
                                    "longer_history": sum(row["group"] == "longer_history" for row in population.values())},
        "execution_compatibility": compatibility,
    }
    return report, population


def capture_rows(api, interaction, scores, positive_u, positive_i, dataset, config, population):
    torch = api["torch"]
    if bool(torch.isnan(scores).any()) or bool(torch.isposinf(scores).any()):
        raise ValueError("NaN or positive-infinite full-sort scores")
    count = scores.shape[0]
    if sorted(positive_u.cpu().tolist()) != list(range(count)):
        raise ValueError("Expected one held-out target per sequential evaluation row")
    top_items = torch.topk(scores, 10, dim=-1).indices.detach().cpu().tolist()
    positive = dict(zip(positive_u.cpu().tolist(), positive_i.cpu().tolist()))
    if not bool(torch.isfinite(scores[positive_u.to(scores.device), positive_i.to(scores.device)]).all()):
        raise ValueError("Held-out target score is masked or nonfinite")
    uid, iid = config["USER_ID_FIELD"], config["ITEM_ID_FIELD"]
    users = dataset.id2token(uid, interaction[uid].cpu().numpy())
    rows = []
    for index, user in enumerate(users):
        user = str(user)
        target = str(dataset.id2token(iid, np.array([positive[index]]))[0])
        if user not in population or population[user]["target"] != target:
            raise ValueError("Native test target/user differs from audited population")
        candidates = top_items[index]
        rank = candidates.index(positive[index]) + 1 if positive[index] in candidates else None
        rows.append(dict(population[user], target_rank=rank, metrics=metrics_at_10(rank)))
    return rows


def trainer_adapter(api, config_state, population):
    class CapturingTrainer(api["Trainer"]):
        def fit(self, *args, **kwargs):
            started = time.monotonic()
            result = super().fit(*args, **kwargs)
            self.recent5_training_seconds = time.monotonic() - started
            self.recent5_best_result = result
            return result
        def _save_checkpoint(self, epoch, *args, **kwargs):
            result = super()._save_checkpoint(epoch, *args, **kwargs)
            self.recent5_best_epoch = epoch
            return result
        def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
            self.recent5_capture = bool(load_best_model)
            if self.recent5_capture:
                path = Path(model_file or self.saved_model_file).resolve()
                path.relative_to(Path(self.config["checkpoint_dir"]).resolve())
                self.recent5_rows = []
                self.recent5_history_mask_supplied = set()
            try:
                result = super().evaluate(eval_data, load_best_model=load_best_model,
                                         model_file=model_file, show_progress=show_progress)
            finally:
                self.recent5_capture = False
            if load_best_model:
                config_state["trainer"] = self
                config_state["test_result"] = result
                config_state["test_dataset"] = eval_data.dataset
            return result
        def _full_sort_batch_eval(self, batched_data):
            output = super()._full_sort_batch_eval(batched_data)
            if getattr(self, "recent5_capture", False):
                interaction, scores, positive_u, positive_i = output
                self.recent5_history_mask_supplied.add(batched_data[1] is not None)
                self.recent5_rows.extend(capture_rows(api, interaction, scores, positive_u, positive_i,
                    config_state["test_dataset_for_capture"], self.config, population))
            return output
    return CapturingTrainer


def primitive(value):
    if isinstance(value, dict): return {str(key): primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [primitive(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return str(value)


def train_existing(dataset, model, source, split_dir, groups_file, run, exported):
    api = recbole_api()
    if not api["torch"].cuda.is_available(): raise RuntimeError("CUDA unavailable; select a GPU runtime in Colab")
    if Path(run).exists(): raise FileExistsError(f"Recent5 run already exists: {run}; choose a fresh --output-root")
    if Path(exported).exists(): raise FileExistsError(f"Statistics result already exists: {exported}")
    parent, recent, _ = validate_config(dataset, model)
    report, population = audit_data(dataset, model, source, split_dir, groups_file)
    run.mkdir(parents=True)
    atomic_json(run / "data_audit.json", report)
    shutil.copy2(parent, run / "parent_config.yaml")
    shutil.copy2(recent, run / "config.yaml")
    config_state = {}
    compatibility = dict(report["execution_compatibility"])
    original_preparation = api["data_preparation"]
    def runtime_config(*args, **kwargs):
        config = api["Config"](*args, **kwargs)
        if config["model"] != MODELS[model] or config["MAX_ITEM_LIST_LENGTH"] != 5:
            raise ValueError("Original training entry point loaded an unexpected config/model")
        config["data_path"] = str(Path(source).parent.resolve())
        config["checkpoint_dir"] = str(run / "checkpoints")
        config_state["config"] = config
        return config
    def preparation(config, data):
        loaders = original_preparation(config, data)
        train, valid, test = loaders
        if (prepared_identity(train, config) != report["target_identity"]["training"]
                or snapshot(valid, config)["target_sha256"] != report["target_identity"]["validation_sha256"]
                or snapshot(test, config)["target_sha256"] != report["target_identity"]["test_sha256"]):
            raise ValueError("Training split differs from preflight audit")
        config_state["test_dataset_for_capture"] = test.dataset
        return loaders
    original_cwd = Path.cwd()
    script = ROOT / f"pipeline/training/train_{model}.py"
    started = time.monotonic()
    try:
        os.chdir(run)
        with input_and_mask_hooks(api, source, compatibility), ExitStack() as stack:
            stack.enter_context(patch.object(api["config_module"], "Config", runtime_config))
            stack.enter_context(patch.object(api["data_module"], "data_preparation", preparation))
            stack.enter_context(patch.object(api["trainer_module"], "Trainer", trainer_adapter(api, config_state, population)))
            stack.enter_context(patch.object(sys, "argv", [str(script), str(recent)]))
            runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(original_cwd)
    finish_run(api, dataset, model, source, run, exported, report, population, config_state,
               compatibility, time.monotonic() - started)


def finish_run(api, dataset, model, source, run, exported, report, population, state, compatibility, elapsed):
    trainer, config = state["trainer"], state["config"]
    rows = validate_per_user(trainer.recent5_rows, population)
    native_result = {key: float(value) for key, value in state["test_result"].items()}
    means = {metric: math.fsum(row["metrics"][metric] for row in rows) / len(rows) for metric in rows[0]["metrics"]}
    native = {metric: native_result[metric.lower()] for metric in means}
    tolerance = .5 * 10 ** (-int(config["metric_decimal_place"])) + 1e-6
    if any(abs(means[key] - native[key]) > tolerance for key in means):
        raise ValueError("Per-user metrics disagree with native RecBole aggregate evaluation")
    if sha256(source) != report["source_sha256"]: raise ValueError("Original source changed during training")
    checkpoint = Path(trainer.saved_model_file).resolve()
    atomic_json(run / "selected_checkpoint.json", {
        "path": str(checkpoint), "sha256": sha256(checkpoint),
        "best_epoch_zero_based": getattr(trainer, "recent5_best_epoch", None),
        "best_valid_score": float(trainer.recent5_best_result[0]),
        "best_valid_result": primitive(trainer.recent5_best_result[1]), "valid_metric": config["valid_metric"],
        "restored_by": "original Trainer.evaluate(load_best_model=True)",
    })
    atomic_json(run / "test_metrics.json", {
        "dataset": "Fashion" if dataset == "fashion" else "Video Games", "model": MODELS[model],
        "counts": {"evaluated_users": len(rows), "expected_users": len(population), "missing_users": 0},
        "metrics": native, "recbole_test_metrics": native_result, "per_user_unrounded_means": means,
        "evaluation": "unchanged native RecBole full-sort; same masks and torch.topk as native collector",
    })
    with (run / "per_user_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (run / "runtime_config.yaml").write_text(yaml.safe_dump(primitive(config.final_config_dict), sort_keys=False), encoding="utf-8")
    atomic_json(run / "run_summary.json", {
        "complete": True, "dataset": dataset, "model": MODELS[model], "context_window": 5,
        "source": str(source), "source_sha256": report["source_sha256"],
        "scientific_difference": {"MAX_ITEM_LIST_LENGTH": {"parent": 50, "recent5": 5}},
        "execution_compatibility": compatibility,
        "runtime_overrides": {"data_path": config["data_path"], "checkpoint_dir": config["checkpoint_dir"]},
        "native_history_mask_supplied": sorted(trainer.recent5_history_mask_supplied),
        "evaluated_users": len(rows), "seed": config["seed"], "elapsed_seconds": elapsed,
        "training_seconds": trainer.recent5_training_seconds,
        "versions": {name: importlib.metadata.version(name) for name in ("recbole", "torch", "numpy", "pandas")},
        "training_script": str(ROOT / f"pipeline/training/train_{model}.py"),
        "training_script_sha256": sha256(ROOT / f"pipeline/training/train_{model}.py"),
        "cuda_device": api["torch"].cuda.get_device_name(0),
    })
    fingerprints = {name: sha256(run / name) for name in EXPORT_FILES if name != "complete.json"}
    atomic_json(run / "complete.json", {"complete": True, "dataset": dataset, "model": model, "files": fingerprints})
    print("Completed recent5 result and statistics export:", export_statistics(run, exported))
