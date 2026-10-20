"""Synthetic Step-05 execution tests; never load model weights or real splits."""
from __future__ import annotations
import ast
import copy
import contextlib
import importlib.util
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import yaml
import submit_stage2_checkpoint_selection as submitter

from checkpoint_selection_shards import (
    aggregate_payloads, atomic_shard_write, make_payload, metrics_from_predictions,
    run_identity, shard_indices, shard_path,
)
from submit_stage2_checkpoint_selection import submission_commands

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("step05", HERE / "05_select_best_checkpoint.py")
step05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step05)


def prediction(index):
    rank = (index % 12) + 1
    return dict(user_id=str(index), target_item_id=f"target{index}",
                target_rank=rank if rank <= 10 else None,
                recommendations=[dict(rank=1, item_id=f"target{index}", score=1.0)] * 10)


class ShardTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(user_id=str(i), history_item_ids=["history"],
                          target_item_id=f"target{i}") for i in range(100)]
        self.predictions = [prediction(i) for i in range(100)]
        self.identity = {"users": 100, "model": "synthetic-model",
                         "settings": {"top_k": 10, "num_beams": 10}}
        self.payloads = []
        for shard in range(8):
            indices = shard_indices(100, 8, shard)
            self.payloads.append(make_payload(self.identity, "checkpoint-1", 8, shard,
                indices, [self.predictions[i] for i in indices], 1.0))

    def aggregate(self, payloads):
        return aggregate_payloads(payloads, self.identity, "checkpoint-1", 8, self.rows)

    def test_partition_deterministic_complete(self):
        first = [shard_indices(100, 8, i) for i in range(8)]
        self.assertEqual(first, [shard_indices(100, 8, i) for i in range(8)])
        self.assertEqual(sorted(i for shard in first for i in shard), list(range(100)))
        full = [shard_indices(109258, 8, i) for i in range(8)]
        self.assertEqual(sum(map(len, full)), 109258)

    def test_exact_metrics_and_order(self):
        metrics, predictions = self.aggregate(self.payloads[::-1])
        self.assertEqual(predictions, self.predictions)
        metrics.pop("elapsed_seconds")
        self.assertEqual(metrics, metrics_from_predictions(self.predictions, 10))
        # Independently check rank contributions, rather than only the shared helper.
        import math
        ranks = [row["target_rank"] for row in self.predictions if row["target_rank"]]
        self.assertEqual(metrics["hit@10"], len(ranks) / 100)
        self.assertAlmostEqual(metrics["ndcg@10"], sum(1/math.log2(r+1) for r in ranks)/100)
        self.assertAlmostEqual(metrics["mrr@10"], sum(1/r for r in ranks)/100)
        self.assertEqual(metrics["recall@10"], metrics["hit@10"])

    def test_missing_shard(self):
        with self.assertRaisesRegex(ValueError, "Missing shards"):
            self.aggregate(self.payloads[:-1])

    def test_duplicate_shard(self):
        with self.assertRaisesRegex(ValueError, "Duplicate shard"):
            self.aggregate(self.payloads + [self.payloads[0]])

    def test_checkpoint_mismatch(self):
        changed = copy.deepcopy(self.payloads)
        changed[0]["checkpoint"] = "checkpoint-2"
        with self.assertRaisesRegex(ValueError, "Checkpoint mismatch"):
            self.aggregate(changed)

    def test_twenty_beam_shard_rejected_by_ten_beam_run(self):
        changed = copy.deepcopy(self.payloads)
        changed[0]["identity"]["settings"]["num_beams"] = 20
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.aggregate(changed)

    def test_validation_and_settings_mismatch(self):
        for key, value in (("validation_sha256", "different"), ("settings", {"top_k": 5})):
            changed = copy.deepcopy(self.payloads)
            changed[0]["identity"][key] = value
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.aggregate(changed)

    def test_membership_count_mapping_and_sums_rejected(self):
        for mutation, message in (
            (lambda p: p["global_indices"].reverse(), "membership"),
            (lambda p: p["predictions"].pop(), "user count"),
            (lambda p: p["predictions"][0].update(user_id="other"), "mapping"),
            (lambda p: p["sums"].update(hit_sum=-1), "sums"),
            (lambda p: p.update(complete=False), "Incomplete"),
        ):
            changed = copy.deepcopy(self.payloads)
            mutation(changed[0])
            with self.assertRaisesRegex(ValueError, message):
                self.aggregate(changed)

    def test_single_shard_compatibility(self):
        indices = shard_indices(100, 1, 0)
        self.assertEqual(indices, list(range(100)))
        payload = make_payload(self.identity, "checkpoint-1", 1, 0,
                               indices, self.predictions, 0)
        metrics, predictions = aggregate_payloads([payload], self.identity,
                                                  "checkpoint-1", 1, self.rows)
        metrics.pop("elapsed_seconds")
        self.assertEqual(metrics, metrics_from_predictions(self.predictions, 10))
        self.assertEqual(predictions, self.predictions)

    def test_atomic_write_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_shard_write(path, self.payloads[0])
            self.assertEqual(json.loads(path.read_text()), self.payloads[0])
            with self.assertRaises(FileExistsError):
                atomic_shard_write(path, self.payloads[1])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_submission_dependency_and_all_checkpoints_per_worker(self):
        for shards in (8, 16):
            run_id, array, aggregate = submission_commands("synthetic.yaml", num_shards=shards)
            self.assertTrue(run_id)
            self.assertIn(f"0-{shards-1}%4", array)
            self.assertIn("afterok:{array_job_id}", aggregate)
            self.assertNotIn("--nodelist", array)
            self.assertNotIn("--gres", array)
            self.assertEqual(array[-1], "worker")
            self.assertEqual(aggregate[-1], "aggregate")

    def test_submission_defaults_and_explicit_concurrency_cli(self):
        for model in ("qwen", "llama"):
            with tempfile.TemporaryDirectory() as directory:
                config_file = Path(directory) / "config.yaml"
                config_file.write_text(yaml.safe_dump({
                    "model": {"name": model}, "tuning": {"stage": 2},
                    "checkpoint_selection": {"max_validation_users": None, "num_beams": 10},
                    "generation": {"num_beams": 10},
                }))
                for override, cap in (([], 4), (["--max-parallel", "2"], 2)):
                    output = io.StringIO()
                    with patch("sys.argv", ["submit", str(config_file), "--dry-run", *override]), \
                            patch.object(submitter.subprocess, "run") as sbatch, \
                            contextlib.redirect_stdout(output):
                        submitter.main()
                    sbatch.assert_not_called()
                    self.assertIn(f"--array 0-7%{cap}", output.getvalue())
                    self.assertIn(" 4 109258 worker", output.getvalue())
                # Refuse stale or inconsistent Step-05/Step-06 configs.
                for stale_section in ("checkpoint_selection", "generation"):
                    config = yaml.safe_load(config_file.read_text())
                    config[stale_section]["num_beams"] = 20
                    config_file.write_text(yaml.safe_dump(config))
                    with patch("sys.argv", ["submit", str(config_file), "--dry-run"]), \
                            contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        submitter.main()
                    config[stale_section]["num_beams"] = 10
                    config_file.write_text(yaml.safe_dump(config))

    def test_stage2_promotion_and_runtime_beam_defaults_for_both_models(self):
        spec = importlib.util.spec_from_file_location("workflow", HERE / "09_tuning_workflow.py")
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)

        def runtime_beams(filename, config):
            tree = ast.parse((HERE / filename).read_text(encoding="utf-8"))
            main = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "main")
            value = next(node.value for node in ast.walk(main)
                         if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "num_beams"
                                 for target in node.targets))
            return eval(compile(ast.Expression(value), filename, "eval"),
                        {"config": config, "nested_value": step05.nested_value})

        for filename in ("05_select_best_checkpoint.py", "06_generate_test_recommendations.py"):
            self.assertEqual(runtime_beams(filename, {}), 10)
        for model in ("qwen", "llama"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = workflow.load_yaml(workflow.DEFAULT_MANIFEST)
                manifest["workflow"]["generated_config_dir"] = str(root / "generated")
                manifest_path = root / "manifest.yaml"
                manifest_path.write_text(yaml.safe_dump(manifest))
                prepare = workflow.build_parser().parse_args([
                    "prepare-stage1-3090", "--manifest", str(manifest_path)])
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare.function(prepare)
                source = (root / "generated/stage1_fast_3090"
                          / f"{model}_lr1e4_r16_a32_d005_e1.yaml")
                # Fixture-only ranking: tests promotion, never claims a Llama winner.
                before = source.read_bytes()
                stage1 = yaml.safe_load(before)
                ranking = root / "ranking.json"
                ranking.write_text(json.dumps({"results": [{
                    "model_key": model, "rank_within_model": 1,
                    "config_file": str(source), "candidate_id": "lr1e4_r16_a32_d005_e1",
                    "experiment": stage1["experiment"]["name"],
                }]}))
                args = workflow.build_parser().parse_args([
                    "promote-stage2", "--ranking", str(ranking), "--model-key", model,
                    "--manifest", str(manifest_path),
                    "--output-dir", str(root / "promoted")])
                with contextlib.redirect_stdout(io.StringIO()):
                    args.function(args)
                generated = root / "promoted/stage2_fast_3090" / f"{model}_lr1e4_r16_a32_d005_e1_full.yaml"
                config = yaml.safe_load(generated.read_text())
                self.assertIsNone(config["checkpoint_selection"]["max_validation_users"])
                self.assertEqual(config["checkpoint_selection"]["batch_size"], 4)
                self.assertEqual(config["checkpoint_selection"]["max_history_length"], 5)
                for filename, section in (("05_select_best_checkpoint.py", "checkpoint_selection"),
                                          ("06_generate_test_recommendations.py", "generation")):
                    self.assertEqual(config[section]["num_beams"], 10)
                    self.assertEqual(runtime_beams(filename, config), 10)
                expected_generation = dict(stage1["generation"], num_beams=10)
                self.assertEqual(config["generation"], expected_generation)
                self.assertEqual(config["model"], stage1["model"])
                self.assertEqual(config["lora"], stage1["lora"])
                self.assertEqual(source.read_bytes(), before)


class Vector:
    def __init__(self, data): self.data = data
    def tolist(self): return self.data
    def __getitem__(self, key): return Vector(self.data[key])


class Scalar:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class Matrix:
    def __init__(self, rows): self.rows = rows; self.shape = (len(rows), len(rows[0]))
    def to(self, device): return self
    def __getitem__(self, key):
        row, columns = key
        return Vector(self.rows[row][columns])


class BatchTests(unittest.TestCase):
    def test_cpu_aggregation_writes_downstream_schema_for_both_models(self):
        for model in ("Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output"
                checkpoints = []
                for number in (500, 1000):
                    checkpoint = output / f"checkpoint-{number}"
                    checkpoint.mkdir(parents=True)
                    (checkpoint / "adapter_config.json").write_text("{}")
                    checkpoints.append(checkpoint)
                records = [dict(user_id=str(i), history_item_ids=["history"],
                                target_item_id=f"target{i}") for i in range(3)]
                validation = root / "validation.jsonl"
                validation.write_text("\n".join(json.dumps(row) for row in records))
                metadata = root / "metadata.json"
                metadata.write_text("{}")
                catalogue = root / "catalogue.json"
                catalogue.write_text('{"items": []}')
                config = {
                    "experiment": {"name": "synthetic", "dataset": "synthetic", "seed": 42},
                    "model": {"name": model},
                    "training": {"output_dir": str(output), "max_sequence_length": 512},
                    "data": {"validation_split_file": str(validation),
                             "metadata_file": str(metadata), "catalogue_file": str(catalogue)},
                    "checkpoint_selection": {"top_k": 10, "num_beams": 10,
                        "batch_size": 4, "max_validation_users": None,
                        "tie_breakers": ["hit", "mrr"]},
                }
                config_file = root / "config.yaml"
                config_file.write_text(yaml.safe_dump(config))
                settings = dict(top_k=10, num_beams=10, length_penalty=1.0,
                    max_sequence_length=512, max_history_length=5, exclude_history=True,
                    generation_batch_size=4, do_sample=False, early_stopping=True,
                    num_return_sequences=10, max_new_tokens="catalogue_max_item_tokens+1")
                identity = run_identity(config, records, checkpoints, settings)
                shard_directory = output / "validation_selection" / "shards" / "synthetic"
                for checkpoint in checkpoints:
                    for index in range(2):
                        indices = shard_indices(3, 2, index)
                        atomic_shard_write(
                            shard_path(shard_directory, identity, checkpoint, 2, index),
                            make_payload(identity, checkpoint, 2, index, indices,
                                         [prediction(i) for i in indices], 1.0))
                args = ["step05", str(config_file), "--aggregate", "--num-shards", "2",
                        "--shard-run", "synthetic", "--expected-users", "3"]
                with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                    step05.main()
                summary_file = output / "validation_selection" / "checkpoint_selection.json"
                summary = json.loads(summary_file.read_text())
                self.assertEqual(summary["base_model"], model)
                self.assertEqual(summary["validation_users"], 3)
                self.assertEqual(summary["selection_tie_breakers"], ["hit@10", "mrr@10"])
                self.assertEqual(len(summary["all_checkpoints"]), 2)
                self.assertEqual(summary["best_checkpoint"], str(checkpoints[0]))
                self.assertTrue((output / "experiment_summary.json").is_file())
                for checkpoint in checkpoints:
                    saved = output / "validation_selection" / f"{checkpoint.name}_predictions.jsonl"
                    predictions = [json.loads(line) for line in saved.read_text().splitlines()]
                    self.assertEqual([row["user_id"] for row in predictions], ["0", "1", "2"])
                # An interrupted rerun must not replace a previously complete summary.
                previous = summary_file.read_bytes()
                missing = shard_path(shard_directory, identity, checkpoints[0], 2, 0)
                missing.rename(missing.with_suffix(".partial"))
                with patch("sys.argv", args), self.assertRaisesRegex(ValueError, "missing"):
                    step05.main()
                self.assertEqual(summary_file.read_bytes(), previous)
                bad_count = args[:-1] + ["109258"]
                with patch("sys.argv", bad_count), self.assertRaisesRegex(ValueError, "Expected 109258"):
                    step05.main()

    def test_real_batch_generation_and_output_mapping(self):
        class Tokenizer:
            pad_token_id = 0
            def apply_chat_template(self, messages, **kwargs):
                return messages[1]["content"]
            def __call__(self, prompts, **kwargs):
                self.prompts = prompts
                self.kwargs = kwargs
                indices = [int(re.search(r"\d+\. item(\d+) \|", text).group(1)) for text in prompts]
                return {"input_ids": Matrix([[0, i, 9] for i in indices]),
                        "attention_mask": Matrix([[0, 1, 1] for _ in indices])}
        class Model:
            device = "cpu"
            def __init__(self): self.batches = []
            def generate(self, input_ids, attention_mask, **kwargs):
                self.batches.append(len(input_ids.rows))
                assert kwargs["num_beams"] == 10 and kwargs["do_sample"] is False
                assert attention_mask.rows == [[0, 1, 1] for _ in input_ids.rows]
                rows = [row + [token] for row in input_ids.rows
                        for token in (100 + row[1], 999)]
                return SimpleNamespace(sequences=Matrix(rows),
                    sequences_scores=[Scalar(score) for _ in input_ids.rows for score in (2, 1)])
        class History:
            def build_callback(self, prompt_length, blocked_items_by_batch):
                assert prompt_length == 3
                self.blocked = blocked_items_by_batch
                return lambda *args: [1]
        model = Model()
        tokenizer = Tokenizer()
        history = History()
        catalogue = SimpleNamespace(item_to_token_sequence={"item": (1,)}, eos_token_id=2,
                                    generated_tokens_to_item=lambda tokens: f"target{tokens[0]-100}")
        records = [dict(user_id=str(i), history_item_ids=[f"item{i}"],
                        target_item_id=f"target{i}") for i in range(5)]
        metrics, predictions = step05.evaluate_adapter(
            model, tokenizer, records, {}, catalogue, history, 5, 512, 2, 2, 10, 1.0)
        self.assertEqual(model.batches, [2, 2, 1])
        self.assertEqual([row["user_id"] for row in predictions], list(map(str, range(5))))
        self.assertTrue(all(row["target_rank"] == 1 for row in predictions))
        self.assertEqual(metrics["hit@2"], 1.0)
        self.assertTrue(tokenizer.kwargs["padding"])
        self.assertEqual(history.blocked, [{"item4"}])
        class FakeOOM(Exception):
            pass
        with patch.object(step05, "torch", SimpleNamespace(
                cuda=SimpleNamespace(OutOfMemoryError=FakeOOM)), create=True), \
                patch.object(model, "generate", side_effect=FakeOOM):
            with self.assertRaisesRegex(RuntimeError, "generation_batch_size=2.*batch-size 1"):
                step05.evaluate_adapter(model, tokenizer, records, {}, catalogue,
                                        history, 5, 512, 2, 2, 10, 1.0)

    def test_constraint_cache_preserves_user_and_batch_exclusions(self):
        class Trie:
            calls = 0
            def get_node(self, tokens):
                self.calls += 1
                return SimpleNamespace(children={1: None, 2: None})
        trie = Trie()
        catalogue = SimpleNamespace(item_to_token_sequence={"a": (1,), "b": (2,)},
            token_sequence_to_item={(1,): "a", (2,): "b"}, trie=trie, eos_token_id=9)
        history = step05.HistoryAwareConstraint(catalogue)
        callback = history.build_callback(0, [{"a"}, {"b"}])
        self.assertEqual(callback(0, Vector([])), [2])
        self.assertEqual(callback(1, Vector([])), [1])
        self.assertEqual(callback(0, Vector([])), [2])
        self.assertEqual(trie.calls, 2)
        next_batch = history.build_callback(0, [{"b"}])
        self.assertEqual(next_batch(0, Vector([])), [1])

    def test_checkpoint_discovery_all_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("checkpoint-1000", "checkpoint-500", "checkpoint-2000"):
                path = root / name
                path.mkdir()
                (path / "adapter_config.json").write_text("{}")
            self.assertEqual([p.name for p in step05.find_checkpoints(root)],
                             ["checkpoint-500", "checkpoint-1000", "checkpoint-2000"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_validation_loss_adapter"
            path.mkdir()
            (path / "adapter_config.json").write_text("{}")
            self.assertEqual(step05.find_checkpoints(Path(directory)), [path])


if __name__ == "__main__":
    unittest.main()
