"""Synthetic Step-06 sharding tests: no weights, GPU jobs, or test metrics."""
from __future__ import annotations
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml
from checkpoint_selection_shards import atomic_shard_write, shard_indices
from recommendation_generation_shards import (
    generation_identity, generation_shard_path, generate_shard, load_and_merge,
    merge_payloads, publish_merged,
)
import submit_stage2_test_generation as submitter

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("step06_sharding_test", HERE / "06_generate_test_recommendations.py")
step06 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = step06
spec.loader.exec_module(step06)


def row(index):
    return dict(record_index=index, user_id=f"u{index}", history=[f"history{index}"],
                target=f"target{index}", recommendations=[f"recommendation{index}"],
                scores=[-1.25], requested_top_k=10, generated_recommendation_count=1)


def settings(batch_size=4):
    return dict(top_k=10, num_beams=10, batch_size=batch_size, max_new_tokens=24,
                max_history_length=5, maximum_input_length=488, exclude_history_items=True,
                length_penalty=1.0, do_sample=False, early_stopping=True, seed=42)


class TestGenerationShards(unittest.TestCase):
    def setUp(self):
        self.records = [dict(user_id=f"u{i}", history_item_ids=[f"history{i}"],
                             target_item_id=f"target{i}") for i in range(100)]
        self.population = [step06.test_record_values(record) for record in self.records]
        self.identity = dict(model="Qwen/Qwen2.5-3B-Instruct", users=100,
            selected_checkpoint={"path": "/synthetic/selected-checkpoint", "files": {"adapter_config.json": "hash"}},
            generation=settings(), test_sha256="test-hash", config_sha256="config-hash")
        self.payloads = []
        for index in range(8):
            indices = shard_indices(100, 8, index)
            self.payloads.append(dict(complete=True, identity=copy.deepcopy(self.identity),
                num_shards=8, shard_index=index, users=len(indices), global_indices=indices,
                predictions=[row(i) for i in indices], elapsed_seconds=1.0))

    def merge(self, payloads):
        return merge_payloads(payloads, self.identity, self.population, 8)

    def test_deterministic_partition_and_order(self):
        partitions = [shard_indices(100, 8, i) for i in range(8)]
        self.assertEqual(partitions, [shard_indices(100, 8, i) for i in range(8)])
        self.assertEqual(sorted(i for part in partitions for i in part), list(range(100)))
        self.assertEqual(self.merge(self.payloads[::-1]), [row(i) for i in range(100)])
        self.assertEqual(sum(len(shard_indices(109258, 8, i)) for i in range(8)), 109258)

    def test_missing_shard(self):
        with self.assertRaisesRegex(ValueError, "Missing shards"):
            self.merge(self.payloads[:-1])

    def test_duplicate_shard(self):
        with self.assertRaisesRegex(ValueError, "Duplicate shard"):
            self.merge(self.payloads + [self.payloads[0]])

    def test_identity_mismatches(self):
        for key, value in (("model", "different-model"),
                           ("selected_checkpoint", {"path": "different"}),
                           ("test_sha256", "other-test"), ("config_sha256", "other-config"),
                           ("generation", dict(settings(), num_beams=20)),
                           ("generation", dict(settings(), max_new_tokens=12))):
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(self.payloads)
                changed[0]["identity"][key] = value
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    self.merge(changed)

    def test_incomplete_membership_and_mapping(self):
        for mutate, message in (
            (lambda p: p.update(complete=False), "Incomplete shard"),
            (lambda p: p["global_indices"].reverse(), "membership"),
            (lambda p: p.update(users=0), "membership"),
            (lambda p: p["predictions"].pop(), "count mismatch"),
            (lambda p: p["predictions"][0].update(user_id="wrong-user"), "mapping"),
            (lambda p: p["predictions"][0].update(record_index=1), "mapping"),
        ):
            changed = copy.deepcopy(self.payloads)
            mutate(changed[0])
            with self.assertRaisesRegex(ValueError, message):
                self.merge(changed)

    def test_single_shard_matches_normal_rows_and_cli_defaults(self):
        with patch("sys.argv", ["step06", "config.yaml"]):
            args = step06.parse_arguments()
        self.assertEqual((args.num_shards, args.shard_index, args.shard_run), (1, 0, None))
        indices = shard_indices(100, 1, 0)
        payload = dict(self.payloads[0], num_shards=1, users=100, global_indices=indices,
                       predictions=[row(i) for i in indices])
        self.assertEqual(merge_payloads([payload], self.identity, self.population, 1),
                         [row(i) for i in range(100)])

    def test_worker_preserves_global_indices_and_publishes_only_on_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.json"
            calls = []
            def generate(batch):
                calls.append([i for i, record in batch])
                return [row(i) for i, record in batch]
            with contextlib.redirect_stdout(io.StringIO()):
                payload = generate_shard(path, self.identity, self.records, self.population,
                                          8, 3, 4, generate)
            self.assertEqual([i for batch in calls for i in batch], shard_indices(100, 8, 3))
            self.assertEqual(json.loads(path.read_text())["predictions"], payload["predictions"])
            with self.assertRaises(FileExistsError):
                generate_shard(path, self.identity, self.records, self.population, 8, 3, 4, generate)
            failed = Path(directory) / "failed.json"
            def fail(batch): raise RuntimeError("interrupted")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                generate_shard(failed, self.identity, self.records, self.population, 8, 0, 4, fail)
            self.assertFalse(failed.exists())

    def test_file_completeness_and_step07_prediction_loader(self):
        # Only execute Step-07's pure file parser, never its evaluation/main.
        tree = ast.parse((HERE / "07_evaluate_recommendations.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "load_predictions")
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
            names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
        namespace = {"json": json}
        exec(compile(ast.fix_missing_locations(module), "step07_parser", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = root / "shards"
            for payload in self.payloads:
                atomic_shard_write(generation_shard_path(shards, self.identity, 8,
                                                        payload["shard_index"]), payload)
            merged, _ = load_and_merge(shards, self.identity, self.population, 8)
            final = root / "final/test_predictions.jsonl"
            publish_merged(final, merged, {"completed_record_count": 100})
            self.assertEqual(namespace["load_predictions"](final), {i: row(i) for i in range(100)})
            original = final.read_bytes()
            with self.assertRaises(FileExistsError):
                publish_merged(final, [], {})
            self.assertEqual(final.read_bytes(), original)
            publish_merged(final, merged, {"completed_record_count": 100}, overwrite=True)
            # Incomplete temporary files are ignored, but extra completed JSON is rejected.
            (shards / ".partial-job").write_text("incomplete")
            self.assertEqual(len(load_and_merge(shards, self.identity, self.population, 8)[0]), 100)
            (shards / "duplicate.json").write_text(json.dumps(self.payloads[0]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_and_merge(shards, self.identity, self.population, 8)

    def test_targets_do_not_change_prompt_messages(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): return messages
        original = self.records[0]
        changed = dict(original, target_item_id="another-held-out-target")
        prompts = []
        for record in (original, changed):
            _, history, _ = step06.test_record_values(record)
            prompts.append(step06.build_chat_prompt(Tokenizer(), history, {}, 5))
        self.assertEqual(*prompts)
        self.assertNotIn("target0", json.dumps(prompts))
        self.assertNotIn("another-held-out-target", json.dumps(prompts))

    def test_submission_resources_exclusions_concurrency_and_dependency(self):
        for parallel in (4, 2):
            run_id, worker, merge = submitter.submission_commands("config.yaml", max_parallel=parallel,
                                                                 exclude="another-bad-node")
            self.assertTrue(run_id)
            self.assertIn(f"0-7%{parallel}", worker)
            self.assertIn("bigbatch", worker)
            self.assertIn("mscluster74,another-bad-node", worker)
            self.assertIn("mscluster74,another-bad-node", merge)
            self.assertIn("--exclusive", worker)
            self.assertNotIn("--exclusive", merge)
            self.assertIn("afterok:{array_job_id}", merge)
            self.assertFalse(any("--gres" in part or "--nodelist" in part for part in worker))
        first = submitter.submission_commands("config.yaml")[0]
        self.assertNotEqual(first, submitter.submission_commands("config.yaml")[0])

    def test_submission_cli_preserves_config_batch_unless_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "config.yaml"
            config_file.write_text(yaml.safe_dump({
                "tuning": {"stage": 2}, "generation": {"num_beams": 10, "batch_size": 3}}))
            for override, expected in (([], 3), (["--generation-batch-size", "5"], 5)):
                output = io.StringIO()
                with patch("sys.argv", ["submit", str(config_file), "--dry-run", *override]), \
                        patch.object(submitter.subprocess, "run") as sbatch, \
                        contextlib.redirect_stdout(output):
                    submitter.main()
                sbatch.assert_not_called()
                self.assertIn(f" {expected} 109258 worker no", output.getvalue())
                self.assertIn("--array 0-7%4", output.getvalue())

    def test_cpu_merge_main_for_both_models_and_selected_checkpoint_identity(self):
        for model in ("Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "selected-by-step05"
                checkpoint.mkdir()
                adapter = checkpoint / "adapter_config.json"
                adapter.write_text("{}")
                selection = root / "checkpoint_selection.json"
                selection.write_text(json.dumps({"best_checkpoint": str(checkpoint),
                                                 "max_prompt_history_length": 5}))
                test = root / "test.jsonl"
                test.write_text("\n".join(json.dumps(record) for record in self.records))
                metadata = root / "metadata.json"
                metadata.write_text("{}")
                catalogue = root / "catalogue.json"
                catalogue.write_text("{}")
                output = root / "output"
                config = dict(experiment={"dataset": "Synthetic", "seed": 42},
                    model={"name": model}, training={"output_dir": str(output), "max_sequence_length": 512},
                    data={"test_split_file": str(test), "metadata_file": str(metadata),
                          "catalogue_file": str(catalogue)}, generation={"num_beams": 10,
                          "checkpoint_selection_file": str(selection)}, tuning={"stage": 2})
                config_file = root / "config.yaml"
                config_file.write_text(yaml.safe_dump(config))
                identity = generation_identity(config, model, checkpoint, selection, test,
                    metadata, catalogue, self.population, settings())
                shards = output / "test_predictions/shards/synthetic"
                for payload in self.payloads:
                    payload = dict(payload, identity=identity)
                    atomic_shard_write(generation_shard_path(shards, identity, 8,
                                                            payload["shard_index"]), payload)
                args = ["step06", str(config_file), "--merge", "--num-shards", "8",
                        "--shard-run", "synthetic", "--expected-users", "100"]
                with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                    step06.main()
                final = output / "test_predictions/test_predictions.jsonl"
                self.assertEqual([json.loads(line) for line in final.read_text().splitlines()],
                                 [row(i) for i in range(100)])
                summary = json.loads((final.parent / "generation_summary.json").read_text())
                self.assertEqual(summary["selected_checkpoint"], str(checkpoint))
                self.assertEqual(summary["base_model"], model)
                self.assertEqual(summary["num_beams"], 10)
                with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaises(FileExistsError):
                    step06.main()
                with patch("sys.argv", [*args[:-1], "109258"]), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaisesRegex(ValueError, "Expected 109258 test users"):
                    step06.main()
                # Adapter replacement is detected; no unsafe checkpoint loading involved.
                adapter.write_text('{"changed": true}')
                with patch("sys.argv", [*args, "--overwrite"]), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaisesRegex(ValueError, "identity mismatch"):
                    step06.main()


if __name__ == "__main__":
    unittest.main()
