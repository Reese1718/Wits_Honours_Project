"""Fashion pipeline tests: CPU data preparation, synthetic summaries, no weights."""
from __future__ import annotations

from collections import Counter
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml
from common import build_recommendation_messages
import fashion_qlora_workflow as fashion
from export_qlora_statistics import export_bundle, user_groups, validate_per_user, REQUIRED_METRICS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("fashion_workflow_test", HERE / "09_tuning_workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
spec = importlib.util.spec_from_file_location("fashion_step05_test", HERE / "05_select_best_checkpoint.py")
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def dump_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class FashionWorkflowTests(unittest.TestCase):
    def prepare_configs(self, root, dataset="fashion", implementation=workflow):
        manifest = implementation.load_yaml(ROOT / f"pipeline/configs/llm/tuning/{dataset}_recent5_three_sweep.yaml")
        manifest["workflow"]["generated_config_dir"] = str(root / "generated")
        manifest_path = root / "manifest.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest))
        args = implementation.build_parser().parse_args([
            "prepare-stage1-3090", "--manifest", str(manifest_path), "--overwrite"])
        with contextlib.redirect_stdout(io.StringIO()):
            args.function(args)
        return manifest_path, sorted((root / "generated/stage1_fast_3090").glob("*.yaml"))

    def test_fashion_config_generation_and_shared_validation_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, paths = self.prepare_configs(root)
            self.assertEqual(len(paths), 6)
            records = [dict(user_id=str(i), history_item_ids=["history"], target_item_id="target")
                       for i in range(1300)]
            validation = root / "validation.jsonl"
            dump_rows(validation, records)
            populations = []
            for path in paths:
                config = workflow.load_yaml(path)
                self.assertIn("fashion_recent5_three", config["experiment"]["name"])
                self.assertTrue(config["data"]["train_file"].startswith("/datasets/dnaidoo3/Fashion/"))
                self.assertTrue(config["data"]["train_file"].endswith("train_recent5_three_per_user.jsonl"))
                self.assertIsNone(config["data"]["max_train_samples"])
                self.assertEqual(config["data"]["max_validation_samples"], 1000)
                self.assertEqual(config["training"]["train_batch_size"], 2)
                self.assertEqual(config["training"]["gradient_accumulation_steps"], 32)
                self.assertEqual(config["training"]["max_sequence_length"], 512)
                self.assertEqual(config["quantization"]["compute_dtype"], "float16")
                self.assertTrue(config["quantization"]["use_double_quant"])
                for section in ("checkpoint_selection", "generation"):
                    self.assertEqual(config[section]["num_beams"], 10)
                    self.assertEqual(config[section]["max_history_length"], 5)
                    self.assertEqual(config[section]["batch_size"], 4)
                settings = config["checkpoint_selection"]
                population = selection.load_validation_records(validation,
                    settings["max_validation_users"], settings["sampling_seed"])
                populations.append([row["user_id"] for row in population])
            self.assertTrue(all(population == populations[0] for population in populations))
            self.assertEqual(len(populations[0]), 1000)
            self.assertEqual(len(selection.load_validation_records(validation, None, 42)), 1300)

    def test_video_games_generation_byte_identical_to_reference(self):
        # Read the committed reference implementation; no checkout or file mutation.
        source = subprocess.run(["git", "show", "HEAD:pipeline/llm/09_tuning_workflow.py"],
                                check=True, capture_output=True, text=True).stdout
        namespace = {"__file__": str(HERE / "09_tuning_workflow.py"), "__name__": "vg_reference"}
        exec(compile(source, "vg_reference", "exec"), namespace)
        reference = SimpleNamespace(**namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, paths = self.prepare_configs(root, "video_games", reference)
            before = {path.name: path.read_bytes() for path in paths}
            self.prepare_configs(root, "video_games")
            self.assertEqual(before, {path.name: path.read_bytes() for path in paths})
            ranking = root / "ranking.json"
            winners = []
            for model in ("qwen", "llama"):
                path = next(path for path in paths if path.name.startswith(model))
                config = workflow.load_yaml(path)
                winners.append(dict(model_key=model, rank_within_model=1, config_file=str(path),
                    candidate_id=config["tuning"]["candidate_id"], experiment=config["experiment"]["name"]))
            dump_json(ranking, {"results": winners})
            arguments = ["promote-stage2", "--manifest", str(manifest_path), "--ranking", str(ranking),
                         "--output-dir", str(root / "promoted"), "--overwrite"]
            for implementation in (reference, workflow):
                args = implementation.build_parser().parse_args(arguments)
                with contextlib.redirect_stdout(io.StringIO()): args.function(args)
                observed = {path.name: path.read_bytes() for path in (root / "promoted").rglob("*.yaml")}
                if implementation is reference: expected = observed
                else: self.assertEqual(observed, expected)

    def test_full_data_promotion_and_scaled_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, paths = self.prepare_configs(root)
            train = root / "train.jsonl"
            dump_rows(train, [{}] * 7894)
            winners = []
            for model in ("qwen", "llama"):
                path = next(path for path in paths if path.name.startswith(model))
                config = workflow.load_yaml(path)
                config["data"]["train_file"] = str(train)
                path.write_text(yaml.safe_dump(config))
                winners.append(dict(model_key=model, rank_within_model=1, config_file=str(path),
                    candidate_id=config["tuning"]["candidate_id"], experiment=config["experiment"]["name"]))
            ranking = root / "ranking.json"
            dump_json(ranking, {"results": winners})
            args = workflow.build_parser().parse_args(["promote-stage2", "--manifest", str(manifest_path),
                "--ranking", str(ranking), "--output-dir", str(root / "promoted")])
            with contextlib.redirect_stdout(io.StringIO()): args.function(args)
            promoted = list((root / "promoted").rglob("*.yaml"))
            self.assertEqual(len(promoted), 2)
            for path in promoted:
                config = workflow.load_yaml(path)
                self.assertIn("fashion_recent5_three", config["experiment"]["name"])
                self.assertNotIn("video_games", config["experiment"]["name"])
                self.assertIsNone(config["data"]["max_train_samples"])
                self.assertIsNone(config["data"]["max_validation_samples"])
                self.assertIsNone(config["checkpoint_selection"]["max_validation_users"])
                self.assertEqual(config["training"]["save_steps"], 50)
                self.assertEqual(config["training"]["save_total_limit"], 2)
                self.assertEqual(config["tuning"]["full_training_records"], 7894)
                self.assertEqual(config["training"]["epochs"], 1)
                for section in ("checkpoint_selection", "generation"):
                    self.assertEqual(config[section]["num_beams"], 10)
            config["data"]["train_file"] = str(root / "tiny.jsonl")
            dump_rows(Path(config["data"]["train_file"]), [{}] * 130)
            self.assertEqual(workflow.stage2_save_interval(config, {"stage2_save_steps": 50,
                "stage2_adapt_save_steps": True}), (1, 130))
            dump_rows(Path(config["data"]["train_file"]), [{}])
            with self.assertRaisesRegex(ValueError, "two optimizer steps"):
                workflow.stage2_save_interval(config, {"stage2_adapt_save_steps": True})

    def test_rank_order_and_dependency_plans_without_submitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, paths = self.prepare_configs(root)
            rows = {}
            for model in ("qwen", "llama"):
                model_paths = [path for path in paths if path.name.startswith(model)]
                for i, path in enumerate(model_paths):
                    rows[path] = dict(model_key=model, candidate_id=path.stem, experiment=path.stem,
                        config_file=str(path), validation_sampling_seed=20261008,
                        validation_sample_sha256="same-population", validation_users=1000,
                        full_validation_set_used=False, validation_loss=[.3, .2, .1][i],
                        **{"validation_ndcg@10": .1, "validation_hit@10": .2,
                           "validation_mrr@10": [.1, .3, .3][i]})
            args = SimpleNamespace(config_dir=str(root / "generated/stage1_fast_3090"), config_list=None,
                model_key=None, allow_incomplete=False, require_full_validation=False,
                output_dir=str(root / "ranked"))
            with patch.object(workflow, "summary_row", side_effect=lambda path: copy.deepcopy(rows[path])), \
                    contextlib.redirect_stdout(io.StringIO()):
                workflow.rank_runs(args)
            ranked = json.loads((root / "ranked/tuning_results.json").read_text())
            self.assertEqual(ranked["tie_breakers"], ["validation_hit@10", "validation_mrr@10", "validation_loss"])
            for model in ("qwen", "llama"):
                expected = [path for path in paths if path.name.startswith(model)][2]
                self.assertEqual(ranked["best_by_model"][model]["config_file"], str(expected))
            plan = fashion.stage1_plan(paths)
            self.assertEqual(len(plan), 7)
            self.assertEqual(len(plan[-1][1]), 6)
            with self.assertRaisesRegex(ValueError, "three unique"):
                fashion.stage1_plan([paths[0]] * 6)
            with patch.object(fashion.subprocess, "run") as submit, contextlib.redirect_stdout(io.StringIO()):
                fashion.submit_plan(plan)
            submit.assert_not_called()
            stage2_paths = []
            for model in ("qwen", "llama"):
                original = next(path for path in paths if path.name.startswith(model))
                config = workflow.load_yaml(original)
                config["tuning"]["stage"] = 2
                path = root / f"{model}_stage2.yaml"
                path.write_text(yaml.safe_dump(config))
                stage2_paths.append(path)
            final = fashion.final_plan(stage2_paths)
            self.assertEqual(len(final), 9)
            self.assertEqual(final[-1][1], ["qwen_evaluate", "llama_evaluate"])
            for model in ("qwen", "llama"):
                self.assertEqual(next(node[1] for node in final if node[0] == f"{model}_generate"), [f"{model}_select"])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                fashion.final_plan([stage2_paths[0], stage2_paths[0]])

    def test_preparation_preserves_targets_recent_five_and_three_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inter = root / "processed/Fashion.inter"
            inter.parent.mkdir()
            metadata = {}
            lines = ["user_id:token\titem_id:token\ttimestamp:float"]
            for user, length in enumerate((5, 6, 7, 8, 9)):
                for i in range(length):
                    item = f"u{user}_item{i}"
                    metadata[item] = {"title": f"Title {item}"}
                    lines.append(f"u{user}\t{item}\t{i+1}")
            inter.write_text("\n".join(lines) + "\n")
            dump_json(root / "exports/Fashion_metadata_lookup.json", metadata)
            # Suppress subprocess stdout while executing actual Steps 01-03.
            original_run = subprocess.run
            def quiet(*args, **kwargs):
                kwargs["stdout"] = subprocess.DEVNULL
                return original_run(*args, **kwargs)
            with patch.object(fashion.subprocess, "run", side_effect=quiet), contextlib.redirect_stdout(io.StringIO()):
                counts = fashion.prepare_data(root)
            self.assertEqual(counts, {"train": 14, "validation": 5, "test": 5})
            splits = root / "processed/llm/splits"
            before = {path.name: path.read_bytes() for path in splits.iterdir()}
            sft = root / "processed/llm/sft"
            examples = fashion.read_rows(sft / "train_recent5_three_per_user.jsonl")
            self.assertLessEqual(max(Counter(row["user_id"] for row in examples).values()), 3)
            for row in examples:
                self.assertLessEqual(row["history_length"], 5)
                self.assertNotIn(row["target_item_id"], row["history_item_ids"])
                self.assertEqual(row["messages"][:-1], build_recommendation_messages(
                    row["history_item_ids"], metadata, 5))
            with patch.object(fashion.subprocess, "run", side_effect=quiet), contextlib.redirect_stdout(io.StringIO()):
                fashion.prepare_data(root)
            self.assertEqual(before, {path.name: path.read_bytes() for path in splits.iterdir()})
            self.assertEqual([row["target_item_id"] for row in fashion.read_rows(splits / "test.jsonl")],
                             [f"u{i}_item{length-1}" for i, length in enumerate((5, 6, 7, 8, 9))])


class StatisticsExportTests(unittest.TestCase):
    def records(self):
        return [dict(user_id="sparse", history_item_ids=[f"a{i}" for i in range(6)], target_item_id="t1"),
                dict(user_id="long", history_item_ids=[f"b{i}" for i in range(7)], target_item_id="t2")]

    def fixture(self, root, model):
        records = self.records()
        dump_rows(root / "test.jsonl", records)
        dump_rows(root / "validation.jsonl", records)
        dump_rows(root / "train.jsonl", [{}] * 3)
        output = root / model
        config = dict(experiment={"dataset": "Fashion", "name": model}, model={"name": model},
            training={"output_dir": str(output)}, tuning={"stage": 2, "model_key": model,
            "workflow": "fashion_recent5_three_fast"}, generation={},
            data={"train_file": str(root / "train.jsonl"), "test_split_file": str(root / "test.jsonl"),
                  "validation_split_file": str(root / "validation.jsonl")})
        config_path = root / f"{model}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        checkpoint = output / "checkpoint-124"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"never export this")
        selection_path = output / "validation_selection/checkpoint_selection.json"
        checkpoint_metrics = selection_path.parent / "checkpoint-124_metrics.json"
        dump_json(checkpoint_metrics, {"users": 2})
        dump_json(selection_path, dict(validation_users=2, full_validation_set_used=True,
            base_model=model, experiment=model, best_checkpoint=str(checkpoint), all_checkpoints=[
                dict(checkpoint_name="checkpoint-124", metrics_file=str(checkpoint_metrics))]))
        dump_json(output / "run_summary.json", dict(dataset="Fashion", base_model=model,
            experiment_name=model, training_examples=3))
        for name in ("train_results.json", "validation_results.json", "all_results.json"):
            dump_json(output / name, {})
        predictions = output / "test_predictions"
        dump_json(predictions / "test_metrics.json", dict(dataset="Fashion", model=model,
            selected_checkpoint=str(checkpoint), counts=dict(evaluated_users=2, test_records=2,
                prediction_records=2, missing_prediction_records=0), metrics={key: .1 for key in REQUIRED_METRICS},
            coverage={f"CatalogueCoverage@{k}": .1 for k in (1,5,10)}))
        dump_json(predictions / "generation_summary.json", dict(base_model=model, num_beams=10,
            completed_record_count=2, test_record_count=2, selected_checkpoint=str(checkpoint)))
        rows = [dict(record_index=i, user_id=record["user_id"], target=record["target_item_id"],
            metrics={key: .1 for key in REQUIRED_METRICS}) for i, record in enumerate(records)]
        dump_rows(predictions / "per_user_metrics.jsonl", rows)
        dump_rows(predictions / "test_predictions.jsonl", [{"never": "export raw predictions"}])
        dump_rows(selection_path.parent / "checkpoint-124_predictions.jsonl", [{}])
        return config_path, rows

    def test_sparse_boundary_complete_alignment_and_no_large_files(self):
        groups = user_groups(self.records())
        self.assertEqual([row["original_history_length"] for row in groups], [7, 8])
        self.assertEqual([row["group"] for row in groups], ["sparse", "longer_history"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = [self.fixture(root, model)[0] for model in ("qwen", "llama")]
            with contextlib.redirect_stdout(io.StringIO()): export_bundle(configs, root / "bundle")
            files = [path.relative_to(root / "bundle").as_posix() for path in (root / "bundle").rglob("*") if path.is_file()]
            self.assertEqual(len(files), 21)  # one group CSV + ten small artifacts per model
            self.assertTrue(all(not path.endswith((".safetensors", ".bin", ".pt")) for path in files))
            self.assertFalse(any("predictions.jsonl" in path for path in files))
            with (root / "bundle/user_groups.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["group"], "sparse")
            with self.assertRaises(FileExistsError): export_bundle(configs, root / "bundle")

    def test_reject_incomplete_duplicate_or_misaligned_per_user_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows = self.fixture(Path(directory), "qwen")
            for changed in (rows[:-1], rows + [rows[0]], [dict(rows[0], user_id="wrong"), rows[1]],
                            [dict(rows[0], metrics={}), rows[1]]):
                with self.assertRaises(ValueError): validate_per_user(changed, self.records())
            with self.assertRaises(ValueError): user_groups([self.records()[0]] * 2)

    def test_reject_different_populations_and_incomplete_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qwen, _ = self.fixture(root, "qwen")
            llama, _ = self.fixture(root, "llama")
            path = root / "llama/validation_selection/checkpoint_selection.json"
            summary = json.loads(path.read_text())
            summary["full_validation_set_used"] = False
            dump_json(path, summary)
            with self.assertRaisesRegex(ValueError, "full Fashion validation"):
                export_bundle([qwen, llama], root / "bundle")
            self.assertFalse((root / "bundle").exists())


if __name__ == "__main__":
    unittest.main()
