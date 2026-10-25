"""CPU-only recent5 baseline config, split, export and score instrumentation tests."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import random
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml
import recent5_baselines as core
import recent5_recbole as native


class ConfigTests(unittest.TestCase):
    def test_all_six_are_exact_single_setting_derivatives_and_parents_unchanged(self):
        for dataset in core.DATASETS:
            for model in core.MODELS:
                with self.subTest(dataset=dataset, model=model):
                    parent, recent, config = core.validate_config(dataset, model)
                    original = subprocess.run(["git", "show", f"HEAD:pipeline/configs/{parent.name}"],
                                              capture_output=True, check=True).stdout
                    # Git stores LF; physical Windows files retain their existing CRLF.
                    self.assertEqual(parent.read_bytes().replace(b"\r\n", b"\n"), original)
                    self.assertEqual(recent.read_bytes(), parent.read_bytes().replace(
                        b"MAX_ITEM_LIST_LENGTH: 50", b"MAX_ITEM_LIST_LENGTH: 5"))
                    self.assertEqual(config["seed"], 42)
                    self.assertEqual(Path(config["data_path"]), core.COLAB_DATA_ROOT)
                    self.assertEqual(config["metrics"], ["Recall", "MRR", "NDCG", "Hit"])
                    self.assertEqual(config["topk"], 10)
        _, _, fashion = core.validate_config("fashion", "bert4rec")
        _, _, games = core.validate_config("video_games", "bert4rec")
        self.assertEqual((fashion["mask_ratio"], games["mask_ratio"]), (.1, .2))
        self.assertEqual((fashion["learning_rate"], games["learning_rate"]), (.0001, .00001))

    def test_colab_cli_defaults_delegate_to_existing_training_adapter(self):
        for dataset in core.DATASETS:
            for model in core.MODELS:
                source = core.COLAB_DATA_ROOT / core.DATASETS[dataset][0] / "processed/original.inter"
                with patch("sys.argv", ["recent5", "run", "--dataset", dataset, "--model", model]), \
                        patch.object(core, "resolve_inter", return_value=source) as resolve, \
                        patch.object(native, "train_existing") as train, \
                        contextlib.redirect_stdout(io.StringIO()):
                    core.main()
                resolve.assert_called_once_with(dataset, core.COLAB_DATA_ROOT, None)
                run, exported = core.experiment_paths(dataset, model)
                train.assert_called_once_with(dataset, model, source, source.parent / "llm/splits",
                    core.ROOT / "results/statistics" / dataset / "user_groups.csv", run, exported)
        with patch("sys.argv", ["recent5", "validate"]), \
                patch.object(native, "train_existing") as train, \
                contextlib.redirect_stdout(io.StringIO()):
            core.main()
        train.assert_not_called()
        self.assertFalse(hasattr(core, "submission_commands"))

    def test_source_resolution_no_new_interaction_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Games/processed/Video Games.inter"
            source.parent.mkdir(parents=True)
            source.write_text("source")
            before = list(root.rglob("*.inter"))
            self.assertEqual(core.resolve_inter("video_games", root), source.resolve())
            self.assertEqual(list(root.rglob("*.inter")), before)
            another = root / "Video_Games/processed/Video_Games.inter"
            another.parent.mkdir(parents=True)
            another.write_text("different source")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                core.resolve_inter("video_games", root)


class NativeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = native.recbole_api()

    def fixture(self, root):
        source = root / "Fashion.inter"
        lines = ["user_id:token\titem_id:token\ttimestamp:float"]
        for user, length in (("u0", 5), ("u1", 7), ("u2", 8), ("u3", 10)):
            lines += [f"{user}\t{user}_item{i}\t{i+1}" for i in range(length)]
        source.write_text("\n".join(lines) + "\n")
        histories, _ = core.source_population(source)
        splits = root / "splits"
        splits.mkdir()
        for name, offset in (("validation", -2), ("test", -1)):
            rows = [dict(user_id=user, target_item_id=events[offset][2],
                         history_item_ids=[event[2] for event in events[:offset]])
                    for user, events in sorted(histories.items())]
            (splits / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        return source, splits

    def test_native_targets_contexts_and_effective_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, splits = self.fixture(root)
            before = source.read_bytes()
            for model in core.MODELS:
                with self.subTest(model=model), contextlib.redirect_stdout(io.StringIO()):
                    report, population = native.audit_data("fashion", model, source, splits,
                                                           expected_users=4)
                    self.assertEqual((report["validation_users"], report["test_users"]), (4, 4))
                    self.assertTrue(report["full_vs_recent_targets_equal"])
                    self.assertTrue(report["recent_five_contexts_equal"])
                    self.assertTrue(report["evidence"]["test"]["targets_equal"])
                    self.assertEqual([population[f"u{i}"]["group"] for i in range(4)],
                                     ["sparse", "sparse", "longer_history", "longer_history"])
                    parent, recent, _ = core.validate_config("fashion", model)
                    full_config = native.native_config(self.api, "fashion", model, parent, source, root, cpu=True)
                    recent_config = native.native_config(self.api, "fashion", model, recent, source, root, cpu=True)
                    expected = dict(full_config.final_config_dict, MAX_ITEM_LIST_LENGTH=5)
                    self.assertEqual(recent_config.final_config_dict, expected)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(list(root.glob("*.inter")), [source])
            # Target mismatch must fail, never be 'repaired' by changing the split.
            rows = core.json_rows(splits / "test.jsonl")
            rows[0]["target_item_id"] = "wrong-heldout-item"
            (splits / "test.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "targets/users mismatch"):
                native.audit_data("fashion", "sasrec", source, splits, expected_users=4)

    def test_bert_zero_buffer_guard_preserves_probability_and_other_models(self):
        api = self.api
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.fixture(Path(directory))
            _, recent, _ = core.validate_config("fashion", "bert4rec")
            config = native.native_config(api, "fashion", "bert4rec", recent, source, directory, cpu=True)
            original_init = api["MaskItemSequence"].__init__
            self.assertEqual(api["MaskItemSequence"](config).mask_item_length, 0)
            from recbole.data.interaction import Interaction
            interaction = Interaction({"item_id_list": api["torch"].tensor([[1,2,3,4,5], [6,7,8,9,10]]),
                                       "item_length": api["torch"].tensor([5,5])})
            # Select one mask for row 1 and zero masks for row 2. Native
            # capacity=0 leaves ragged arrays; the guard fixes storage only.
            probabilities = [.99] + [.99, .01, .99, .99, .99] + [.99] * 5
            with patch("recbole.data.transform.random.random", side_effect=probabilities):
                with self.assertRaises(ValueError):
                    api["MaskItemSequence"](config)(SimpleNamespace(num=lambda _: 30), interaction)
            compatibility = {}
            with native.input_and_mask_hooks(api, source, compatibility):
                transform = api["MaskItemSequence"](config)
                self.assertEqual(transform.mask_item_length, 1)
                self.assertEqual(transform.mask_ratio, .1)
                self.assertEqual(config["mask_ratio"], .1)
                self.assertTrue(compatibility["bert_zero_mask_buffer_minimum_one"])
                with patch("recbole.data.transform.random.random", side_effect=probabilities):
                    transformed = transform(SimpleNamespace(num=lambda _: 30), interaction)
                self.assertEqual(tuple(transformed[config["MASK_INDEX"]].shape), (2, 1))
                config["MAX_ITEM_LIST_LENGTH"] = 50
                self.assertEqual(api["MaskItemSequence"](config).mask_item_length, 5)
            self.assertIs(api["MaskItemSequence"].__init__, original_init)

    def test_score_capture_uses_native_ranking_stable_ids_and_no_rng(self):
        torch = self.api["torch"]
        config = {"USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id"}
        population = {
            "b": dict(record_index=0, user_id="b", target="i2", original_history_length=7, group="sparse"),
            "a": dict(record_index=1, user_id="a", target="i4", original_history_length=8, group="longer_history"),
        }
        class Dataset:
            def id2token(self, field, indices):
                tokens = ["PAD", "a", "b"] if field == "user_id" else [f"i{i}" for i in range(12)]
                return np.array([tokens[int(index)] for index in indices])
        interaction = {"user_id": torch.tensor([1, 2])}
        scores = torch.tensor([[float('-inf'), 0, 1, 2, 20, 3, 4, 5, 6, 7, 8, 9],
                               [float('-inf'), 0, 20, 2, 1, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.float)
        original = scores.clone()
        rng = torch.random.get_rng_state().clone()
        rows = native.capture_rows(self.api, interaction, scores, torch.tensor([0, 1]),
                                  torch.tensor([4, 2]), Dataset(), config, population)
        self.assertTrue(torch.equal(scores, original))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual([row["user_id"] for row in rows], ["a", "b"])
        self.assertTrue(all(row["metrics"]["Hit@10"] == 1 for row in rows))
        ordered = core.validate_per_user(rows, population)
        self.assertEqual([row["user_id"] for row in ordered], ["b", "a"])
        for changed in (rows[:-1], rows + [rows[0]], [dict(rows[0], target="wrong"), rows[1]],
                        [dict(rows[0], target_rank=0), rows[1]],
                        [dict(rows[0], metrics={**rows[0]["metrics"], "MRR@10": float("nan")}), rows[1]]):
            with self.assertRaises(ValueError): core.validate_per_user(changed, population)

    def test_trainer_adapter_preserves_fit_evaluation_masks_and_save_calls(self):
        torch = self.api["torch"]
        population = {"u": dict(record_index=0, user_id="u", target="i1", original_history_length=9,
                                group="longer_history")}
        class Dataset:
            def id2token(self, field, indices):
                return np.array(["u"] * len(indices)) if field == "user_id" else np.array([f"i{i}" for i in indices])
        class BaseTrainer:
            def fit(self, *args, **kwargs): self.fit_call = (args, kwargs); return .2, {"ndcg@10": .2}
            def _save_checkpoint(self, epoch, *args, **kwargs): self.saved_epoch = epoch
            def evaluate(self, data, **kwargs):
                self.evaluate_call = kwargs
                self._full_sort_batch_eval(data)
                return {"hit@10": 1.0}
            def _full_sort_batch_eval(self, data):
                return data[0], data[2], torch.tensor([0]), torch.tensor([1])
        state = {"test_dataset_for_capture": Dataset()}
        api = dict(self.api, Trainer=BaseTrainer)
        trainer_class = native.trainer_adapter(api, state, population)
        trainer = trainer_class()
        with tempfile.TemporaryDirectory() as directory:
            trainer.config = {"checkpoint_dir": directory, "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id"}
            trainer.saved_model_file = str(Path(directory) / "own-checkpoint.pth")
            scores = torch.tensor([[float('-inf'), 20, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.float)
            batch = ({"user_id": torch.tensor([1])}, None, scores)
            data = type("Data", (tuple,), {})(batch)
            data.dataset = Dataset()
            trainer.fit("train", "valid", saved=True, show_progress=True)
            self.assertEqual(trainer.fit_call, (("train", "valid"), {"saved": True, "show_progress": True}))
            trainer._save_checkpoint(3)
            self.assertEqual((trainer.saved_epoch, trainer.recent5_best_epoch), (3, 3))
            trainer.evaluate(data, load_best_model=False)
            self.assertNotIn("test_result", state)
            trainer.evaluate(data, load_best_model=True)
            self.assertEqual(trainer.evaluate_call["load_best_model"], True)
            self.assertEqual(trainer.recent5_rows[0]["original_history_length"], 9)
            self.assertEqual(trainer.recent5_history_mask_supplied, {False})


class BertMaskSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = native.recbole_api()
        from recbole.data.interaction import Interaction
        cls.Interaction = Interaction
        cls.dataset = SimpleNamespace(num=lambda _: 30)

    def config(self, directory, dataset="fashion", model="bert4rec", recent=True):
        parent, child = core.configs(dataset, model)
        return native.native_config(self.api, dataset, model, child if recent else parent,
                                    Path(directory) / "original.inter", directory, cpu=True)

    def batch(self, lengths, width=5):
        torch = self.api["torch"]
        return self.Interaction({
            "item_id_list": torch.tensor([list(range(1, length+1)) + [0]*(width-length)
                                           for length in lengths]),
            "item_length": torch.tensor(lengths),
        })

    def assert_supervision(self, transformed, original, transform):
        indices = transformed[transform.MASK_INDEX]
        self.assertTrue(bool((indices > 0).any(dim=1).all()))
        for row in range(len(original)):
            for slot, encoded in enumerate(indices[row].tolist()):
                if encoded == 0:
                    continue
                position = 0 if encoded == 5 else encoded
                item = original[transform.ITEM_SEQ][row, position].item()
                self.assertGreater(item, 0)  # never supervise padding
                self.assertLess(position, original[transform.ITEM_SEQ_LEN][row].item())
                self.assertEqual(transformed[transform.POS_ITEMS][row, slot].item(), item)
                self.assertEqual(transformed[transform.MASK_ITEM_SEQ][row, position].item(), 30)
                negative = transformed[transform.NEG_ITEMS][row, slot].item()
                self.assertGreater(negative, 0)
                self.assertNotIn(negative, original[transform.ITEM_SEQ][row].tolist())

    def test_native_storage_only_fix_reproduces_nan_ce_loss(self):
        torch = self.api["torch"]
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            transform = self.api["MaskItemSequence"](config)
            transform.mask_item_length = 1  # reproduce the incomplete old fix
            model = self.api["BERT4Rec"](config, self.dataset)
            model.mask_item_length = 1
            for probabilities in ([.99] * 6, [.99, .01, .99, .99, .99, .99]):
                with patch("recbole.data.transform.random.random", side_effect=probabilities):
                    transformed = transform(self.dataset, self.batch([5]))
                self.assertEqual((transformed[transform.MASK_INDEX] > 0).sum().item(), 0)
                with torch.no_grad():
                    self.assertFalse(bool(torch.isfinite(model.calculate_loss(transformed))))

    def test_empty_and_index_zero_only_masks_use_real_targets_and_native_negatives(self):
        torch = self.api["torch"]
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            compatibility = {}
            with native.input_and_mask_hooks(self.api, Path(directory)/"original.inter", compatibility):
                transform = self.api["MaskItemSequence"](config)
                self.assertEqual((transform.mask_ratio, transform.mask_item_length), (.1, 1))
                for length in (1, 2, 3, 5):
                    for zero_only in (False, True):
                        original = self.batch([length])
                        probabilities = [.99] + ([.01] + [.99]*(length-1) if zero_only else [.99]*length)
                        with patch("recbole.data.transform.random.random", side_effect=probabilities), \
                                patch.object(transform, "_neg_sample", wraps=transform._neg_sample) as negative:
                            transformed = transform(self.dataset, copy.deepcopy(original))
                        self.assertEqual(negative.call_count, 2 if zero_only else 1)
                        self.assert_supervision(transformed, original, transform)
                        self.assertTrue(torch.equal(transformed[transform.ITEM_SEQ], original[transform.ITEM_SEQ]))
                        expected = length-1 if length > 1 else 5
                        self.assertEqual(transformed[transform.MASK_INDEX].item(), expected)
                self.assertTrue(compatibility["bert_minimum_mask_storage_one"])
                self.assertTrue(compatibility["bert_forced_nonempty_training_masks"])
                self.assertEqual(compatibility["forced_mask_count"], 8)
                self.assertEqual(compatibility["forced_position_zero_mask_count"], 2)
                with self.assertRaisesRegex(ValueError, "empty history"):
                    transform(self.dataset, self.batch([0]))

    def test_usable_stochastic_masks_are_unchanged(self):
        torch = self.api["torch"]
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            transform_before = self.api["MaskItemSequence"](config)
            transform_before.mask_item_length = 1
            for probabilities in ([.99, .99, .99, .01, .99, .99], [.99] + [.01]*5):
                rng = random.getstate()
                with patch("recbole.data.transform.random.random", side_effect=probabilities):
                    expected = transform_before(self.dataset, self.batch([5]))
                expected_rng = random.getstate()
                random.setstate(rng)
                compatibility = {}
                with native.input_and_mask_hooks(self.api, Path(directory)/"original.inter", compatibility):
                    transform = self.api["MaskItemSequence"](config)
                    with patch("recbole.data.transform.random.random", side_effect=probabilities):
                        actual = transform(self.dataset, self.batch([5]))
                self.assertEqual(random.getstate(), expected_rng)
                for field in expected.columns:
                    self.assertTrue(torch.equal(expected[field], actual[field]), field)
                self.assertEqual(compatibility["forced_mask_count"], 0)

    def test_native_training_loader_is_safe_but_full_sort_transforms_are_unchanged(self):
        torch = self.api["torch"]
        original_eval_init = self.api["FullSortEvalDataLoader"].__init__
        original_call = self.api["MaskItemSequence"].__call__
        with tempfile.TemporaryDirectory() as directory:
            source, _ = NativeAuditTests().fixture(Path(directory))
            config = self.config(directory)
            compatibility = {}
            with native.input_and_mask_hooks(self.api, source, compatibility):
                self.api["init_seed"](42, True)
                data = self.api["create_dataset"](config)
                train, valid, test = self.api["data_preparation"](config, data)
                self.assertTrue(train.transform._recent5_mask_safety)
                for loader in (valid, test):
                    self.assertFalse(loader.transform._recent5_mask_safety)
                    raw = loader.dataset[np.arange(len(loader.dataset))]
                    with patch("recbole.data.transform.random.random", return_value=.99):
                        expected = original_call(loader.transform, loader.dataset, copy.deepcopy(raw))
                        actual, history, _, _ = loader.collate_fn(list(range(len(loader.dataset))))
                    self.assertIsNone(history)
                    for field in expected.columns:
                        self.assertTrue(torch.equal(expected[field], actual[field]), field)
                    self.assertEqual((actual[loader.transform.MASK_INDEX] > 0).sum().item(), 0)
                self.assertEqual(compatibility["forced_mask_count"], 0)
                with patch("recbole.data.transform.random.random", return_value=.99):
                    batch = train.collate_fn([0])
                self.assertGreater((batch[train.transform.MASK_INDEX] > 0).sum().item(), 0)
                self.assertEqual(compatibility["forced_mask_count"], 1)
        self.assertIs(self.api["FullSortEvalDataLoader"].__init__, original_eval_init)

    def test_repeated_training_batch_preflight_has_finite_native_ce_loss(self):
        torch = self.api["torch"]
        original_call = self.api["MaskItemSequence"].__call__
        original_gather = self.api["BERT4Rec"].multi_hot_embed
        original_reconstruct = self.api["BERT4Rec"].reconstruct_test_data
        original_full_sort = self.api["BERT4Rec"].full_sort_predict
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                for dataset, ratio in (("fashion", .1), ("video_games", .2)):
                    config = self.config(directory, dataset)
                    compatibility = {}
                    self.api["init_seed"](42, True)
                    with native.input_and_mask_hooks(self.api, Path(directory)/"original.inter", compatibility):
                        transform = self.api["MaskItemSequence"](config)
                        model = self.api["BERT4Rec"](config, self.dataset)
                        self.assertEqual((transform.mask_ratio, model.mask_ratio), (ratio, ratio))
                        self.assertEqual(model.mask_item_length, 1)
                        model.train()  # actual CE forward with original dropout; no optimizer or training
                        for iteration in range(48):
                            lengths = [1] if iteration % 3 == 0 else [1, 2, 3, 4, 5]
                            original = self.batch(lengths)
                            # Include guaranteed all-empty draws as well as native stochastic draws.
                            context = (patch("recbole.data.transform.random.random", return_value=.99)
                                       if iteration % 7 == 0 else contextlib.nullcontext())
                            with context:
                                transformed = transform(self.dataset, copy.deepcopy(original))
                            self.assert_supervision(transformed, original, transform)
                            self.assertGreater((transformed[transform.MASK_INDEX] > 0).sum().item(), 0)
                            with torch.no_grad():
                                loss = model.calculate_loss(transformed)
                            self.assertTrue(bool(torch.isfinite(loss)), (dataset, iteration))
                        self.assertGreater(compatibility["forced_mask_count"], 0)
                        self.assertGreater(compatibility["forced_position_zero_mask_count"], 0)
                        model.eval()
                        evaluation = self.batch([1, 2, 5])
                        with torch.no_grad():
                            expected_scores = model.full_sort_predict(evaluation)
                    with torch.no_grad():
                        self.assertTrue(torch.equal(model.full_sort_predict(evaluation), expected_scores))
            self.assertIs(self.api["MaskItemSequence"].__call__, original_call)
            self.assertIs(self.api["BERT4Rec"].multi_hot_embed, original_gather)
            self.assertIs(self.api["BERT4Rec"].reconstruct_test_data, original_reconstruct)
            self.assertIs(self.api["BERT4Rec"].full_sort_predict, original_full_sort)
        finally:
            torch.set_num_threads(threads)

    def test_length_fifty_and_other_models_remain_native(self):
        from recbole.data.transform import construct_transform, Equal
        torch = self.api["torch"]
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, recent=False)
            original = self.api["MaskItemSequence"](config)
            with patch("recbole.data.transform.random.random", return_value=.99):
                expected = original(self.dataset, self.batch([1, 5], width=50))
            compatibility = {}
            with native.input_and_mask_hooks(self.api, Path(directory)/"original.inter", compatibility):
                transform = self.api["MaskItemSequence"](config)
                self.assertEqual(transform.mask_item_length, 5)
                model = self.api["BERT4Rec"](config, self.dataset)
                self.assertFalse(model._recent5_mask_safety)
                self.assertEqual(model.position_embedding.num_embeddings, 50)
                with patch("recbole.data.transform.random.random", return_value=.99):
                    actual = transform(self.dataset, self.batch([1, 5], width=50))
                for field in expected.columns:
                    self.assertTrue(torch.equal(expected[field], actual[field]), field)
                for model_name in ("sasrec", "gru4rec"):
                    other = self.config(directory, model=model_name)
                    equal = construct_transform(other)
                    self.assertIsInstance(equal, Equal)
                    batch = self.batch([1, 5])
                    self.assertIs(equal(self.dataset, batch), batch)
            self.assertEqual(compatibility, {})


class ExportTests(unittest.TestCase):
    def test_allowlist_integrity_and_no_weights_or_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            for name in core.EXPORT_FILES:
                if name != "complete.json": (run / name).write_text('{}\n')
            (run / "checkpoints").mkdir()
            (run / "checkpoints/model.pth").write_text("never export weights")
            (run / "dataset-cache.pth").write_text("never export cache")
            core.atomic_json(run / "complete.json", {"complete": True,
                "files": {name: core.sha256(run / name) for name in core.EXPORT_FILES if name != "complete.json"}})
            destination = core.export_statistics(run, root / "statistics")
            self.assertEqual({path.name for path in destination.iterdir()}, set(core.EXPORT_FILES))
            self.assertFalse(any(path.suffix in (".pth", ".pt", ".bin") for path in destination.rglob("*")))
            with self.assertRaises(FileExistsError): core.export_statistics(run, destination)
            (run / "test_metrics.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                core.export_statistics(run, root / "other")
            core.atomic_json(run / "complete.json", {"complete": True, "files": {}})
            with self.assertRaisesRegex(ValueError, "allowlist"):
                core.validate_completion(run)

    def test_groups_use_original_lengths_not_recent_context(self):
        histories = {"sparse": [(i, i, f"i{i}") for i in range(7)],
                     "long": [(i, i, f"j{i}") for i in range(8)]}
        validation = {user: history[-2][2] for user, history in histories.items()}
        test = {user: history[-1][2] for user, history in histories.items()}
        population, _ = core.evaluation_population(histories, validation, test)
        self.assertEqual(population["sparse"]["group"], "sparse")
        self.assertEqual(population["long"]["group"], "longer_history")
        self.assertEqual(population["sparse"]["original_history_length"], 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user_groups.csv"
            path.write_text("record_index,user_id,original_history_length,group\n0,long,5,sparse\n1,sparse,5,sparse\n")
            with self.assertRaisesRegex(ValueError, "ORIGINAL"):
                core.evaluation_population(histories, validation, test, groups_file=path)


if __name__ == "__main__": unittest.main()
