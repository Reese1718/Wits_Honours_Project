"""CPU-only Stage-2 execution tests. Never instantiate or train a model."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from stage2_execution import (
    CHECKPOINT_MARKER, Stage2CheckpointCallback, identity_for,
    stage2_completed, stage2_resume, write_json,
)


class Stage2ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.config = {
            "experiment": {"name": "qwen_stage2_winner"},
            "model": {"name": "Qwen/Qwen2.5-3B-Instruct"},
            "training": {"epochs": 1, "output_dir": str(self.output)},
            "tuning": {"stage": 2, "candidate_id": "winner"},
        }

    def initialize(self):
        self.assertIsNone(stage2_resume(self.config, self.output))

    def checkpoint(self, step, complete=True):
        path = self.output / f"checkpoint-{step}"
        path.mkdir()
        if complete:
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt",
                         "training_args.bin", "adapter_model.safetensors"):
                (path / name).write_bytes(b"synthetic state placeholder")
            write_json(path / "trainer_state.json", {"global_step": step})
            write_json(path / "adapter_config.json", {
                "base_model_name_or_path": self.config["model"]["name"],
            })
            write_json(path / CHECKPOINT_MARKER, {
                "identity": identity_for(self.config, self.output), "global_step": step,
            })
        return path

    def completion_artifacts(self):
        adapter = self.output / "best_validation_loss_adapter"
        adapter.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
            (adapter / name).write_bytes(b"test")
        for name in ("train_results.json", "validation_results.json"):
            write_json(self.output / name, {})
        write_json(self.output / "trainer_state.json", {
            "epoch": 1.0, "global_step": 4576, "max_steps": 4576,
        })
        write_json(self.output / "run_summary.json", {
            "config": self.config, "experiment_name": self.config["experiment"]["name"],
            "base_model": self.config["model"]["name"], "saved_adapter": str(adapter),
            "training_plan": {"estimated_optimizer_steps": 4577},
        })

    def test_stage1_never_auto_resumes(self):
        self.checkpoint(500)
        self.config["tuning"]["stage"] = 1
        self.assertIsNone(stage2_resume(self.config, self.output))
        self.assertFalse(stage2_completed(self.config, self.output))

    def test_empty_output_starts_fresh(self):
        self.initialize()
        self.initialize()

    def test_checkpoint_500(self):
        self.initialize()
        path = self.checkpoint(500)
        self.assertEqual(stage2_resume(self.config, self.output), str(path))

    def test_llama_uses_same_native_resume(self):
        self.config["model"]["name"] = "meta-llama/Llama-3.2-3B-Instruct"
        self.config["experiment"]["name"] = "llama_stage2_winner"
        self.initialize()
        path = self.checkpoint(1000)
        self.assertEqual(stage2_resume(self.config, self.output), str(path))

    def test_latest_checkpoint_1000(self):
        self.initialize()
        self.checkpoint(500)
        path = self.checkpoint(1000)
        self.assertEqual(stage2_resume(self.config, self.output), str(path))

    def test_incomplete_newest_falls_back(self):
        self.initialize()
        path = self.checkpoint(500)
        self.checkpoint(1000, complete=False)
        self.assertEqual(stage2_resume(self.config, self.output), str(path))

    def test_corrupt_json_falls_back(self):
        self.initialize()
        path = self.checkpoint(500)
        broken = self.checkpoint(1000)
        (broken / "trainer_state.json").write_text("broken", encoding="utf-8")
        self.assertEqual(stage2_resume(self.config, self.output), str(path))

    def test_all_incomplete_starts_fresh(self):
        self.initialize()
        self.checkpoint(500, complete=False)
        self.assertIsNone(stage2_resume(self.config, self.output))

    def test_missing_optimizer_falls_back(self):
        self.initialize()
        path = self.checkpoint(500)
        broken = self.checkpoint(1000)
        (broken / "optimizer.pt").unlink()
        self.assertEqual(stage2_resume(self.config, self.output), str(path))

    def test_completed_run_is_detected(self):
        self.completion_artifacts()
        self.assertTrue(stage2_completed(self.config, self.output))
        (self.output / "validation_results.json").unlink()
        self.assertFalse(stage2_completed(self.config, self.output))

    def test_summary_alone_does_not_mean_complete(self):
        self.completion_artifacts()
        write_json(self.output / "trainer_state.json", {
            "epoch": 0.7, "global_step": 3000, "max_steps": 4577,
        })
        self.assertFalse(stage2_completed(self.config, self.output))

    def test_other_experiment_and_external_checkpoint_rejected(self):
        self.initialize()
        self.checkpoint(500)
        changed = copy.deepcopy(self.config)
        changed["model"]["name"] = "meta-llama/Llama-3.2-3B-Instruct"
        with self.assertRaises(RuntimeError):
            stage2_resume(changed, self.output)
        with self.assertRaises(RuntimeError):
            stage2_resume(self.config, self.output, str(self.output.parent))

    def test_checkpoint_cannot_be_adopted_without_identity(self):
        self.checkpoint(500)
        with self.assertRaises(RuntimeError):
            stage2_resume(self.config, self.output)

    def test_callback_marks_only_after_native_save(self):
        self.initialize()
        path = self.checkpoint(500)
        (path / CHECKPOINT_MARKER).unlink()
        callback = Stage2CheckpointCallback(self.config, self.output)
        control = SimpleNamespace(should_save=False)
        callback.on_epoch_end(None, None, control)
        self.assertTrue(control.should_save)
        callback.on_save(SimpleNamespace(output_dir=str(self.output)),
                         SimpleNamespace(global_step=500, is_world_process_zero=True), control)
        self.assertEqual(stage2_resume(self.config, self.output), str(path))


if __name__ == "__main__":
    unittest.main()
