"""Native Trainer checkpoint selection and completion checks for Stage 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

IDENTITY = "stage2_identity.json"
CHECKPOINT_MARKER = "stage2_checkpoint_complete.json"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity_for(config: dict, output: Path) -> dict:
    return {
        "output_directory": str(output.resolve()),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def nonempty(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def stage2_completed(config: dict, output: Path) -> bool:
    """Require the existing normal summary, final Trainer state and adapter."""
    if config.get("tuning", {}).get("stage") != 2:
        return False
    try:
        summary = read_json(output / "run_summary.json")
        if summary.get("config") != config:
            raise RuntimeError("Run summary belongs to a different experiment/config.")
        if summary.get("experiment_name") != config["experiment"]["name"]:
            raise RuntimeError("Run summary experiment name does not match.")
        adapter = Path(summary["saved_adapter"]).resolve()
        if (
            adapter != (output / "best_validation_loss_adapter").resolve()
            or adapter.parent != output.resolve()
        ):
            raise RuntimeError("Completed adapter is outside this experiment.")
        state = read_json(output / "trainer_state.json")
        total_steps = state["max_steps"]
        return (
            float(state["epoch"]) >= float(config["training"]["epochs"])
            and state["global_step"] >= total_steps > 0
            and summary["base_model"] == config["model"]["name"]
            and all(nonempty(output / name) for name in (
                "train_results.json", "validation_results.json", "trainer_state.json",
            ))
            and nonempty(adapter / "adapter_config.json")
            and nonempty(adapter / "tokenizer_config.json")
            and any(nonempty(adapter / name) for name in (
                "adapter_model.safetensors", "adapter_model.bin",
            ))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def checkpoint_step(path: Path, config: dict, output: Path) -> int | None:
    if path.is_symlink() or path.resolve().parent != output.resolve():
        raise RuntimeError(f"Checkpoint is outside the current output directory: {path}")
    try:
        marker = read_json(path / CHECKPOINT_MARKER)
        if marker.get("identity") != identity_for(config, output):
            raise RuntimeError(f"Checkpoint belongs to another experiment: {path}")
        required = (
            "adapter_config.json", "trainer_state.json", "optimizer.pt",
            "scheduler.pt", "rng_state.pth", "scaler.pt", "training_args.bin",
        )
        if not all(nonempty(path / name) for name in required):
            return None
        if not any(nonempty(path / name) for name in (
            "adapter_model.safetensors", "adapter_model.bin",
        )):
            return None
        state = read_json(path / "trainer_state.json")
        step = state["global_step"]
        if not isinstance(step, int) or step <= 0 or path.name != f"checkpoint-{step}":
            return None
        if marker.get("global_step") != step:
            return None
        adapter = read_json(path / "adapter_config.json")
        if adapter.get("base_model_name_or_path") != config["model"]["name"]:
            raise RuntimeError(f"Checkpoint belongs to another model: {path}")
        return step
    except (OSError, ValueError, KeyError, TypeError):
        return None


def stage2_resume(config: dict, output: Path, requested: str | None = None) -> str | None:
    if config.get("tuning", {}).get("stage") != 2:
        return None  # Stage 1 never automatically resumes.
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = identity_for(config, output)
    metadata = output / IDENTITY
    if metadata.exists():
        if read_json(metadata) != identity:
            raise RuntimeError("Stage-2 output belongs to a different experiment/config.")
    else:
        if any(p.name != ".stage2.lock" for p in output.iterdir()):
            raise RuntimeError("Nonempty Stage-2 output has no identity; review it before resuming.")
        write_json(metadata, identity)

    latest = get_last_checkpoint(str(output))
    paths = list(output.glob("checkpoint-*"))
    valid = []
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"External checkpoint link is forbidden: {path}")
        if not path.is_dir():
            continue
        step = checkpoint_step(path, config, output)
        if step is not None:
            valid.append((step, path))
        else:
            print(f"Stage-2 resume check: ignoring incomplete checkpoint {path.name}.")
    if requested and requested.lower() != "latest":
        selected = Path(requested).resolve()
        matches = [path for _, path in valid if path.resolve() == selected]
        if not matches:
            raise RuntimeError("Explicit checkpoint is outside this run or incomplete.")
        chosen = matches[0]
    elif valid:
        # Use native discovery when valid, otherwise choose the latest verified
        # Trainer global_step, never filesystem timestamps.
        chosen = next((p for _, p in valid if str(p) == latest), None)
        if chosen is None:
            chosen = max(valid, key=lambda item: item[0])[1]
    else:
        print("Stage-2 resume check: no valid checkpoint found; starting fresh from step 0.")
        return None
    print(f"Stage-2 resume check: resuming from {chosen.name}.")
    print("Resuming Stage 2 from existing Trainer state.")
    return str(chosen)


class Stage2CheckpointCallback(TrainerCallback):
    def __init__(self, config: dict, output: Path):
        self.identity = identity_for(config, output)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Preserve the final epoch state too, including a timeout during the
        # later summary/adapter export. Periodic saving remains every 500 steps.
        control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_json(checkpoint / CHECKPOINT_MARKER, {
                "identity": self.identity, "global_step": state.global_step,
            })
        return control
