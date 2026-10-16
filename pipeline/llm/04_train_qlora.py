"""
Fine-tune a causal language model as a sequential recommender using QLoRA.

The script:

1. Loads the base model using 4-bit NF4 quantization.
2. Prepares it for k-bit training.
3. Adds LoRA adapters to its linear layers.
4. Loads the validated train and validation SFT datasets.
5. Trains only on assistant completion tokens.
6. Saves checkpoints, metrics, tokenizer, and the best validation-loss adapter.

Checkpoint selection using recommendation NDCG@10 happens in Step 05.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Helps reduce CUDA memory fragmentation.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from common import (
    load_yaml,
    resolve_project_path,
    set_random_seed,
    write_json,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM recommender using QLoRA."
    )

    parser.add_argument(
        "config",
        help="Path to the QLoRA YAML configuration file.",
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Checkpoint path to resume from, or 'latest' to resume "
            "from the latest checkpoint in the output directory."
        ),
    )

    return parser.parse_args()


def nested_value(
    config: dict[str, Any],
    path: str,
    default: Any = None,
    required: bool = False,
) -> Any:
    """Read a dot-separated value from a nested configuration."""

    current: Any = config

    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            if required:
                raise KeyError(
                    f"Missing required configuration value: {path}"
                )

            return default

        current = current[component]

    return current


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def select_compute_dtype(
    requested_dtype: str,
) -> tuple[torch.dtype, bool, bool]:
    """
    Select the QLoRA computation dtype.

    A100 and other Ampere-or-newer GPUs use bfloat16.
    Older GPUs such as T4 fall back to float16 when auto is selected.
    """

    requested = requested_dtype.lower()

    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False

        return torch.float16, False, True

    if requested in {"bfloat16", "bf16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bfloat16 was requested, but this GPU does not support it. "
                "Use compute_dtype: auto or float16."
            )

        return torch.bfloat16, True, False

    if requested in {"float16", "fp16"}:
        return torch.float16, False, True

    raise ValueError(
        "quantization.compute_dtype must be auto, bfloat16, or float16."
    )


def validate_environment() -> dict[str, Any]:
    """Validate CUDA availability and collect hardware information."""

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Step 04 must run in a GPU runtime."
        )

    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    capability = torch.cuda.get_device_capability(gpu_index)
    total_memory = torch.cuda.get_device_properties(
        gpu_index
    ).total_memory

    return {
        "gpu_index": gpu_index,
        "gpu_name": gpu_name,
        "compute_capability": list(capability),
        "total_memory_bytes": total_memory,
        "total_memory_gib": total_memory / (1024 ** 3),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "local_rank": local_rank,
        "global_rank": global_rank,
        "world_size": world_size,
    }


def validate_input_files(
    train_path: Path,
    validation_path: Path,
) -> None:
    missing = [
        path
        for path in (train_path, validation_path)
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)

        raise FileNotFoundError(
            f"QLoRA dataset files are missing:\n{formatted}"
        )


def convert_to_prompt_completion(
    example: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a system/user/assistant conversation into TRL's
    conversational prompt-completion format.

    Completion-only loss ensures that the model is trained on the
    target item ID rather than on the system and user prompt tokens.
    """

    messages = example.get("messages")

    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(
            "Every SFT example must contain exactly three messages."
        )

    expected_roles = ["system", "user", "assistant"]
    actual_roles = [
        message.get("role")
        for message in messages
        if isinstance(message, dict)
    ]

    if actual_roles != expected_roles:
        raise ValueError(
            f"Incorrect message roles: {actual_roles}. "
            f"Expected: {expected_roles}."
        )

    assistant_message = messages[-1]
    target_item_id = str(example.get("target_item_id", ""))

    if assistant_message.get("content") != target_item_id:
        raise ValueError(
            "Assistant response does not match target_item_id."
        )

    return {
        "prompt": messages[:-1],
        "completion": [assistant_message],
    }


def load_sft_datasets(
    train_path: Path,
    validation_path: Path,
    preprocessing_workers: int,
    max_train_samples: int | None,
    max_validation_samples: int | None,
    shuffle_before_sampling: bool,
    seed: int,
) -> DatasetDict:
    """Load JSONL files and convert them for completion-only SFT."""

    datasets = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(validation_path),
        },
    )

    train_columns = datasets["train"].column_names
    validation_columns = datasets["validation"].column_names

    datasets["train"] = datasets["train"].map(
        convert_to_prompt_completion,
        remove_columns=train_columns,
        num_proc=preprocessing_workers,
        desc="Preparing training conversations",
    )

    datasets["validation"] = datasets["validation"].map(
        convert_to_prompt_completion,
        remove_columns=validation_columns,
        num_proc=preprocessing_workers,
        desc="Preparing validation conversations",
    )

    if max_train_samples is not None:
        sample_count = min(
            max_train_samples,
            len(datasets["train"]),
        )

        if shuffle_before_sampling:
            datasets["train"] = datasets["train"].shuffle(
                seed=seed
            )

        datasets["train"] = datasets["train"].select(
            range(sample_count)
        )

    if max_validation_samples is not None:
        sample_count = min(
            max_validation_samples,
            len(datasets["validation"]),
        )

        if shuffle_before_sampling:
            datasets["validation"] = datasets[
                "validation"
            ].shuffle(seed=seed)

        datasets["validation"] = datasets["validation"].select(
            range(sample_count)
        )

    if len(datasets["train"]) == 0:
        raise ValueError("Training dataset is empty.")

    if len(datasets["validation"]) == 0:
        raise ValueError("Validation dataset is empty.")

    return datasets


def resolve_resume_checkpoint(
    requested_checkpoint: str | None,
) -> str | bool | None:
    if requested_checkpoint is None:
        return None

    if requested_checkpoint.lower() == "latest":
        return True

    checkpoint_path = resolve_project_path(requested_checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {checkpoint_path}"
        )

    return str(checkpoint_path)


def count_parameters(model) -> dict[str, int | float]:
    trainable = 0
    total = 0

    for parameter in model.parameters():
        parameter_count = parameter.numel()
        total += parameter_count

        if parameter.requires_grad:
            trainable += parameter_count

    percentage = (
        100.0 * trainable / total
        if total > 0
        else 0.0
    )

    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percentage": percentage,
    }


def enforce_trainable_fp32(model, stage: str) -> None:
    """Keep optimizer weights FP32 without casting the frozen 4-bit base."""
    counts = {torch.float32: 0, torch.float16: 0, torch.bfloat16: 0}
    bf16_names = []
    converted = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        counts[parameter.dtype] = (
            counts.get(parameter.dtype, 0) + parameter.numel()
        )
        if parameter.dtype == torch.bfloat16:
            bf16_names.append(name)
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)
            converted += parameter.numel()

    print(f"Trainable parameter dtypes ({stage}, before FP32 enforcement):")
    for dtype, count in counts.items():
        print(f"  {dtype}: {count:,} elements")
    for name in bf16_names:
        print(f"  BF16 trainable parameter: {name}")
    print(f"  Converted to FP32: {converted:,} elements")
    print("Trainable parameter dtypes:")
    final_counts = {torch.float32: 0, torch.float16: 0, torch.bfloat16: 0}
    for parameter in model.parameters():
        if parameter.requires_grad:
            final_counts[parameter.dtype] = (
                final_counts.get(parameter.dtype, 0) + parameter.numel()
            )
    for dtype, count in final_counts.items():
        print(f"  {dtype}: {count:,} elements")

    bf16_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.dtype == torch.bfloat16
    ]
    if bf16_trainable:
        raise RuntimeError(
            "BF16 trainable parameters remain before FP16 training: "
            + ", ".join(bf16_trainable[:20])
        )


def main() -> None:
    workflow_started_at = time.monotonic()
    args = parse_arguments()

    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)

    hardware = validate_environment()

    dataset_name = str(
        nested_value(
            config,
            "experiment.dataset",
            required=True,
        )
    )

    experiment_name = str(
        nested_value(
            config,
            "experiment.name",
            required=True,
        )
    )

    seed = int(
        nested_value(
            config,
            "experiment.seed",
            default=42,
        )
    )

    model_name = str(
        nested_value(
            config,
            "model.name",
            required=True,
        )
    )

    trust_remote_code = bool(
        nested_value(
            config,
            "model.trust_remote_code",
            default=False,
        )
    )

    train_path = resolve_project_path(
        nested_value(
            config,
            "data.train_file",
            required=True,
        )
    )

    validation_path = resolve_project_path(
        nested_value(
            config,
            "data.validation_file",
            required=True,
        )
    )

    output_directory = resolve_project_path(
        nested_value(
            config,
            "training.output_dir",
            required=True,
        )
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    validate_input_files(
        train_path=train_path,
        validation_path=validation_path,
    )

    set_random_seed(seed)

    requested_dtype = str(
        nested_value(
            config,
            "quantization.compute_dtype",
            default="auto",
        )
    )

    compute_dtype, use_bf16, use_fp16 = select_compute_dtype(
        requested_dtype
    )

    use_double_quant = bool(
        nested_value(
            config,
            "quantization.use_double_quant",
            default=True,
        )
    )

    quant_type = str(
        nested_value(
            config,
            "quantization.quant_type",
            default="nf4",
        )
    )

    if quant_type != "nf4":
        raise ValueError(
            "This experiment requires NF4 quantization."
        )

    gradient_checkpointing = bool(
        nested_value(
            config,
            "training.gradient_checkpointing",
            default=True,
        )
    )

    preprocessing_workers = int(
        nested_value(
            config,
            "data.preprocessing_workers",
            default=1,
        )
    )

    max_train_samples = nested_value(
        config,
        "data.max_train_samples",
        default=None,
    )

    max_validation_samples = nested_value(
        config,
        "data.max_validation_samples",
        default=None,
    )

    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)

    if max_validation_samples is not None:
        max_validation_samples = int(
            max_validation_samples
        )

    if max_train_samples is not None and max_train_samples <= 0:
        raise ValueError("data.max_train_samples must be positive.")

    if (
        max_validation_samples is not None
        and max_validation_samples <= 0
    ):
        raise ValueError(
            "data.max_validation_samples must be positive."
        )

    shuffle_before_sampling = bool(
        nested_value(
            config,
            "data.shuffle_before_sampling",
            default=False,
        )
    )

    is_primary_process = hardware["global_rank"] == 0

    if is_primary_process:
        print("=" * 72)
        print("QLORA SEQUENTIAL RECOMMENDER TRAINING")
        print("=" * 72)
        print(f"Experiment:       {experiment_name}")
        print(f"Dataset:          {dataset_name}")
        print(f"Base model:       {model_name}")
        print(f"GPU:              {hardware['gpu_name']}")
        print(
            f"GPU memory:       "
            f"{hardware['total_memory_gib']:.2f} GiB"
        )
        print(f"GPU processes:    {hardware['world_size']}")
        print(f"Compute dtype:    {compute_dtype}")
        print(f"Training file:    {train_path}")
        print(f"Validation file:  {validation_path}")
        print(f"Output directory: {output_directory}")
        print()

    print("Loading SFT datasets...")

    datasets = load_sft_datasets(
        train_path=train_path,
        validation_path=validation_path,
        preprocessing_workers=preprocessing_workers,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        shuffle_before_sampling=shuffle_before_sampling,
        seed=seed,
    )

    print(f"Training examples:   {len(datasets['train']):,}")
    print(
        f"Validation examples: "
        f"{len(datasets['validation']):,}"
    )
    print()

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a padding token nor an EOS token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading the base model in 4-bit...")

    model_kwargs: dict[str, Any] = {
        "quantization_config": quantization_config,
        "dtype": compute_dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "device_map": {"": hardware["gpu_index"]},
    }

    attention_implementation = nested_value(
        config,
        "model.attention_implementation",
        default=None,
    )

    if attention_implementation:
        model_kwargs["attn_implementation"] = (
            attention_implementation
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )

    lora_config = LoraConfig(
        r=int(
            nested_value(
                config,
                "lora.rank",
                default=16,
            )
        ),
        lora_alpha=int(
            nested_value(
                config,
                "lora.alpha",
                default=32,
            )
        ),
        lora_dropout=float(
            nested_value(
                config,
                "lora.dropout",
                default=0.05,
            )
        ),
        bias=str(
            nested_value(
                config,
                "lora.bias",
                default="none",
            )
        ),
        task_type="CAUSAL_LM",
        target_modules=nested_value(
            config,
            "lora.target_modules",
            default="all-linear",
        ),
    )

    eval_strategy = str(
        nested_value(
            config,
            "training.eval_strategy",
            default="epoch",
        )
    )

    save_strategy = str(
        nested_value(
            config,
            "training.save_strategy",
            default="epoch",
        )
    )

    if eval_strategy != save_strategy:
        raise ValueError(
            "training.eval_strategy and training.save_strategy must "
            "match when loading the best checkpoint."
        )

    save_total_limit = nested_value(
        config,
        "training.save_total_limit",
        default=None,
    )

    if save_total_limit is not None:
        save_total_limit = int(save_total_limit)

    report_to = nested_value(
        config,
        "training.report_to",
        default="none",
    )

    tf32_enabled = (
        hardware["compute_capability"][0] >= 8
    )

    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled

    train_batch_size = int(
        nested_value(
            config,
            "training.train_batch_size",
            default=4,
        )
    )
    eval_batch_size = int(
        nested_value(
            config,
            "training.eval_batch_size",
            default=4,
        )
    )
    gradient_accumulation_steps = int(
        nested_value(
            config,
            "training.gradient_accumulation_steps",
            default=8,
        )
    )
    training_epochs = float(
        nested_value(
            config,
            "training.epochs",
            default=3,
        )
    )

    if train_batch_size <= 0 or eval_batch_size <= 0:
        raise ValueError("Training and evaluation batch sizes must be positive.")

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "training.gradient_accumulation_steps must be positive."
        )

    if training_epochs <= 0:
        raise ValueError("training.epochs must be positive.")

    effective_batch_size = (
        train_batch_size
        * gradient_accumulation_steps
        * hardware["world_size"]
    )
    batches_per_epoch = math.ceil(
        len(datasets["train"])
        / (train_batch_size * hardware["world_size"])
    )
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )
    estimated_optimizer_steps = math.ceil(
        optimizer_steps_per_epoch * training_epochs
    )
    learning_rate = float(
        nested_value(
            config,
            "training.learning_rate",
            default=2e-4,
        )
    )
    warmup_ratio = float(
        nested_value(
            config,
            "training.warmup_ratio",
            default=0.03,
        )
    )
    warmup_steps = math.ceil(
        estimated_optimizer_steps * warmup_ratio
    )

    if is_primary_process:
        print("Training plan:")
        print(f"  Per-device batch:         {train_batch_size}")
        print(
            "  Gradient accumulation:    "
            f"{gradient_accumulation_steps}"
        )
        print(f"  Effective global batch:   {effective_batch_size}")
        print(f"  Epochs:                   {training_epochs}")
        print(
            "  Estimated optimizer steps: "
            f"{estimated_optimizer_steps:,}"
        )
        print()

    training_arguments = SFTConfig(
        output_dir=str(output_directory),
        run_name=experiment_name,
        num_train_epochs=training_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=float(
            nested_value(
                config,
                "training.weight_decay",
                default=0.0,
            )
        ),
        warmup_steps=warmup_steps,
        lr_scheduler_type=str(
            nested_value(
                config,
                "training.lr_scheduler_type",
                default="cosine",
            )
        ),
        optim=str(
            nested_value(
                config,
                "training.optimizer",
                default="paged_adamw_8bit",
            )
        ),
        max_grad_norm=float(
            nested_value(
                config,
                "training.max_grad_norm",
                default=0.3,
            )
        ),
        logging_steps=int(
            nested_value(
                config,
                "training.logging_steps",
                default=10,
            )
        ),
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=tf32_enabled,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        max_length=int(
            nested_value(
                config,
                "training.max_sequence_length",
                default=2048,
            )
        ),
        eos_token=nested_value(
            config,
            "model.eos_token",
            default=None,
        ),
        packing=bool(
            nested_value(
                config,
                "training.packing",
                default=False,
            )
        ),
        completion_only_loss=True,
        dataset_num_proc=preprocessing_workers,
        dataloader_num_workers=int(
            nested_value(
                config,
                "training.dataloader_workers",
                default=2,
            )
        ),
        report_to=report_to,
        seed=seed,
        data_seed=seed,
        remove_unused_columns=True,
        ddp_find_unused_parameters=(
            False if hardware["world_size"] > 1 else None
        ),
    )

    print("Creating QLoRA trainer...")

    if use_bf16:
        raise RuntimeError(
            "BF16 training is disabled for this project. "
            "Use compute_dtype=float16."
        )

    if not use_fp16:
        raise RuntimeError(
            "FP16 training must be enabled for this project."
        )

    if compute_dtype != torch.float16:
        raise RuntimeError(
            f"Expected torch.float16 compute dtype, got {compute_dtype}."
        )

    # Prepare adapters here so the dtype check includes every trainable weight.
    # No modules_to_save or embedding/lm_head unfreezing is requested.
    model = get_peft_model(model, lora_config)
    enforce_trainable_fp32(model, "before trainer creation")

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
    )

    # Some TRL versions cast quantized-model trainables to BF16 inside
    # SFTTrainer even with fp16=True. Restore FP32 before optimizer creation;
    # FP16 autocast and the base model's NF4/FP16 compute remain unchanged.
    enforce_trainable_fp32(trainer.model, "after trainer creation")

    parameter_statistics = count_parameters(trainer.model)

    print()
    print("Parameter statistics:")
    print(
        f"  Trainable: "
        f"{parameter_statistics['trainable_parameters']:,}"
    )
    print(
        f"  Total:     "
        f"{parameter_statistics['total_parameters']:,}"
    )
    print(
        f"  Percentage:"
        f" {parameter_statistics['trainable_percentage']:.4f}%"
    )
    print()

    resume_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint
    )

    print("Starting QLoRA training...")

    training_result = trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )

    trainer.save_state()
    trainer.save_metrics(
        "train",
        training_result.metrics,
    )

    validation_metrics = trainer.evaluate()

    trainer.save_metrics(
        "validation",
        validation_metrics,
    )

    best_loss_adapter = (
        output_directory / "best_validation_loss_adapter"
    )

    trainer.save_model(str(best_loss_adapter))
    tokenizer.save_pretrained(best_loss_adapter)

    workflow_runtime_seconds = (
        time.monotonic() - workflow_started_at
    )

    run_summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "base_model": model_name,
        "seed": seed,
        "training_examples": len(datasets["train"]),
        "validation_examples": len(datasets["validation"]),
        "training_loss": training_result.metrics.get("train_loss"),
        "validation_loss": validation_metrics.get("eval_loss"),
        "training_metrics": training_result.metrics,
        "validation_metrics": validation_metrics,
        "runtime_seconds": workflow_runtime_seconds,
        "best_validation_loss_checkpoint": (
            trainer.state.best_model_checkpoint
        ),
        "best_validation_loss": (
            trainer.state.best_metric
        ),
        "saved_adapter": str(best_loss_adapter),
        "parameter_statistics": parameter_statistics,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": quant_type,
            "double_quantization": use_double_quant,
            "compute_dtype": str(compute_dtype),
        },
        "lora": {
            "rank": lora_config.r,
            "alpha": lora_config.lora_alpha,
            "dropout": lora_config.lora_dropout,
            "bias": lora_config.bias,
            "target_modules": nested_value(
                config,
                "lora.target_modules",
                default="all-linear",
            ),
        },
        "hardware": hardware,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "peft": package_version("peft"),
            "trl": package_version("trl"),
            "accelerate": package_version("accelerate"),
            "bitsandbytes": package_version("bitsandbytes"),
        },
        "config_file": str(config_path),
        "config": config,
    }

    run_summary["training_plan"] = {
        "world_size": hardware["world_size"],
        "per_device_train_batch_size": train_batch_size,
        "gradient_accumulation_steps": (
            gradient_accumulation_steps
        ),
        "effective_global_batch_size": effective_batch_size,
        "epochs": training_epochs,
        "estimated_optimizer_steps": estimated_optimizer_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "shuffle_before_sampling": shuffle_before_sampling,
    }

    if trainer.is_world_process_zero():
        write_json(
            run_summary,
            output_directory / "run_summary.json",
        )

    print()
    print("=" * 72)
    print("QLORA TRAINING COMPLETE")
    print("=" * 72)
    print(
        f"Best validation-loss checkpoint: "
        f"{trainer.state.best_model_checkpoint}"
    )
    print(
        f"Best validation loss: "
        f"{trainer.state.best_metric}"
    )
    print(f"Saved adapter: {best_loss_adapter}")
    print(
        f"Run summary: "
        f"{output_directory / 'run_summary.json'}"
    )
    print()
    print(
        "Step 05 will evaluate the saved checkpoints using "
        "validation NDCG@10."
    )


if __name__ == "__main__":
    main()
