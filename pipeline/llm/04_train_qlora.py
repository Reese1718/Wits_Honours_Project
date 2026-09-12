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
import os
import platform
import sys
from pathlib import Path
from typing import Any

# Helps reduce CUDA memory fragmentation.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
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

        datasets["train"] = datasets["train"].select(
            range(sample_count)
        )

    if max_validation_samples is not None:
        sample_count = min(
            max_validation_samples,
            len(datasets["validation"]),
        )

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


def main() -> None:
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
        "device_map": {
            "": torch.cuda.current_device(),
        },
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

    training_arguments = SFTConfig(
        output_dir=str(output_directory),
        run_name=experiment_name,
        num_train_epochs=float(
            nested_value(
                config,
                "training.epochs",
                default=3,
            )
        ),
        per_device_train_batch_size=int(
            nested_value(
                config,
                "training.train_batch_size",
                default=4,
            )
        ),
        per_device_eval_batch_size=int(
            nested_value(
                config,
                "training.eval_batch_size",
                default=4,
            )
        ),
        gradient_accumulation_steps=int(
            nested_value(
                config,
                "training.gradient_accumulation_steps",
                default=8,
            )
        ),
        learning_rate=float(
            nested_value(
                config,
                "training.learning_rate",
                default=2e-4,
            )
        ),
        weight_decay=float(
            nested_value(
                config,
                "training.weight_decay",
                default=0.0,
            )
        ),
        warmup_ratio=float(
            nested_value(
                config,
                "training.warmup_ratio",
                default=0.03,
            )
        ),
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
    )

    print("Creating QLoRA trainer...")

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

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

    run_summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "base_model": model_name,
        "training_examples": len(datasets["train"]),
        "validation_examples": len(datasets["validation"]),
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
