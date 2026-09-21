import os
import sys
import time
import threading

import psutil
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed


# ==========================================================
# Project Configuration
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)

if len(sys.argv) != 2:
    raise ValueError(
        "Usage: python train_bert4rec.py <config_file>"
    )

CONFIG_FILE = os.path.join(
    PIPELINE_DIR,
    "configs",
    sys.argv[1],
)

if not os.path.isfile(CONFIG_FILE):
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_FILE}"
    )


# ==========================================================
# Runtime Resource Monitoring
# ==========================================================

def start_resource_monitor(interval=30):
    """
    Periodically report CPU RAM and GPU memory usage.

    This is diagnostic only. It does not modify the training
    process or model behaviour.
    """

    process = psutil.Process(os.getpid())

    def monitor():
        while True:
            try:
                ram = psutil.virtual_memory()
                process_ram = process.memory_info().rss

                message = (
                    f"[RESOURCE] "
                    f"RAM={ram.percent:.1f}% "
                    f"available={ram.available / 1024**3:.2f}GB "
                    f"process={process_ram / 1024**3:.2f}GB"
                )

                if torch.cuda.is_available():
                    gpu_allocated = (
                        torch.cuda.memory_allocated() / 1024**3
                    )

                    gpu_reserved = (
                        torch.cuda.memory_reserved() / 1024**3
                    )

                    message += (
                        f" "
                        f"GPU_allocated={gpu_allocated:.2f}GB "
                        f"GPU_reserved={gpu_reserved:.2f}GB"
                    )

                print(message, flush=True)

                time.sleep(interval)

            except Exception as error:
                print(
                    f"[RESOURCE MONITOR ERROR] {error}",
                    flush=True,
                )
                break

    monitor_thread = threading.Thread(
        target=monitor,
        daemon=True,
    )

    monitor_thread.start()

    return monitor_thread


# ==========================================================
# Main Training Function
# ==========================================================

def train_bert4rec():
    print("=" * 70)
    print("BERT4Rec Training")
    print("=" * 70)

    print("\nUsing configuration:")
    print(CONFIG_FILE)

    # ======================================================
    # Load RecBole configuration
    # ======================================================

    config = Config(
        model="BERT4Rec",
        config_file_list=[CONFIG_FILE],
    )

    # ======================================================
    # Print important configuration
    # ======================================================

    print("\nBERT4Rec Configuration:")
    print("-" * 70)

    config_parameters = [
        "dataset",
        "hidden_size",
        "inner_size",
        "n_layers",
        "n_heads",
        "hidden_dropout_prob",
        "attn_dropout_prob",
        "hidden_act",
        "layer_norm_eps",
        "initializer_range",
        "mask_ratio",
        "loss_type",
        "learning_rate",
        "train_batch_size",
        "epochs",
        "stopping_step",
        "seed",
        "reproducibility",
        "device",
    ]

    for parameter in config_parameters:
        print(f"{parameter}: {config[parameter]}")

    print("-" * 70)

    # ======================================================
    # Set random seed before dataset preparation
    # ======================================================

    init_seed(
        config["seed"],
        config["reproducibility"],
    )

    # ======================================================
    # Initialise logging
    # ======================================================

    init_logger(config)

    # ======================================================
    # Hardware
    # ======================================================

    print("\nHardware:")
    print("-" * 70)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print("GPU memory:", round(gpu_memory, 2), "GB")

    print("RecBole device:", config["device"])
    print("-" * 70)

    # ======================================================
    # Initial Runtime Resource Check
    # ======================================================

    print("\nInitial runtime resource usage:")
    print("-" * 70)

    initial_ram = psutil.virtual_memory()
    initial_process_ram = (
        psutil.Process(os.getpid()).memory_info().rss
    )

    print(
        f"System RAM usage: "
        f"{initial_ram.percent:.1f}%"
    )

    print(
        f"System RAM available: "
        f"{initial_ram.available / 1024**3:.2f} GB"
    )

    print(
        f"Python process RAM: "
        f"{initial_process_ram / 1024**3:.2f} GB"
    )

    if torch.cuda.is_available():
        print(
            f"GPU allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        print(
            f"GPU reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )

    print("-" * 70)

    # ======================================================
    # Create Dataset
    # ======================================================

    print("\nCreating dataset...")

    dataset = create_dataset(config)

    print("\nDataset created successfully.")
    print(dataset)

    # ======================================================
    # Runtime Resource Check After Dataset Creation
    # ======================================================

    print("\nRuntime resources after dataset creation:")
    print("-" * 70)

    ram = psutil.virtual_memory()
    process_ram = (
        psutil.Process(os.getpid()).memory_info().rss
    )

    print(f"System RAM usage: {ram.percent:.1f}%")
    print(
        f"System RAM available: "
        f"{ram.available / 1024**3:.2f} GB"
    )
    print(
        f"Python process RAM: "
        f"{process_ram / 1024**3:.2f} GB"
    )

    print("-" * 70)

    # ======================================================
    # Prepare Train / Validation / Test Sets
    # ======================================================

    print("\nPreparing train/validation/test sets...")

    train_data, valid_data, test_data = data_preparation(
        config,
        dataset,
    )

    print("\nData preparation complete.")

    # ======================================================
    # Runtime Resource Check After Data Preparation
    # ======================================================

    print("\nRuntime resources after data preparation:")
    print("-" * 70)

    ram = psutil.virtual_memory()
    process_ram = (
        psutil.Process(os.getpid()).memory_info().rss
    )

    print(f"System RAM usage: {ram.percent:.1f}%")
    print(
        f"System RAM available: "
        f"{ram.available / 1024**3:.2f} GB"
    )
    print(
        f"Python process RAM: "
        f"{process_ram / 1024**3:.2f} GB"
    )

    print("-" * 70)

    # ======================================================
    # Reset Seed Before Model Initialisation
    # ======================================================

    init_seed(
        config["seed"] + config["local_rank"],
        config["reproducibility"],
    )

    # ======================================================
    # Create BERT4Rec Model
    # ======================================================

    print("\nCreating BERT4Rec model...")

    model = BERT4Rec(
        config,
        train_data.dataset,
    ).to(config["device"])

    print(model)

    # ======================================================
    # Model Information
    # ======================================================

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel Information:")
    print("-" * 70)
    print(f"Total parameters:      {total_parameters:,}")
    print(f"Trainable parameters:  {trainable_parameters:,}")
    print(
        "Model device:          "
        f"{next(model.parameters()).device}"
    )
    print("-" * 70)

    # ======================================================
    # Create Trainer
    # ======================================================

    trainer = Trainer(config, model)

    # ======================================================
    # Start Runtime Resource Monitor
    # ======================================================

    print("\nStarting runtime resource monitor...")
    print(
        "Resource measurements will be printed every "
        "30 seconds."
    )

    start_resource_monitor(interval=30)

    # ======================================================
    # Train Model
    # ======================================================

    print("\n" + "=" * 70)
    print("STARTING BERT4Rec TRAINING")
    print("=" * 70)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=True,
    )

    # ======================================================
    # Best Validation Results
    # ======================================================

    print("\n" + "=" * 70)
    print("BEST VALIDATION RESULT")
    print("=" * 70)

    print(f"Best validation score: {best_valid_score}")

    print("Validation results:")
    print(best_valid_result)

    # ======================================================
    # Final Test Evaluation
    # ======================================================

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print("\nLoading best checkpoint for final evaluation...")

    # ======================================================
    # PyTorch Checkpoint Loading Workaround
    # ======================================================

    original_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load

    try:
        test_result = trainer.evaluate(
            test_data,
            load_best_model=True,
            show_progress=True,
        )
    finally:
        torch.load = original_torch_load

    # ======================================================
    # Print Test Results
    # ======================================================

    print("\nTest results:")
    print(test_result)

    # ======================================================
    # Finished
    # ======================================================

    print("\n" + "=" * 70)
    print("BERT4Rec TRAINING COMPLETE")
    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    train_bert4rec()