import os
import sys

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
    # Create Dataset
    # ======================================================

    print("\nCreating dataset...")

    dataset = create_dataset(config)

    print("\nDataset created successfully.")
    print(dataset)

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
    # Reset Seed Before Model Initialisation
    # ======================================================

    # Dataset preparation can consume different amounts of randomness for
    # datasets of different sizes. Resetting here follows RecBole's standard
    # workflow and gives paired runs using the same seed a controlled model
    # initialisation and training RNG state.
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
