import os
import sys
import torch

import numpy as np

# RecBole compatibility with NumPy 2.x
if not hasattr(np, "float_"):
    np.float_ = np.float64


# NumPy 2 compatibility for RecBole
import numpy as np
np.float_ = np.float64
np.complex_ = np.complex128
np.unicode_ = np.str_

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.gru4rec import GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

import numpy as np

# Compatibility for RecBole with NumPy 2
for name, replacement in {
    "float_": np.float64,
    "complex_": np.complex128,
    "unicode_": np.str_,
}.items():
    if not hasattr(np, name):
        setattr(np, name, replacement)


# ==========================================================
# Project Configuration
# ==========================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PIPELINE_DIR = os.path.dirname(
    SCRIPT_DIR
)

if len(sys.argv) != 2:
    raise ValueError(
        "Usage: python train_gru4rec.py <config_file>"
    )

CONFIG_FILE = os.path.join(
    PIPELINE_DIR,
    "configs",
    sys.argv[1]
)


# ==========================================================
# Main Training Function
# ==========================================================

def train_gru4rec():

    print("=" * 70)
    print("GRU4Rec Training")
    print("=" * 70)

    print("\nUsing configuration:")
    print(CONFIG_FILE)


    # ======================================================
    # Load RecBole Configuration
    # ======================================================

    config = Config(
        model="GRU4Rec",
        config_file_list=[
            CONFIG_FILE
        ]
    )


    # ======================================================
    # Print Important Configuration
    # ======================================================

    print("\nGRU4Rec Configuration:")
    print("-" * 70)

    config_parameters = [
        "hidden_size",
        "num_layers",
        "dropout_prob",
        "loss_type",
        "learning_rate",
        "train_batch_size",
        "epochs",
        "device",
    ]

    for parameter in config_parameters:

        try:
            print(
                f"{parameter}: {config[parameter]}"
            )
        except KeyError:
            pass

    print("-" * 70)


    # ======================================================
    # Set Random Seed
    # ======================================================

    init_seed(
        config["seed"],
        config["reproducibility"]
    )


    # ======================================================
    # Initialise Logging
    # ======================================================

    init_logger(config)


    # ======================================================
    # Hardware
    # ======================================================

    print("\nHardware:")
    print("-" * 70)

    print(
        "PyTorch version:",
        torch.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        gpu_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / 1024**3
        )

        print(
            "GPU memory:",
            round(gpu_memory, 2),
            "GB"
        )

    print(
        "RecBole device:",
        config["device"]
    )

    print("-" * 70)


    # ======================================================
    # Create Dataset
    # ======================================================

    print("\nCreating dataset...")

    dataset = create_dataset(
        config
    )

    print("\nDataset created successfully.")

    print(dataset)


    # ======================================================
    # Prepare Train / Validation / Test Sets
    # ======================================================

    print(
        "\nPreparing train/validation/test sets..."
    )

    train_data, valid_data, test_data = data_preparation(
        config,
        dataset
    )

    print(
        "\nData preparation complete."
    )


    # ======================================================
    # Create GRU4Rec Model
    # ======================================================

    print(
        "\nCreating GRU4Rec model..."
    )

    model = GRU4Rec(
        config,
        train_data.dataset
    ).to(
        config["device"]
    )

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

    print(
        f"Total parameters:      {total_parameters:,}"
    )

    print(
        f"Trainable parameters:  {trainable_parameters:,}"
    )

    print(
        f"Model device:          "
        f"{next(model.parameters()).device}"
    )

    print("-" * 70)


    # ======================================================
    # Create Trainer
    # ======================================================

    trainer = Trainer(
        config,
        model
    )


    # ======================================================
    # Train Model
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING GRU4Rec TRAINING"
    )

    print(
        "=" * 70
    )

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=True
    )


    # ======================================================
    # Best Validation Results
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST VALIDATION RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Best validation score: {best_valid_score}"
    )

    print(
        "Validation results:"
    )

    print(
        best_valid_result
    )


    # ======================================================
    # Final Test Evaluation
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nLoading best checkpoint for final evaluation..."
    )


    # ======================================================
    # PyTorch Checkpoint Loading Workaround
    # ======================================================

    original_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):

        kwargs["weights_only"] = False

        return original_torch_load(
            *args,
            **kwargs
        )

    torch.load = trusted_torch_load

    try:

        test_result = trainer.evaluate(
            test_data,
            load_best_model=True,
            show_progress=True
        )

    finally:

        torch.load = original_torch_load


    # ======================================================
    # Print Test Results
    # ======================================================

    print(
        "\nTest results:"
    )

    print(
        test_result
    )


    # ======================================================
    # Finished
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "GRU4Rec TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    train_gru4rec()