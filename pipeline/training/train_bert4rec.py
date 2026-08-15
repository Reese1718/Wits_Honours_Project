import os
import sys
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


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
    sys.argv[1]
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

    # ------------------------------------------------------
    # Load RecBole configuration
    # ------------------------------------------------------

    config = Config(
        model="BERT4Rec",
        config_file_list=[CONFIG_FILE]
    )

    # ------------------------------------------------------
    # Check initial model parameters
    # ------------------------------------------------------

    print("\nChecking initial model parameters...")

    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"BAD INITIAL PARAMETER: {name}")

    # ------------------------------------------------------
    # Print important configuration
    # ------------------------------------------------------

    print("\nBERT4Rec Configuration:")
    print("-" * 70)

    config_parameters = [
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
        "device",
    ]

    for parameter in config_parameters:
        print(f"{parameter}: {config[parameter]}")

    print("-" * 70)

    # ------------------------------------------------------
    # Set random seed
    # ------------------------------------------------------

    init_seed(
        config["seed"],
        config["reproducibility"]
    )

    # ------------------------------------------------------
    # Initialise logging
    # ------------------------------------------------------

    init_logger(config)

    # ------------------------------------------------------
    # Check hardware
    # ------------------------------------------------------

    print("\nHardware:")
    print("-" * 70)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB"
        )

    print("RecBole device:", config["device"])
    print("-" * 70)

    # ------------------------------------------------------
    # Create dataset
    # ------------------------------------------------------

    print("\nCreating dataset...")

    dataset = create_dataset(config)

    print("\nDataset created successfully.")
    print(dataset)

    # ------------------------------------------------------
    # Prepare train / validation / test sets
    # ------------------------------------------------------

    print("\nPreparing train/validation/test sets...")

    train_data, valid_data, test_data = data_preparation(
        config,
        dataset
    )

    print("\nData preparation complete.")

    # ------------------------------------------------------
    # Create BERT4Rec model
    # ------------------------------------------------------

    print("\nCreating BERT4Rec model...")

    model = BERT4Rec(
        config,
        train_data.dataset
    ).to(config["device"])

    print(model)

    # ------------------------------------------------------
    # Model information
    # ------------------------------------------------------

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
    print(f"Model device:          {next(model.parameters()).device}")
    print("-" * 70)


    # ------------------------------------------------------
    # Test one BERT4Rec training batch
    # ------------------------------------------------------

    print("\nTesting one training batch...")

    model.train()

    batch = next(iter(train_data))
    batch = batch.to(config["device"])

    loss = model.calculate_loss(batch.interaction)

    print("Initial batch loss:", loss.item())
    print("Loss is NaN:", torch.isnan(loss).item())
    print("Loss is Inf:", torch.isinf(loss).item())

    loss.backward()

    print("\nGradient check:")

    bad_gradients = False

    for name, param in model.named_parameters():

        if param.grad is not None:

            if torch.isnan(param.grad).any():
                print(f"NaN gradient: {name}")
                bad_gradients = True

            elif torch.isinf(param.grad).any():
                print(f"Inf gradient: {name}")
                bad_gradients = True

    if not bad_gradients:
        print("All gradients are finite.")

    model.zero_grad()

    # ------------------------------------------------------
    # Create trainer
    # ------------------------------------------------------

    trainer = Trainer(
        config,
        model
    )

    # ------------------------------------------------------
    # Train model
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING BERT4Rec TRAINING")
    print("=" * 70)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=True
    )

    # ------------------------------------------------------
    # Best validation results
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("BEST VALIDATION RESULT")
    print("=" * 70)

    print(f"Best validation score: {best_valid_score}")

    print("Validation results:")
    print(best_valid_result)

    # ------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print("\nLoading best checkpoint for final evaluation...")

    # RecBole's saved checkpoint needs weights_only=False
    # with newer PyTorch versions.

    original_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load

    try:

        test_result = trainer.evaluate(
            test_data,
            load_best_model=True,
            show_progress=True
        )

    finally:

        # Restore original torch.load
        torch.load = original_torch_load

    print("\nTest results:")
    print(test_result)

    # ------------------------------------------------------
    # Finished
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("BERT4Rec TRAINING COMPLETE")
    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    train_bert4rec()