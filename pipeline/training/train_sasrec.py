import os

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


# ==========================================================
# Project Configuration
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)

CONFIG_FILE = os.path.join(
    PIPELINE_DIR,
    "configs",
    "sasrec.yaml"
)


# ==========================================================
# Main Training Function
# ==========================================================

def train_sasrec():

    print("=" * 70)
    print("SASRec Training")
    print("=" * 70)

    print(f"\nUsing configuration:")
    print(CONFIG_FILE)

    # ------------------------------------------------------
    # Load RecBole configuration
    # ------------------------------------------------------

    config = Config(
        model="SASRec",
        config_file_list=[CONFIG_FILE]
    )

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
    # Create SASRec model
    # ------------------------------------------------------

    print("\nCreating SASRec model...")

    model = SASRec(
        config,
        train_data.dataset
    ).to(config["device"])

    print(model)

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
    print("STARTING SASRec TRAINING")
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

    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=True
    )

    print("\nTest results:")
    print(test_result)

    # ------------------------------------------------------
    # Finished
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("SASRec TRAINING COMPLETE")
    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    train_sasrec()
