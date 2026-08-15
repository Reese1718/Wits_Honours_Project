import os
import sys
import random
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import MaskItemSequence
from recbole.data.interaction import Interaction
from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


# ==========================================================
# Patch: guarantee at least 1 masked position per sequence
# ==========================================================
#
# Stock RecBole's MaskItemSequence masks each item independently
# according to mask_ratio.
#
# With short sequences, it is possible for ZERO items to be masked.
# BERT4Rec's CE loss then receives zero valid targets:
#
#     loss = sum(loss * targets) / sum(targets)
#
# If sum(targets) == 0:
#
#     loss = 0 / 0 = NaN
#
# This patch guarantees that every non-empty sequence has at least
# one masked position.
#
# IMPORTANT:
# This patch is applied BEFORE create_dataset() and
# data_preparation() so that the modified masking behaviour is used
# when the training dataset is constructed.
# ==========================================================


def _patched_mask_call(self, dataset, interaction):

    item_seq = interaction[self.ITEM_SEQ]

    device = item_seq.device
    batch_size = item_seq.size(0)

    n_items = dataset.num(self.ITEM_ID)

    sequence_instances = item_seq.cpu().numpy().tolist()

    masked_item_sequence = []
    pos_items = []
    neg_items = []
    masked_index = []

    # ------------------------------------------------------
    # Fine-tuning masking mode
    # ------------------------------------------------------

    if random.random() < self.ft_ratio:

        interaction = self._append_mask_last(
            interaction,
            n_items,
            device
        )

    # ------------------------------------------------------
    # Standard BERT4Rec masking
    # ------------------------------------------------------

    else:

        for instance in sequence_instances:

            masked_sequence = instance.copy()

            pos_item = []
            neg_item = []
            index_ids = []

            valid_len = 0

            # --------------------------------------------------
            # Randomly mask items
            # --------------------------------------------------

            for index_id, item in enumerate(instance):

                # Padding begins here
                if item == 0:
                    break

                valid_len += 1

                prob = random.random()

                if prob < self.mask_ratio:

                    pos_item.append(item)

                    neg_item.append(
                        self._neg_sample(
                            instance,
                            n_items
                        )
                    )

                    masked_sequence[index_id] = n_items

                    index_ids.append(index_id)

            # --------------------------------------------------
            # FIX:
            # Guarantee at least one masked position
            # --------------------------------------------------

            if len(index_ids) == 0 and valid_len > 0:

                force_idx = valid_len - 1

                pos_item.append(
                    instance[force_idx]
                )

                neg_item.append(
                    self._neg_sample(
                        instance,
                        n_items
                    )
                )

                masked_sequence[force_idx] = n_items

                index_ids.append(force_idx)

            # --------------------------------------------------
            # Store sequence information
            # --------------------------------------------------

            masked_item_sequence.append(
                masked_sequence
            )

            pos_items.append(
                self._padding_sequence(
                    pos_item,
                    self.mask_item_length
                )
            )

            neg_items.append(
                self._padding_sequence(
                    neg_item,
                    self.mask_item_length
                )
            )

            masked_index.append(
                self._padding_sequence(
                    index_ids,
                    self.mask_item_length
                )
            )

        # ------------------------------------------------------
        # Convert lists to tensors
        # ------------------------------------------------------

        masked_item_sequence = torch.tensor(
            masked_item_sequence,
            dtype=torch.long,
            device=device
        ).view(
            batch_size,
            -1
        )

        pos_items = torch.tensor(
            pos_items,
            dtype=torch.long,
            device=device
        ).view(
            batch_size,
            -1
        )

        neg_items = torch.tensor(
            neg_items,
            dtype=torch.long,
            device=device
        ).view(
            batch_size,
            -1
        )

        masked_index = torch.tensor(
            masked_index,
            dtype=torch.long,
            device=device
        ).view(
            batch_size,
            -1
        )

        # ------------------------------------------------------
        # Add the generated fields to the interaction
        # ------------------------------------------------------

        new_dict = {
            self.MASK_ITEM_SEQ: masked_item_sequence,
            self.POS_ITEMS: pos_items,
            self.NEG_ITEMS: neg_items,
            self.MASK_INDEX: masked_index,
        }

        interaction.update(
            Interaction(new_dict)
        )

    return interaction


# ==========================================================
# Apply masking patch
# ==========================================================
#
# This MUST happen before create_dataset()/data_preparation().
# ==========================================================

MaskItemSequence.__call__ = _patched_mask_call


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

    # ======================================================
    # Load RecBole configuration
    # ======================================================

    config = Config(
        model="BERT4Rec",
        config_file_list=[
            CONFIG_FILE
        ]
    )

    # ======================================================
    # Print important configuration
    # ======================================================

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

        print(
            f"{parameter}: {config[parameter]}"
        )

    print("-" * 70)

    # ======================================================
    # Set random seed
    # ======================================================

    init_seed(
        config["seed"],
        config["reproducibility"]
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
    # Create dataset
    # ======================================================

    print("\nCreating dataset...")

    dataset = create_dataset(
        config
    )

    print("\nDataset created successfully.")

    print(dataset)

    # ======================================================
    # Prepare train / validation / test sets
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
    # Create BERT4Rec model
    # ======================================================

    print(
        "\nCreating BERT4Rec model..."
    )

    model = BERT4Rec(
        config,
        train_data.dataset
    ).to(
        config["device"]
    )

    print(model)

    # ======================================================
    # Model information
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
    # Check initial model parameters
    # ======================================================

    print(
        "\nChecking initial model parameters..."
    )

    bad_parameters = False

    for name, param in model.named_parameters():

        if torch.isnan(param).any():

            print(
                f"NaN initial parameter: {name}"
            )

            bad_parameters = True

        elif torch.isinf(param).any():

            print(
                f"Inf initial parameter: {name}"
            )

            bad_parameters = True

    if not bad_parameters:

        print(
            "All initial model parameters are finite."
        )

    # ======================================================
    # Test one BERT4Rec training batch
    # ======================================================

    print(
        "\nTesting one training batch..."
    )

    model.train()

    batch = next(
        iter(train_data)
    )

    batch = batch.to(
        config["device"]
    )

    # ------------------------------------------------------
    # Check masked positions
    # ------------------------------------------------------

    masked_index = batch.interaction[
        model.MASK_INDEX
    ]

    num_masked = (
        masked_index > 0
    ).sum().item()

    print(
        "Masked positions in test batch:",
        num_masked
    )

    if num_masked == 0:

        raise RuntimeError(
            "ZERO MASKED POSITIONS FOUND. "
            "The MaskItemSequence patch is not working."
        )

    # ------------------------------------------------------
    # Calculate loss
    # ------------------------------------------------------

    loss = model.calculate_loss(
        batch.interaction
    )

    print(
        "Initial batch loss:",
        loss.item()
    )

    print(
        "Loss is NaN:",
        torch.isnan(loss).item()
    )

    print(
        "Loss is Inf:",
        torch.isinf(loss).item()
    )

    if torch.isnan(loss).item():

        raise RuntimeError(
            "Initial BERT4Rec loss is NaN."
        )

    if torch.isinf(loss).item():

        raise RuntimeError(
            "Initial BERT4Rec loss is Inf."
        )

    # ======================================================
    # Gradient check
    # ======================================================

    loss.backward()

    print(
        "\nGradient check:"
    )

    bad_gradients = False

    for name, param in model.named_parameters():

        if param.grad is not None:

            if torch.isnan(param.grad).any():

                print(
                    f"NaN gradient: {name}"
                )

                bad_gradients = True

            elif torch.isinf(param.grad).any():

                print(
                    f"Inf gradient: {name}"
                )

                bad_gradients = True

    if not bad_gradients:

        print(
            "All gradients are finite."
        )

    # ------------------------------------------------------
    # Stop immediately if gradients are invalid
    # ------------------------------------------------------

    if bad_gradients:

        raise RuntimeError(
            "Invalid gradients detected before training."
        )

    # Clear gradients before actual training

    model.zero_grad()

    # ======================================================
    # Create trainer
    # ======================================================

    trainer = Trainer(
        config,
        model
    )

    # ======================================================
    # Train model
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING BERT4Rec TRAINING"
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
    # Best validation results
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
    # Final test evaluation
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
    # Handle PyTorch checkpoint loading
    # ======================================================
    #
    # Newer PyTorch versions use weights_only=True by
    # default. RecBole's checkpoint requires the full
    # checkpoint object.
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

        # Restore original torch.load

        torch.load = original_torch_load

    # ======================================================
    # Print test results
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
        "BERT4Rec TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    train_bert4rec()