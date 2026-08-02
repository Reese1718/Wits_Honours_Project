import os
from recbole.quick_start import run_recbole

# 1. Dynamic Path Resolution for Local VS Code
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASETS_ROOT = os.path.join(BASE_DIR, "datasets")

# 2. RecBole Configuration Dictionary
config_dict = {
    # Core Setup
    'model': 'SASRec',
    'dataset': 'Fashion',
    'data_path': DATASETS_ROOT,
    
    # Data Loading (Matching your .inter file columns)
    'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp',
    'MAX_ITEM_LIST_LENGTH': 50,
    
    # Evaluation Setup: Leave-One-Out (Chronological) against all items
    'eval_args': {
        'split': {'LS': 'valid_and_test'},  # Leave-One-Out splitting
        'group_by': 'user',
        'order': 'TO',                      # Temporal Order (chronological)
        'mode': 'full'                      # Evaluate against all unseen items
    },
    'metrics': ['Hit', 'NDCG'],
    'topk': [10],
    'valid_metric': 'NDCG@10',
    'metric_decimal_place': 4,
    
    # Model Hyperparameters (Standard SASRec Baselines)
    'embedding_size': 64,
    'num_layers': 2,
    'n_heads': 2,
    'dropout_prob': 0.2,
    'loss_type': 'CE',                      # Cross-Entropy Loss
    
    # THE FIX: Disable negative sampling for CE loss
    'train_neg_sample_args': None,
    
    # Training Loop Constraints (SET FOR LOCAL LAPTOP TESTING)
    'epochs': 1,                            # Reduced for a quick local sanity check
    'train_batch_size': 512,                # Smaller batch size for CPU testing
    'learning_rate': 0.001,
}

if __name__ == '__main__':
    print(f"Starting SASRec Baseline Evaluation on {config_dict['dataset']}...")
    
    # Execute RecBole pipeline
    run_recbole(
        model=config_dict['model'],
        dataset=config_dict['dataset'],
        config_dict=config_dict
    )