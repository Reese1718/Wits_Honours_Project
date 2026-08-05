import os
import torch
from recbole.quick_start import run_recbole

# --- PyTorch 2.6+ Security Hotfix for RecBole ---
# Prevents crash during final evaluation model loading
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ------------------------------------------------

# 1. Dynamic Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASETS_ROOT = os.path.join(BASE_DIR, "datasets")

# 2. RecBole Configuration Dictionary
config_dict = {
    # Core Setup
    'model': 'SASRec',
    'dataset': 'Fashion', 
    'data_path': DATASETS_ROOT,
    
    'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp',
    'MAX_ITEM_LIST_LENGTH': 50,
    
    # Evaluation Setup: Leave-One-Out 
    'eval_args': {
        'split': {'LS': 'valid_and_test'},  
        'group_by': 'user',
        'order': 'TO',                      
        'mode': 'full'                      
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
    'loss_type': 'CE',                      
    
    # THE FIX: Disable negative sampling for CE loss
    'train_neg_sample_args': None,
    
    # Training Loop Constraints (SET FOR GPU CLUSTER/COLAB)
    'epochs': 100,                          
    'train_batch_size': 2048,               
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