import os
import torch
import logging
import numpy as np
import random
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

# Configuration Toggle
DRY_RUN = False # Set to False for Longleaf

def safe_load_checkpoint(model, checkpoint_path):
    """
    Loads weights, cleans keys, and performs SMART INITIALIZATION (Mean Weight Transfer).
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    cleaned_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned_dict[new_key] = v

    filtered_dict = {}
    loaded_day_weights = []
    loaded_day_biases = []
    
    for k, v in cleaned_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
            if "day_weights" in k:
                loaded_day_weights.append(v)
            if "day_biases" in k:
                loaded_day_biases.append(v)
        else:
            pass

    # Smart Initialization: Use average of old days for new days
    if loaded_day_weights:
        print(f"   [Smart Init] Found {len(loaded_day_weights)} trained day layers.")
        mean_day_weight = torch.stack(loaded_day_weights).mean(dim=0)
        mean_day_bias = torch.stack(loaded_day_biases).mean(dim=0)
        
        initialized_count = 0
        for i in range(model.n_days):
            weight_key = f"day_weights.{i}"
            bias_key = f"day_biases.{i}"
            
            if weight_key not in filtered_dict:
                filtered_dict[weight_key] = mean_day_weight.clone()
                filtered_dict[bias_key] = mean_day_bias.clone()
                initialized_count += 1
        
        print(f"   [Smart Init] Applied average weights to {initialized_count} new sessions.")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print("   Weights loaded successfully.")

def freeze_layers(model):
    """
    Freezes GRU and Output layers, keeping only Day Layers trainable.
    """
    print("Freezing GRU and Output layers...")
    frozen_count = 0
    active_count = 0
    
    for name, param in model.named_parameters():
        if "day_" in name:
            param.requires_grad = True
            active_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
            
    print(f"   Freezing Complete: {active_count} parameters trainable (Day Layers), {frozen_count} frozen.")

def bias_dataset_towards_new_data(dataset, old_year="2023", keep_ratio=0.1):
    """
    Removes most 'Old' days from the training set to force the model to focus on 'New' data.
    
    Args:
        old_year: Sessions containing this string will be pruned.
        keep_ratio: Fraction of old days to keep (e.g., 0.1 means keep 10% of 2023 days).
    """
    print(f"Biasing dataset: Keeping 100% of New Data, but only {keep_ratio*100}% of '{old_year}' data...")
    
    all_days = list(dataset.trial_indicies.keys())
    new_days = []
    old_days = []
    
    for d in all_days:
        session_name = dataset.trial_indicies[d]['session_path']
        if old_year in session_name:
            old_days.append(d)
        else:
            new_days.append(d)
            
    # Randomly select a small subset of old days to prevent catastrophic forgetting
    num_old_to_keep = max(1, int(len(old_days) * keep_ratio))
    kept_old_days = random.sample(old_days, num_old_to_keep)
    
    final_days = new_days + kept_old_days
    
    # Rebuild the index dictionary
    new_trial_indices = {d: dataset.trial_indicies[d] for d in final_days}
    dataset.trial_indicies = new_trial_indices
    
    # Regenerate the batch schedule
    if dataset.split == 'train':
        dataset.batch_index = dataset.create_batch_index_train()
    else:
        dataset.batch_index = dataset.create_batch_index_test()
    dataset.n_batches = len(dataset.batch_index)
    
    print(f"   [Sampling] Pruned dataset. Training on {len(new_days)} New Days and {len(kept_old_days)} Old Days.")

def filter_empty_days(dataset):
    valid_days = {d: info for d, info in dataset.trial_indicies.items() if os.path.exists(info['session_path'])}
    dataset.trial_indicies = valid_days
    dataset.batch_index = dataset.create_batch_index_train() if dataset.split == 'train' else dataset.create_batch_index_test()
    dataset.n_batches = len(dataset.batch_index)

def main():
    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    yaml_path = os.path.join(script_dir, "rnn_args.yaml")
    args = OmegaConf.load(yaml_path)
    args["dataset"]["dataset_dir"] = os.path.join(project_root, "data", "hdf5_data_final")
    
    if DRY_RUN:
        print("\n*** RUNNING IN DRY RUN MODE (Safe for Local) ***")
        args["gpu_number"] = "-1"
        args["num_training_batches"] = 1
        args["dataset"]["batch_size"] = 2
        args["dataset"]["num_dataloader_workers"] = 0 
        args["batches_per_val_step"] = 1000
        args["log_individual_day_val_PER"] = False 
        dry_run_path = os.path.join(script_dir, "trained_models", "dry_run_output")
        args["output_dir"] = dry_run_path
        args["checkpoint_dir"] = os.path.join(dry_run_path, "checkpoint")
    else:
        print("\n*** RUNNING IN FINE-TUNE MODE (For Longleaf) ***")
        finetune_path = os.path.join(script_dir, "trained_models", "finetuned_rnn")
        args["output_dir"] = finetune_path
        args["checkpoint_dir"] = os.path.join(finetune_path, "checkpoint")
        args["num_training_batches"] = 12000

    args["init_from_checkpoint"] = False 
    args["save_best_checkpoint"] = True

    print("Initializing Trainer...")
    trainer = BrainToTextDecoder_Trainer(args)
    
    checkpoint_path = os.path.join(script_dir, "trained_models", "baseline_rnn", "checkpoint", "best_checkpoint")
    if not os.path.exists(checkpoint_path):
         raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")
         
    safe_load_checkpoint(trainer.model, checkpoint_path)
    freeze_layers(trainer.model)
    
    print("Filtering dataset for available files...")
    filter_empty_days(trainer.train_dataset)
    filter_empty_days(trainer.val_dataset)

    # --- APPLY BIASED SAMPLING HERE ---
    # This forces the trainer to see mostly 2025/2024 data
    if not DRY_RUN:
        bias_dataset_towards_new_data(trainer.train_dataset, old_year="2023", keep_ratio=0.10)
    # ----------------------------------

    if DRY_RUN and len(trainer.val_dataset.batch_index) > 0:
        print("[DRY RUN] Truncating validation set to 1 batch...")
        first_key = list(trainer.val_dataset.batch_index.keys())[0]
        trainer.val_dataset.batch_index = {0: trainer.val_dataset.batch_index[first_key]}
        trainer.val_dataset.n_batches = 1
    
    print("Starting Training Loop...")
    trainer.train()

if __name__ == "__main__":
    main()