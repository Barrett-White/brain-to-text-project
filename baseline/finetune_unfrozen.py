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
    Standard loading for Stage 2. Since we are loading from a model 
    that already has the correct shapes (from Stage 1), we don't need 
    complex filtering logic anymore.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load weights directly (strict=False handles minor metadata differences if any)
    # We still clean keys just in case
    state_dict = checkpoint['model_state_dict']
    clean_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict, strict=False)
    print("   Weights loaded successfully.")

def bias_dataset_towards_new_data(dataset, old_year="2023", keep_ratio=0.1):
    """
    Keeps biasing towards new data to force the GRU to adapt to 2025 signals.
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
            
    num_old_to_keep = max(1, int(len(old_days) * keep_ratio))
    kept_old_days = random.sample(old_days, num_old_to_keep)
    
    final_days = new_days + kept_old_days
    new_trial_indices = {d: dataset.trial_indicies[d] for d in final_days}
    dataset.trial_indicies = new_trial_indices
    
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
    
    # --- STAGE 2 SETTINGS: UNFREEZE & LOW LR ---
    args["lr_max"] = 0.00005      # 1e-5 (Very low to protect knowledge)
    args["lr_min"] = 0.000005     # 1e-6
    args["lr_max_day"] = 0.00005  # Keep input layer aligned
    args["lr_min_day"] = 0.000005
    # -------------------------------------------

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
        print("\n*** RUNNING IN STAGE 2 FINE-TUNE MODE (Unfrozen) ***")
        # Save to a NEW folder 'finetuned_rnn_unfrozen'
        finetune_path = os.path.join(script_dir, "trained_models", "finetuned_rnn_unfrozen")
        args["output_dir"] = finetune_path
        args["checkpoint_dir"] = os.path.join(finetune_path, "checkpoint")
        args["num_training_batches"] = 10000

    args["init_from_checkpoint"] = False 
    args["save_best_checkpoint"] = True

    print("Initializing Trainer...")
    trainer = BrainToTextDecoder_Trainer(args)
    
    # --- LOAD FROM STAGE 1 CHECKPOINT ---
    # We load the model we just finished training (finetuned_rnn)
    checkpoint_path = os.path.join(script_dir, "trained_models", "finetuned_rnn", "checkpoint", "best_checkpoint")
    if not os.path.exists(checkpoint_path):
         raise FileNotFoundError(f"Could not find Stage 1 checkpoint at {checkpoint_path}")
         
    safe_load_checkpoint(trainer.model, checkpoint_path)

    # freeze_layers(trainer.model) 
    print("INFO: All layers are UNFROZEN for Stage 2 fine-tuning.")
    # ---------------------------------------------
    
    print("Filtering dataset for available files...")
    filter_empty_days(trainer.train_dataset)
    filter_empty_days(trainer.val_dataset)

    # Apply Biased Sampling (Focus on 2025)
    if not DRY_RUN:
        bias_dataset_towards_new_data(trainer.train_dataset, old_year="2023", keep_ratio=0.10)

    if DRY_RUN and len(trainer.val_dataset.batch_index) > 0:
        print("[DRY RUN] Truncating validation set to 1 batch...")
        first_key = list(trainer.val_dataset.batch_index.keys())[0]
        trainer.val_dataset.batch_index = {0: trainer.val_dataset.batch_index[first_key]}
        trainer.val_dataset.n_batches = 1
    
    print("Starting Training Loop...")
    trainer.train()

if __name__ == "__main__":
    main()