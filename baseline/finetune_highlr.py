import os
import torch
import logging
import numpy as np
import random
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

DRY_RUN = False

def safe_load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    clean_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    print("   Weights loaded successfully.")

def bias_dataset_towards_new_data(dataset, old_year="2023", keep_ratio=0.05):
    # NOTE: Reduced keep_ratio to 5% to focus even HARDER on new data
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    yaml_path = os.path.join(script_dir, "rnn_args.yaml")
    args = OmegaConf.load(yaml_path)
    args["dataset"]["dataset_dir"] = os.path.join(project_root, "data", "hdf5_data_final")
    
    # --- AGGRESSIVE SETTINGS ---
    args["lr_max"] = 0.0001       # 1e-4 (10x higher than Stage 2)
    args["lr_min"] = 0.00001      # 1e-5
    args["lr_max_day"] = 0.0001   
    args["lr_min_day"] = 0.00001
    
    # Reduce Augmentation (Make it easier to learn the signal)
    args["dataset"]["data_transforms"]["white_noise_std"] = 0.2     # Reduced from 1.0
    args["dataset"]["data_transforms"]["constant_offset_std"] = 0.05 # Reduced from 0.2
    # ---------------------------

    if DRY_RUN:
        # (Keep Dry Run logic same as before if needed)
        pass
    else:
        print("\n*** RUNNING IN STAGE 3 AGGRESSIVE MODE ***")
        finetune_path = os.path.join(script_dir, "trained_models", "finetuned_rnn_aggressive")
        args["output_dir"] = finetune_path
        args["checkpoint_dir"] = os.path.join(finetune_path, "checkpoint")
        args["num_training_batches"] = 8000

    args["init_from_checkpoint"] = False 
    args["save_best_checkpoint"] = True

    print("Initializing Trainer...")
    trainer = BrainToTextDecoder_Trainer(args)
    
    # Load from STAGE 1 (Aligned) checkpoint, not Stage 2 (Stuck)
    # We want to retry the unfreezing with better parameters
    checkpoint_path = os.path.join(script_dir, "trained_models", "finetuned_rnn", "checkpoint", "best_checkpoint")
    
    if not os.path.exists(checkpoint_path):
         # Fallback to Stage 2 if Stage 1 deleted
         checkpoint_path = os.path.join(script_dir, "trained_models", "finetuned_rnn_unfrozen", "checkpoint", "best_checkpoint")
    
    safe_load_checkpoint(trainer.model, checkpoint_path)
    
    # UNFREEZE EVERYTHING
    print("INFO: All layers are UNFROZEN.")
    
    print("Filtering dataset for available files...")
    filter_empty_days(trainer.train_dataset)
    filter_empty_days(trainer.val_dataset)

    if not DRY_RUN:
        bias_dataset_towards_new_data(trainer.train_dataset, old_year="2023", keep_ratio=0.05)
    
    print("Starting Training Loop...")
    trainer.train()

if __name__ == "__main__":
    main()