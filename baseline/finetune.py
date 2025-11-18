import os
import torch
import logging
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

def safe_load_checkpoint(model, checkpoint_path):
    """
    Loads weights from a checkpoint into the model, skipping layers 
    with shape mismatches (like the Day Layers for new days).
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # Filter out weights that don't match shape
    filtered_dict = {}
    ignored_keys = []
    
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                ignored_keys.append(k)
        else:
            # Handle module. prefix issues if trained with DataParallel
            k_fixed = k.replace("module.", "")
            if k_fixed in model_dict and v.shape == model_dict[k_fixed].shape:
                filtered_dict[k_fixed] = v
    
    if len(ignored_keys) > 0:
        print(f"   [WARNING] Skipped {len(ignored_keys)} layers due to shape mismatch (Expected for new days).")
        print(f"   First few skipped: {ignored_keys[:5]}")
    
    # Overwrite entries in the existing state dict
    model_dict.update(filtered_dict)
    
    # Load the new state dict
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
            
    print(f"   Freezing Complete: {active_count} parameters trainable (Day Layers), {frozen_count} frozen (GRU/Body).")

def main():
    # 1. Load Configuration
    # Ensure you have updated rnn_args.yaml to include the NEW sessions in the 'sessions' list
    args = OmegaConf.load("rnn_args.yaml")
    
    DRY_RUN = True 
    # -------------------------------------

    if DRY_RUN:
        print("RUNNING IN DRY RUN MODE (Safe for Local)")
        args["gpu_number"] = "-1"              # Force CPU (prevents VRAM crash)
        args["num_training_batches"] = 5       # Run only 5 steps
        args["dataset"]["batch_size"] = 2      # Tiny batch size
        args["dataset"]["num_dataloader_workers"] = 0 # Disable multiprocessing
        
        # Save to a trash folder so we don't mess up real results
        args["output_dir"] = "dry_run_output"
        args["checkpoint_dir"] = "dry_run_output/checkpoint"
    else:
        # REAL RUN SETTINGS (For Longleaf)
        # Save to a NEW folder to keep fine-tuned model separate
        args["output_dir"] = "trained_models/finetuned_rnn"
        args["checkpoint_dir"] = "trained_models/finetuned_rnn/checkpoint"



    # 2. Force specific args for Fine-Tuning
    args["init_from_checkpoint"] = False  # We load manually to handle mismatches
    args["save_best_checkpoint"] = True
    
    # Adjust training length for fine-tuning (optional, 120k might be too long for just day alignment)
    # args["num_training_batches"] = 10000 
    
    print("Initializing Trainer...")
    trainer = BrainToTextDecoder_Trainer(args)
    
    # 3. Safe Load Weights
    # Point this to your best_checkpoint file
    checkpoint_path = "checkpoint/best_checkpoint" 
    if not os.path.exists(checkpoint_path):
         raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}. Please check path.")
         
    safe_load_checkpoint(trainer.model, checkpoint_path)
    
    # 4. Freeze Layers
    freeze_layers(trainer.model)
    
    # 5. Filter Missing Data (Critical for Fine-Tuning on partial datasets)
    # This prevents the trainer from crashing if you didn't upload the OLD 2023 data files
    print("Filtering dataset for available files...")
    
    def filter_empty_days(dataset):
        valid_days = {d: info for d, info in dataset.trial_indicies.items() if len(info['trials']) > 0}
        dropped_days = len(dataset.trial_indicies) - len(valid_days)
        if dropped_days > 0:
            print(f"   Dropped {dropped_days} sessions that had no data found on disk.")
        dataset.trial_indicies = valid_days
        dataset.batch_index = dataset.create_batch_index_train() if dataset.split == 'train' else dataset.create_batch_index_test()
        dataset.n_batches = len(dataset.batch_index)

    filter_empty_days(trainer.train_dataset)
    filter_empty_days(trainer.val_dataset)
    
    # 6. Start Training
    print("Starting Fine-Tuning...")
    trainer.train()

if __name__ == "__main__":
    main()