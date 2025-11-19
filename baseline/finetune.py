import os
import torch
import logging
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

# Configuration Toggle
DRY_RUN = False  # Set to True for local testing, False for Longleaf

def safe_load_checkpoint(model, checkpoint_path):
    """
    Loads weights from a checkpoint into the model, skipping layers 
    with shape mismatches and cleaning up key names.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Load to CPU first to avoid OOM errors
    # Added weights_only=False to support older checkpoints
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # Clean keys and Filter out weights that do not match shape
    cleaned_dict = {}
    for k, v in pretrained_dict.items():
        # Remove prefixes from torch.compile or DataParallel
        new_key = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned_dict[new_key] = v

    filtered_dict = {}
    ignored_keys = []
    
    for k, v in cleaned_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                ignored_keys.append(k)
        else:
            pass
    
    if len(ignored_keys) > 0:
        print(f"   [WARNING] Skipped {len(ignored_keys)} layers due to shape mismatch (Expected for Day Layers).")
    
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
            
    print(f"   Freezing Complete: {active_count} parameters trainable (Day Layers), {frozen_count} frozen.")

def filter_empty_days(dataset):
    """
    Removes days from the dataset that do not have corresponding HDF5 files on disk.
    """
    valid_days = {}
    for d, info in dataset.trial_indicies.items():
        if os.path.exists(info['session_path']):
             valid_days[d] = info
        else:
             pass

    dropped_count = len(dataset.trial_indicies) - len(valid_days)
    if dropped_count > 0:
        print(f"   [INFO] Dropped {dropped_count} sessions not found on disk.")
    
    dataset.trial_indicies = valid_days
    if dataset.split == 'train':
        dataset.batch_index = dataset.create_batch_index_train()
    else:
        dataset.batch_index = dataset.create_batch_index_test()
    dataset.n_batches = len(dataset.batch_index)

def main():
    # Setup Paths relative to this script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    yaml_path = os.path.join(script_dir, "rnn_args.yaml")
    print(f"Loading config from: {yaml_path}")
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found at {yaml_path}")

    args = OmegaConf.load(yaml_path)

    # Fix data path to be absolute based on project root
    args["dataset"]["dataset_dir"] = os.path.join(project_root, "data", "hdf5_data_final")
    print(f"Data directory set to: {args['dataset']['dataset_dir']}")
    
    # Configure Output Directories inside baseline/trained_models
    if DRY_RUN:
        print("\n*** RUNNING IN DRY RUN MODE (Safe for Local) ***")
        args["gpu_number"] = "-1"              # Force CPU
        args["num_training_batches"] = 1       # Run only 1 step (Minimum needed to test loop)
        args["dataset"]["batch_size"] = 2      # Tiny batch size
        args["dataset"]["num_dataloader_workers"] = 0 
        args["batches_per_val_step"] = 1000    # Prevent intermediate validation
        
        args["log_individual_day_val_PER"] = False # Stop dividing by 0 when no trial  because we forced the validation to run only 1 batch
        # Output path: baseline/trained_models/dry_run_output
        dry_run_path = os.path.join(script_dir, "trained_models", "dry_run_output")
        args["output_dir"] = dry_run_path
        args["checkpoint_dir"] = os.path.join(dry_run_path, "checkpoint")
        
    else:
        print("\n*** RUNNING IN FINE-TUNE MODE (For Longleaf) ***")
        # Output path: baseline/trained_models/finetuned_rnn
        finetune_path = os.path.join(script_dir, "trained_models", "finetuned_rnn")
        args["output_dir"] = finetune_path
        args["checkpoint_dir"] = os.path.join(finetune_path, "checkpoint")
        
        args["num_training_batches"] = 10000 

    # Force specific args for Fine-Tuning logic
    args["init_from_checkpoint"] = False 
    args["save_best_checkpoint"] = True

    print("Initializing Trainer...")
    trainer = BrainToTextDecoder_Trainer(args)
    
    # Safe Load Weights
    # Path: baseline/trained_models/baseline_rnn/checkpoint/best_checkpoint
    checkpoint_path = os.path.join(script_dir, "trained_models", "baseline_rnn", "checkpoint", "best_checkpoint")
    
    if not os.path.exists(checkpoint_path):
         raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}. Please check path.")
         
    safe_load_checkpoint(trainer.model, checkpoint_path)
    
    # Freeze Layers
    freeze_layers(trainer.model)
    
    # Filter Missing Data 
    print("Filtering dataset for available files...")
    filter_empty_days(trainer.train_dataset)
    filter_empty_days(trainer.val_dataset)

    # --- SPEED OPTIMIZATION FOR DRY RUN ---
    if DRY_RUN and len(trainer.val_dataset.batch_index) > 0:
        print("[DRY RUN] Truncating validation set to 1 batch to save time...")
        # Force validation dataset to only have 1 batch
        first_batch_key = list(trainer.val_dataset.batch_index.keys())[0]
        trainer.val_dataset.batch_index = {0: trainer.val_dataset.batch_index[first_batch_key]}
        trainer.val_dataset.n_batches = 1
    # --------------------------------------
    
    # Start Training
    print("Starting Training Loop...")
    trainer.train()

if __name__ == "__main__":
    main()