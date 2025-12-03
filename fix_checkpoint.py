import sys
import torch
import os
import numpy as np
import shutil

# 1. Apply the Patch to open the "infected" file
try:
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
except AttributeError:
    pass

# 2. Paths
base_dir = "baseline/trained_models/finetuned_rnn_unfrozen/checkpoint"
original_file = os.path.join(base_dir, "best_checkpoint")
backup_file = os.path.join(base_dir, "best_checkpoint_backup_numpy2")
temp_file = os.path.join(base_dir, "temp_fixed_checkpoint")

print(f"Target File: {original_file}")

# 3. Load the model (The patch makes this work)
print("Loading infected checkpoint...")
if torch.cuda.is_available():
    checkpoint = torch.load(original_file)
else:
    checkpoint = torch.load(original_file, map_location=torch.device('cpu'))

print("Loaded successfully.")

# 4. Backup the original (Safety First)
if not os.path.exists(backup_file):
    print("Creating backup...")
    shutil.copy(original_file, backup_file)

# 5. Save the Clean Version
# Since we are running this in your Numpy 1.24 environment, 
# Torch automatically saves it in the clean Numpy 1.x format.
print("Saving clean version...")
torch.save(checkpoint, temp_file)

# 6. Swap the files
# We overwrite 'best_checkpoint' with the new clean version.
os.rename(temp_file, original_file)

print(f"Done! 'best_checkpoint' is now fixed and ready to use.")