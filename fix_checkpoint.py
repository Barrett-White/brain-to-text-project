import sys
import torch
import os
import numpy as np

# 1. Apply the Patch (One last time) to allow loading the "infected" file
try:
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
except AttributeError:
    pass

# 2. Paths
# INPUT: The broken finetuned model
input_path = "baseline/trained_models/finetuned_rnn_unfrozen/checkpoint/best_checkpoint"
# OUTPUT: The new clean model
output_path = "baseline/trained_models/finetuned_rnn_unfrozen/checkpoint/best_checkpoint_clean.pt"

print(f"Loading {input_path}...")

# 3. Load the checkpoint
# The patch above allows this to succeed
if torch.cuda.is_available():
    checkpoint = torch.load(input_path)
else:
    checkpoint = torch.load(input_path, map_location=torch.device('cpu'))

print("Checkpoint loaded successfully into memory.")

# 4. Re-Save the checkpoint
# Since we are running this in your Numpy 1.24 environment, 
# torch.save will write it using Numpy 1.x format automatically.
print(f"Saving cleaned version to {output_path}...")
torch.save(checkpoint, output_path)

print("Done! The new file is 'Numpy 1.x' compatible.")