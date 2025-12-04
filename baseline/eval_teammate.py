import numpy as np
import argparse
import lm_decoder
import os
from tqdm import tqdm
from evaluate_model_helpers import rearrange_speech_logits_pt

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--npz_file', type=str, required=True, help="Path to the NPZ file containing logits")
parser.add_argument('--lm_path', type=str, required=True, help="Path to the language model directory")
parser.add_argument('--lm_beta', type=float, default=90.0, help="Word insertion penalty")
parser.add_argument('--acoustic_scale', type=float, default=0.325, help="Acoustic scale factor")
parser.add_argument('--logits_key', type=str, default='logits', help="Key name inside the NPZ file (default: logits)")
args = parser.parse_args()

# SETUP DECODER
print("Initializing Decoder...")
# Note: We use standard decoding options. 
# 7000=max_active, 200=min_active, 17.0=beam, 8.0=lattice_beam, 1.0=ctc_blank_skip, 0.0=length_penalty, 1=nbest
decode_opts = lm_decoder.DecodeOptions(7000, 200, 17.0, 8.0, args.acoustic_scale, 1.0, 0.0, 1)
decode_resource = lm_decoder.DecodeResource(
    os.path.join(args.lm_path, 'TLG.fst'), "", "", 
    os.path.join(args.lm_path, 'words.txt'), ""
)
decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

# LOAD DATA
print(f"Loading {args.npz_file}...")
try:
    data = np.load(args.npz_file)
    # Auto-detect key if 'logits' isn't found
    if args.logits_key not in data:
        print(f"Key '{args.logits_key}' not found. Available keys: {list(data.keys())}")
        # Fallback to the first key found
        first_key = list(data.keys())[0]
        print(f"Defaulting to key: '{first_key}'")
        logits_list = data[first_key]
    else:
        logits_list = data[args.logits_key]
except Exception as e:
    print(f"Error loading NPZ: {e}")
    exit(1)

# DECODE LOOP
print(f"Decoding {len(logits_list)} sentences...")
results = []

for i, logits_raw in enumerate(tqdm(logits_list)):
    # Reorder columns (Python -> C++ Map) using the helper function
    # This fixes the "Zsa Zsa" bug by moving Silence from index 41 to 1
    logits_reordered = rearrange_speech_logits_pt(logits_raw)[0]
    
    try:
        decoder.Reset()
        # Pass raw logits (float32) to C++
        # C++ handles log_softmax internally
        lm_decoder.DecodeNumpy(decoder, logits_reordered, np.zeros_like(logits_reordered), np.log(args.lm_beta))
        decoder.FinishDecoding()
        
        res = decoder.result()
        text = res[0].sentence if len(res) > 0 else ""
    except Exception as e:
        print(f"Error on trial {i}: {e}")
        text = ""
    
    results.append(text)

# Save results
output_filename = "teammate_predictions.txt"
with open(output_filename, "w") as f:
    f.write("\n".join(results))
print(f"Saved predictions to {output_filename}")