import numpy as np
import argparse
import lm_decoder
import os
from tqdm import tqdm

# --- HELPER FUNCTION (The Critical Mapping Fix) ---
def rearrange_speech_logits_pt(logits):
    # Rearrange: [BLANK, ...PHONEMES..., SIL] -> [BLANK, SIL, ...PHONEMES...]
    # This maps Python Index 41 (Silence) to C++ Index 1 (Silence)
    return np.concatenate((logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]), axis=-1)

# --- MAIN SCRIPT ---
parser = argparse.ArgumentParser()
parser.add_argument('--npz_file', type=str, required=True, help="Path to the NPZ file")
parser.add_argument('--lm_path', type=str, required=True, help="Path to language model directory")
parser.add_argument('--lm_beta', type=float, default=90.0, help="Word insertion penalty")
parser.add_argument('--acoustic_scale', type=float, default=0.325, help="Acoustic scale")
parser.add_argument('--logits_key', type=str, default='logits', help="Key inside the NPZ file")
args = parser.parse_args()

# 1. SETUP DECODER
print("Initializing Decoder...")
TLG_path = os.path.join(args.lm_path, 'TLG.fst')
words_path = os.path.join(args.lm_path, 'words.txt')

# Standard Options: max_active=7000, min_active=200, beam=17.0, lattice_beam=8.0
decode_opts = lm_decoder.DecodeOptions(7000, 200, 17.0, 8.0, args.acoustic_scale, 1.0, 0.0, 1)
decode_resource = lm_decoder.DecodeResource(TLG_path, "", "", words_path, "")
decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

# 2. LOAD DATA
print(f"Loading {args.npz_file}...")
try:
    # FIX: allow_pickle=True lets it load complex arrays/lists
    data = np.load(args.npz_file, allow_pickle=True)
    
    # Auto-detect key if default fails
    if args.logits_key not in data:
        print(f"Key '{args.logits_key}' not found. Available keys: {list(data.keys())}")
        first_key = list(data.keys())[0]
        print(f"Defaulting to first available key: '{first_key}'")
        logits_list = data[first_key]
    else:
        logits_list = data[args.logits_key]

except Exception as e:
    print(f"Error loading NPZ: {e}")
    exit(1)

# 3. DECODE LOOP
print(f"Decoding {len(logits_list)} sentences...")
results = []

for i, logits_raw in enumerate(tqdm(logits_list)):
    # Apply the "Zsa Zsa" fix (Reordering)
    # Note: We assume logits_raw shape is (Time, 42)
    # We add a batch dim (1, Time, 42) for the function, then remove it [0]
    logits_reshaped = logits_raw[np.newaxis, :, :]
    logits_reordered = rearrange_speech_logits_pt(logits_reshaped)[0]
    
    try:
        decoder.Reset()
        # Pass raw logits (float32) to C++
        # We use np.log(beta) because the C++ engine expects it
        lm_decoder.DecodeNumpy(decoder, logits_reordered, np.zeros_like(logits_reordered), np.log(args.lm_beta))
        decoder.FinishDecoding()
        
        res = decoder.result()
        text = res[0].sentence if len(res) > 0 else ""
    except Exception as e:
        print(f"Error on trial {i}: {e}")
        text = ""
    
    results.append(text)
    # print(f"Trial {i}: {text}") # Uncomment to print every line

# 4. SAVE RESULTS
output_filename = "teammate_predictions.txt"
with open(output_filename, "w") as f:
    f.write("\n".join(results))

print(f"\nDone! Saved {len(results)} predictions to {output_filename}")