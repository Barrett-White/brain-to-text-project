import numpy as np
import argparse
import lm_decoder
import os
import re
import editdistance
from tqdm import tqdm

# --- HELPER FUNCTIONS ---
def rearrange_speech_logits_pt(logits):
    # Rearrange: [BLANK, ...PHONEMES..., SIL] -> [BLANK, SIL, ...PHONEMES...]
    return np.concatenate((logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]), axis=-1)

def remove_punctuation(sentence):
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower().replace('--', '').replace(" '", "'").strip()
    return ' '.join(sentence.split())

# --- MAIN SCRIPT ---
parser = argparse.ArgumentParser()
parser.add_argument('--npz_file', type=str, required=True, help="Path to the NPZ file")
parser.add_argument('--lm_path', type=str, required=True, help="Path to language model directory")
# We include alpha/beta to match your main script structure
parser.add_argument('--lm_alpha', type=float, default=0.55, help="Language Model Weight (Legacy/Rescoring)")
parser.add_argument('--lm_beta', type=float, default=90.0, help="Word insertion penalty")
parser.add_argument('--acoustic_scale', type=float, default=0.325, help="Acoustic scale")
parser.add_argument('--logits_key', type=str, default='logits', help="Key for input logits")
parser.add_argument('--labels_key', type=str, default='sentence_label', help="Key for true text labels")
args = parser.parse_args()

# 1. SETUP DECODER
print("Initializing Decoder...")
TLG_path = os.path.join(args.lm_path, 'TLG.fst')
words_path = os.path.join(args.lm_path, 'words.txt')

# Alpha is unused in the C++ constructor, but we keep acoustic_scale
decode_opts = lm_decoder.DecodeOptions(7000, 200, 17.0, 8.0, args.acoustic_scale, 1.0, 0.0, 1)
decode_resource = lm_decoder.DecodeResource(TLG_path, "", "", words_path, "")
decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

# 2. LOAD DATA
print(f"Loading {args.npz_file}...")
try:
    data = np.load(args.npz_file, allow_pickle=True)
    
    if args.logits_key not in data:
        first_key = list(data.keys())[0]
        print(f"WARNING: Key '{args.logits_key}' not found. Defaulting to: '{first_key}'")
        logits_list = data[first_key]
    else:
        logits_list = data[args.logits_key]

    if args.labels_key in data:
        print(f"Found labels under key: '{args.labels_key}'")
        labels_list = data[args.labels_key]
    else:
        # Auto-detect labels
        found_label = False
        for key in ['transcriptions', 'labels', 'y', 'text']:
            if key in data:
                print(f"Auto-detected labels under key: '{key}'")
                labels_list = data[key]
                found_label = True
                break
        if not found_label:
            print("WARNING: No labels found. WER cannot be calculated.")
            labels_list = None

except Exception as e:
    print(f"Error loading NPZ: {e}")
    exit(1)

# 3. DECODE & CALCULATE WER
print(f"Decoding {len(logits_list)} sentences...")
results = []
total_dist = 0
total_words = 0

for i, logits_raw in enumerate(tqdm(logits_list)):
    # Reorder Logits (The Critical Fix)
    logits_reshaped = logits_raw[np.newaxis, :, :]
    logits_reordered = rearrange_speech_logits_pt(logits_reshaped)[0]
    
    # Run Decoder
    try:
        decoder.Reset()
        # We pass acoustic_scale (in setup) and beta (here). Alpha is implicit in the graph.
        lm_decoder.DecodeNumpy(decoder, logits_reordered, np.zeros_like(logits_reordered), np.log(args.lm_beta))
        decoder.FinishDecoding()
        res = decoder.result()
        pred_text = res[0].sentence if len(res) > 0 else ""
    except Exception as e:
        print(f"Error on trial {i}: {e}")
        pred_text = ""
    
    results.append(pred_text)

    # Calculate WER
    if labels_list is not None:
        true_raw = labels_list[i]
        if isinstance(true_raw, bytes): true_raw = true_raw.decode('utf-8')
        
        t_cl = remove_punctuation(str(true_raw))
        p_cl = remove_punctuation(pred_text)
        
        dist = editdistance.eval(t_cl.split(), p_cl.split())
        total_dist += dist
        total_words += len(t_cl.split())
        
        if i % 50 == 0:
            print(f"\nTrial {i}")
            print(f"True: {t_cl}")
            print(f"Pred: {p_cl}")
            if len(t_cl.split()) > 0:
                print(f"WER: {dist / len(t_cl.split()):.2f}")

# 4. FINAL REPORT
print("-" * 30)
if total_words > 0:
    print(f"Aggregate Word Error Rate (WER): {100 * total_dist / total_words:.2f}%")
else:
    print("Aggregate WER: N/A")

output_filename = "teammate_predictions.txt"
with open(output_filename, "w") as f:
    f.write("\n".join(results))
print(f"Saved predictions to {output_filename}")