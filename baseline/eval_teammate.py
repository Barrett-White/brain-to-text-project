import numpy as np
import argparse
import lm_decoder
import os
import re
import editdistance
from tqdm import tqdm
import pandas as pd
from omegaconf import OmegaConf

# Import existing helpers
from evaluate_model_helpers import load_h5py_file, rearrange_speech_logits_pt

def remove_punctuation(sentence):
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower().replace('--', '').replace(" '", "'").strip()
    return ' '.join(sentence.split())

# --- ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument('--npz_file', type=str, required=True, help="Teammate's logits file")
parser.add_argument('--lm_path', type=str, required=True, help="Path to language model folder")
parser.add_argument('--model_path', type=str, required=True, help="Path to YOUR model folder (to find args.yaml)")
parser.add_argument('--data_dir', type=str, required=True, help="Path to YOUR hdf5 data")
parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv')
parser.add_argument('--eval_type', type=str, default='val', help="val or test")

# Tuning
parser.add_argument('--lm_beta', type=float, default=90.0)
parser.add_argument('--acoustic_scale', type=float, default=0.325)
parser.add_argument('--logits_key', type=str, default='logits')
args = parser.parse_args()

# 1. SETUP DECODER
print("Initializing Decoder...")
TLG_path = os.path.join(args.lm_path, 'TLG.fst')
words_path = os.path.join(args.lm_path, 'words.txt')
decode_opts = lm_decoder.DecodeOptions(7000, 200, 17.0, 8.0, args.acoustic_scale, 1.0, 0.0, 1)
decode_resource = lm_decoder.DecodeResource(TLG_path, "", "", words_path, "")
decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

# 2. LOAD TRUE LABELS & METADATA
print(f"Loading Ground Truth from {args.data_dir}...")
b2txt_csv_df = pd.read_csv(args.csv_path)
model_args = OmegaConf.load(os.path.join(args.model_path, 'checkpoint/args.yaml'))

# We need to store metadata to match the print format of evaluate_model.py
meta_sessions = []
meta_trials = []
meta_labels = []

for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(args.data_dir, session)) if f.endswith('.hdf5')]
    target_file = f'data_{args.eval_type}.hdf5'
    
    if target_file in files:
        data = load_h5py_file(os.path.join(args.data_dir, session, target_file), b2txt_csv_df)
        
        # Extend our metadata lists
        meta_sessions.extend([session] * len(data['sentence_label']))
        meta_trials.extend(data['trial_num'])
        meta_labels.extend(data['sentence_label'])

print(f"Loaded {len(meta_labels)} labels.")

# 3. LOAD TEAMMATE LOGITS
print(f"Loading Logits from {args.npz_file}...")
try:
    npz_data = np.load(args.npz_file, allow_pickle=True)
    if args.logits_key not in npz_data:
        key = list(npz_data.keys())[0]
        print(f"WARNING: Key '{args.logits_key}' not found. Using '{key}'")
        logits_list = npz_data[key]
    else:
        logits_list = npz_data[args.logits_key]
except Exception as e:
    print(f"Error loading NPZ: {e}")
    exit(1)

# 4. ALIGNMENT CHECK
min_len = min(len(logits_list), len(meta_labels))
if len(logits_list) != len(meta_labels):
    print(f"WARNING: Count mismatch. Logits: {len(logits_list)}, Labels: {len(meta_labels)}")
    print(f"Truncating to {min_len}...")

# 5. DECODE LOOP
print(f"Decoding {min_len} sentences...")
pred_sentences = []

for i in tqdm(range(min_len)):
    logits_raw = logits_list[i]

    # Reorder Logits
    logits_reshaped = logits_raw[np.newaxis, :, :]
    logits_reordered = rearrange_speech_logits_pt(logits_reshaped)[0]
    
    # Decode
    try:
        decoder.Reset()
        lm_decoder.DecodeNumpy(decoder, logits_reordered, np.zeros_like(logits_reordered), np.log(args.lm_beta))
        decoder.FinishDecoding()
        res = decoder.result()
        pred_text = res[0].sentence if len(res) > 0 else ""
    except Exception as e:
        pred_text = ""
    
    pred_sentences.append(pred_text)

# 6. WER CALCULATION & PRINTING (Exact format of evaluate_model.py)
print("\n--- RESULTS ---")
total_dist = 0
total_words = 0

for i in range(len(pred_sentences)):
    # Retrieve metadata for this specific trial
    session_id = meta_sessions[i]
    trial_id = meta_trials[i]
    true_raw = meta_labels[i]
    
    # Handle bytes if necessary
    if isinstance(true_raw, bytes): 
        true_raw = true_raw.decode('utf-8')

    # Clean strings
    t_cl = remove_punctuation(str(true_raw))
    p_cl = remove_punctuation(pred_sentences[i])
    
    # Calculate Edit Distance
    dist = editdistance.eval(t_cl.split(), p_cl.split())
    total_dist += dist
    total_words += len(t_cl.split())

    # Exact Print Format requested
    print(f'{session_id} - Trial {trial_id}')
    print(f'True: {t_cl}')
    print(f'Pred: {p_cl}')
    
    if len(t_cl.split()) > 0:
        wer = dist / len(t_cl.split())
        print(f'WER: {wer:.2f}')
    else:
        print('WER: N/A')
    print()

# Final Summary
print("-" * 30)
if total_words > 0:
    agg_wer = 100 * total_dist / total_words
    print(f"Aggregate Word Error Rate (WER): {agg_wer:.2f}%")
else:
    print("Aggregate WER: N/A")

# Save
with open("teammate_wer_results.txt", "w") as f:
    f.write("\n".join(pred_sentences))
print("Predictions saved to teammate_wer_results.txt")