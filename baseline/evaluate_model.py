import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

# Import the compiled C++ extension
import lm_decoder

from rnn_model import GRUDecoder
from evaluate_model_helpers import *

parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model.')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'])
parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv')
parser.add_argument('--gpu_number', type=int, default=0)

# Decoder Arguments
parser.add_argument('--lm_path', type=str, required=True)
parser.add_argument('--lm_alpha', type=float, default=0.2)
parser.add_argument('--lm_beta', type=float, default=2.0)
parser.add_argument('--acoustic_scale', type=float, default=0.8)
parser.add_argument('--beam', type=float, default=17.0)

args = parser.parse_args()

def get_reorder_indices(python_vocab, tokens_txt_path):
    """
    Creates a mapping to shuffle Python logits to match C++ tokens.txt
    """
    # 1. Load C++ Map
    cpp_map = {}
    with open(tokens_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token = parts[0]
                idx = int(parts[1])
                cpp_map[token] = idx
    
    # 2. Create Reorder List
    # Size is max index found in C++ map + 1 (usually 58)
    max_cpp_idx = max(cpp_map.values())
    reorder_list = [0] * (max_cpp_idx + 1)
    
    # 3. Fill the map
    # Iterate through C++ expectations and find where they are in Python list
    for token, cpp_idx in cpp_map.items():
        # Handle Special Cases
        if token == '<eps>': continue # Not used
        if token == '<blk>': 
            py_idx = 0 # Python BLANK
        elif token == 'SIL':
            py_idx = 41 # Python Space (' | ')
        elif token in python_vocab:
            py_idx = python_vocab.index(token)
        else:
            # Disambiguation symbols (#0, #1) or missing tokens get mapped to Blank
            py_idx = 0 
            
        reorder_list[cpp_idx] = py_idx
        
    return np.array(reorder_list, dtype=np.int32)
# ------------------------------------------

# --- SETUP ---
model_path = args.model_path
data_dir = args.data_dir
eval_type = args.eval_type
device = torch.device(f"cuda:{args.gpu_number}" if torch.cuda.is_available() else "cpu")

# Load Metadata
b2txt_csv_df = pd.read_csv(args.csv_path)
model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))

# Load Model
model = GRUDecoder(
    neural_dim = model_args['model']['n_input_features'],
    n_units = model_args['model']['n_units'], 
    n_days = len(model_args['dataset']['sessions']),
    n_classes = model_args['dataset']['n_classes'],
    rnn_dropout = model_args['model']['rnn_dropout'],
    input_dropout = model_args['model']['input_network']['input_layer_dropout'],
    n_layers = model_args['model']['n_layers'],
    patch_size = model_args['model']['patch_size'],
    patch_stride = model_args['model']['patch_stride'],
)

checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), map_location=device, weights_only=False)
for key in list(checkpoint['model_state_dict'].keys()):
    checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
    checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
model.load_state_dict(checkpoint['model_state_dict'])  
model.to(device)
model.eval()

# Load Data
test_data = {}
total_test_trials = 0
for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
    if f'data_{eval_type}.hdf5' in files:
        eval_file = os.path.join(data_dir, session, f'data_{eval_type}.hdf5')
        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data
        total_test_trials += len(test_data[session]["neural_features"])

# --- STEP 1: INFERENCE ---
print("\n--- STARTING INFERENCE ---")
with tqdm(total=total_test_trials, desc='Running Inference', unit='trial') as pbar:
    for session, data in test_data.items():
        data['logits'] = []
        input_layer = model_args['dataset']['sessions'].index(session)
        
        for trial in range(len(data['neural_features'])):
            neural_input = data['neural_features'][trial]
            neural_input = np.expand_dims(neural_input, axis=0)
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32) # float32 for V100

            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)
            pbar.update(1)

# --- STEP 2: DECODING ---
print("\nInitializing C++ Decoder...")

TLG_path = os.path.join(args.lm_path, 'TLG.fst')
words_path = os.path.join(args.lm_path, 'words.txt')
tokens_path = os.path.join(args.lm_path, 'tokens.txt') # Needed for mapping

# Build the Reordering Map
print("Building Logit Reorder Map...")
reorder_indices = get_reorder_indices(LOGIT_TO_PHONEME, tokens_path)

# Initialize Decoder
decode_opts = lm_decoder.DecodeOptions(
    7000, 200, args.beam, 8.0, args.acoustic_scale, 1.0, 0.0, 1
)
decode_resource = lm_decoder.DecodeResource(TLG_path, "", "", words_path, "")
decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

lm_results = {'session': [], 'block': [], 'trial': [], 'true_sentence': [], 'pred_sentence': []}

print("Running Decoding...")
with tqdm(total=total_test_trials, desc='Decoding', unit='trial') as pbar:
    for session in test_data.keys():
        for trial in range(len(test_data[session]['logits'])):
            
            # Prepare Logits
            logits_tensor = test_data[session]['logits'][trial]
            if torch.is_tensor(logits_tensor):
                logits_np = logits_tensor.squeeze(0).float().cpu().numpy()
            else:
                logits_np = logits_tensor[0]

            # --- APPLY REORDERING ---
            # This shuffles the columns to match what C++ expects
            # New_Logits[:, C++_Index] = Old_Logits[:, Python_Index]
            # We must pad logits if C++ expects more tokens (e.g. disambiguation symbols)
            T, C = logits_np.shape
            if len(reorder_indices) > C:
                # Pad with -inf (impossible probability)
                pad_width = len(reorder_indices) - C
                # Create a larger array filled with very low probability
                padded_logits = np.full((T, len(reorder_indices)), -10000.0, dtype=np.float32)
                # Map the known logits into their new positions
                # We can't do simple indexing because reorder_indices maps Target -> Source
                # and some targets (like #0) map to Source 0 (Blank), which is wrong.
                # So we do it explicitly for valid tokens:
                
                # Faster numpy way:
                # reorder_indices[j] tells us which column from logits_np goes to column j in new array
                valid_indices = reorder_indices < C
                padded_logits[:, valid_indices] = logits_np[:, reorder_indices[valid_indices]]
                logits_ready = padded_logits
            else:
                logits_ready = logits_np[:, reorder_indices]
            # ------------------------

            try:
                decoder.Reset()
                lm_decoder.DecodeNumpy(decoder, logits_ready, np.zeros_like(logits_ready), np.log(args.lm_beta))
                decoder.FinishDecoding()
                
                if len(decoder.result()) > 0:
                    decoded_text = decoder.result()[0].sentence
                else:
                    decoded_text = ""
            except Exception as e:
                print(f"Error: {e}")
                decoded_text = ""

            lm_results['session'].append(session)
            lm_results['block'].append(test_data[session]['block_num'][trial])
            lm_results['trial'].append(test_data[session]['trial_num'][trial])
            
            if eval_type == 'val':
                lm_results['true_sentence'].append(test_data[session]['sentence_label'][trial])
            else:
                lm_results['true_sentence'].append(None)
                
            lm_results['pred_sentence'].append(decoded_text)
            pbar.update(1)

# --- STEP 3: WER ---
if eval_type == 'val':
    total_true_length = 0
    total_edit_distance = 0

    for i in range(len(lm_results['pred_sentence'])):
        true_sentence = remove_punctuation(lm_results['true_sentence'][i] or "").strip()
        pred_sentence = remove_punctuation(lm_results['pred_sentence'][i] or "").strip()
        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())

        total_true_length += len(true_sentence.split())
        total_edit_distance += ed

        print(f'{lm_results["session"][i]} - Trial {lm_results["trial"][i]}')
        print(f'True: {true_sentence}')
        print(f'Pred: {pred_sentence}')
        print(f'WER: {ed / len(true_sentence.split()):.2f}' if len(true_sentence.split()) > 0 else 'N/A')
        print()

    if total_true_length > 0:
        print(f'Aggregate WER: {100 * total_edit_distance / total_true_length:.2f}%')

output_file = os.path.join(model_path, f'baseline_rnn_{eval_type}_predicted_sentences_{time.strftime("%Y%m%d_%H%M%S")}.csv')
df_out = pd.DataFrame({'id': range(len(lm_results['pred_sentence'])), 'text': lm_results['pred_sentence']})
df_out.to_csv(output_file, index=False)
print(f"Saved to {output_file}")