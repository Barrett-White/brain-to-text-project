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
parser.add_argument('--lm_path', type=str, required=True, help='Directory containing TLG.fst and words.txt')
parser.add_argument('--lm_alpha', type=float, default=0.55, help='Used for rescoring (not used in baseline beam search).')
parser.add_argument('--lm_beta', type=float, default=2.0, help='Word Insertion Penalty (Blank Penalty).')
parser.add_argument('--acoustic_scale', type=float, default=0.325, help='Scaling factor for acoustic logits.')
parser.add_argument('--beam', type=float, default=17.0, help='Beam width.')

args = parser.parse_args()

# --- SETUP ---
model_path = args.model_path
data_dir = args.data_dir
eval_type = args.eval_type
device = torch.device(f"cuda:{args.gpu_number}" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

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

print(f'Total trials: {total_test_trials}')

# --- STEP 1: INFERENCE (Get Logits) ---
with tqdm(total=total_test_trials, desc='Running Inference', unit='trial') as pbar:
    for session, data in test_data.items():
        data['logits'] = []
        input_layer = model_args['dataset']['sessions'].index(session)
        
        for trial in range(len(data['neural_features'])):
            neural_input = data['neural_features'][trial]
            neural_input = np.expand_dims(neural_input, axis=0)
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.float16) # Changed to float16 for Volta compatibility

            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)
            pbar.update(1)

# --- STEP 2: DECODING (Using raw C++ API) ---
print("\nInitializing C++ Decoder...")

# 1. Define Paths
TLG_path = os.path.join(args.lm_path, 'TLG.fst')
words_path = os.path.join(args.lm_path, 'words.txt')

if not os.path.exists(TLG_path):
    raise ValueError(f"TLG.fst not found at {TLG_path}")

# 2. Configure Options
decode_opts = lm_decoder.DecodeOptions(
    7000,   # max_active
    200,    # min_active
    args.beam, 
    8.0,    # lattice_beam
    args.acoustic_scale, 
    1.0,    # ctc_blank_skip_threshold
    0.0,    # length_penalty
    1       # nbest
)

# 3. Load Resources
decode_resource = lm_decoder.DecodeResource(
    TLG_path,
    "", # G_path (unused for basic decoding)
    "", # rescore_G_path (unused)
    words_path,
    ""  # vocab_path (unused)
)

# 4. Instantiate the Engine
decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)
print("Decoder Ready.")

lm_results = {'session': [], 'block': [], 'trial': [], 'true_sentence': [], 'pred_sentence': []}

print("Running Beam Search Decoding...")
with tqdm(total=total_test_trials, desc='Decoding', unit='trial') as pbar:
    for session in test_data.keys():
        for trial in range(len(test_data[session]['logits'])):
            
            logits_tensor = test_data[session]['logits'][trial]
            if torch.is_tensor(logits_tensor):
                logits_np = logits_tensor.squeeze(0).float().cpu().numpy()
            else:
                logits_np = logits_tensor[0]

            # --- DIRECT C++ API CALL ---
            try:
                # Reset state for new sentence
                decoder.Reset()
                
                # Feed data (Note: np.log(beta) acts as the insertion penalty in this API)
                lm_decoder.DecodeNumpy(decoder, logits_np, np.zeros_like(logits_np), np.log(args.lm_beta))
                
                # Finalize
                decoder.FinishDecoding()
                
                # Extract result
                if len(decoder.result()) > 0:
                    decoded_text = decoder.result()[0].sentence
                else:
                    decoded_text = ""
                    
            except Exception as e:
                print(f"Decoding error: {e}")
                decoded_text = ""
            # ---------------------------

            lm_results['session'].append(session)
            lm_results['block'].append(test_data[session]['block_num'][trial])
            lm_results['trial'].append(test_data[session]['trial_num'][trial])
            
            if eval_type == 'val':
                lm_results['true_sentence'].append(test_data[session]['sentence_label'][trial])
            else:
                lm_results['true_sentence'].append(None)
                
            lm_results['pred_sentence'].append(decoded_text)
            pbar.update(1)

# --- STEP 3: CALCULATE WER ---
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
        print(f'True:      {true_sentence}')
        print(f'Pred:      {pred_sentence}')
        denom = len(true_sentence.split())
        wer = ed / denom if denom > 0 else 0.0
        print(f'WER: {wer:.2f}')
        print()

    print(f'Total Edit Distance: {total_edit_distance}')
    print(f'Total Words: {total_true_length}')
    if total_true_length > 0:
        print(f'Aggregate WER: {100 * total_edit_distance / total_true_length:.2f}%')

# Save CSV
output_file = os.path.join(model_path, f'baseline_rnn_{eval_type}_predicted_sentences_{time.strftime("%Y%m%d_%H%M%S")}.csv')
df_out = pd.DataFrame({'id': range(len(lm_results['pred_sentence'])), 'text': lm_results['pred_sentence']})
df_out.to_csv(output_file, index=False)
print(f"Saved results to {output_file}")