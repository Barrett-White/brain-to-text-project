import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse
import lm_decoder
import re  # <--- FIXED: Added missing import
from rnn_model import GRUDecoder

# --- HELPER FUNCTIONS ---
LOGIT_TO_PHONEME = [
    'BLANK', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', ' | '
]

def rearrange_speech_logits_pt(logits):
    # Rearrange: [BLANK, ...PHONEMES..., SIL] -> [BLANK, SIL, ...PHONEMES...]
    return np.concatenate((logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]), axis=-1)

def remove_punctuation(sentence):
    # Normalize string: remove punctuation, lowercase, single spaces
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower().replace('--', '').replace(" '", "'").strip()
    return ' '.join(sentence.split())

def load_h5py_file(file_path, b2txt_csv_df):
    import h5py
    data = {'neural_features': [], 'sentence_label': [], 'block_num': [], 'trial_num': []}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            g = f[key]
            data['neural_features'].append(g['input_features'][:])
            data['sentence_label'].append(g.attrs['sentence_label'])
            data['block_num'].append(g.attrs['block_num'])
            data['trial_num'].append(g.attrs['trial_num'])
    return data

def runSingleDecodingStep(x, input_layer, model, device):
    # Pass session index as a tensor so the RNN selects the right day-weights
    day_idx_tensor = torch.tensor([input_layer], device=device)
    with torch.no_grad():
        logits, _ = model(x=x, day_idx=day_idx_tensor, states=None, return_state=True)
    return logits.float().cpu().numpy()

# --- MAIN SCRIPT ---
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--eval_type', type=str, default='test')
parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv')
parser.add_argument('--gpu_number', type=int, default=0)
parser.add_argument('--lm_path', type=str, required=True)
parser.add_argument('--lm_alpha', type=float, default=0.55)
parser.add_argument('--lm_beta', type=float, default=2.0)
parser.add_argument('--acoustic_scale', type=float, default=0.325)
parser.add_argument('--beam', type=float, default=17.0)
args = parser.parse_args()

# Setup
device = torch.device(f"cuda:{args.gpu_number}" if torch.cuda.is_available() else "cpu")
b2txt_csv_df = pd.read_csv(args.csv_path)
model_args = OmegaConf.load(os.path.join(args.model_path, 'checkpoint/args.yaml'))

model = GRUDecoder(
    neural_dim=model_args['model']['n_input_features'],
    n_units=model_args['model']['n_units'],
    n_days=len(model_args['dataset']['sessions']),
    n_classes=model_args['dataset']['n_classes'],
    rnn_dropout=model_args['model']['rnn_dropout'],
    input_dropout=model_args['model']['input_network']['input_layer_dropout'],
    n_layers=model_args['model']['n_layers'],
    patch_size=model_args['model']['patch_size'],
    patch_stride=model_args['model']['patch_stride'],
)
checkpoint = torch.load(os.path.join(args.model_path, 'checkpoint/best_checkpoint'), map_location=device)
state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(state_dict)
model.to(device).eval()

# Decoder Setup
TLG_path = os.path.join(args.lm_path, 'TLG.fst')
words_path = os.path.join(args.lm_path, 'words.txt')
decode_opts = lm_decoder.DecodeOptions(7000, 200, args.beam, 8.0, args.acoustic_scale, 1.0, 0.0, 1)
decode_resource = lm_decoder.DecodeResource(TLG_path, "", "", words_path, "")
decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

print("\n--- STARTING EVALUATION ---")
lm_results = {'session': [], 'trial': [], 'true': [], 'pred': []}

# Load Data Loop
for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(args.data_dir, session)) if f.endswith('.hdf5')]
    if f'data_{args.eval_type}.hdf5' not in files: continue
    
    print(f"Processing Session: {session}")
    data = load_h5py_file(os.path.join(args.data_dir, session, f'data_{args.eval_type}.hdf5'), b2txt_csv_df)
    
    input_layer = model_args['dataset']['sessions'].index(session)
    
    for trial in tqdm(range(len(data['neural_features']))):
        neural_input = torch.tensor(data['neural_features'][trial], device=device, dtype=torch.float32).unsqueeze(0)
        
        # 1. Inference
        logits_raw = runSingleDecodingStep(neural_input, input_layer, model, device)
        
        # 2. Reorder (Mapping Fix)
        logits_reordered = rearrange_speech_logits_pt(logits_raw)[0]

        # 3. Decode
        try:
            decoder.Reset()
            # Send Raw Logits to C++ (No Python Softmax)
            lm_decoder.DecodeNumpy(decoder, logits_reordered, np.zeros_like(logits_reordered), np.log(args.lm_beta))
            decoder.FinishDecoding()
            res = decoder.result()
            decoded_text = res[0].sentence if len(res) > 0 else ""
        except Exception as e:
            print(f"Error: {e}")
            decoded_text = ""

        lm_results['session'].append(session)
        lm_results['trial'].append(data['trial_num'][trial])
        lm_results['true'].append(data['sentence_label'][trial])
        lm_results['pred'].append(decoded_text)

# WER Calculation
total_dist, total_words = 0, 0
for t, p in zip(lm_results['true'], lm_results['pred']):
    t_cl = remove_punctuation(t or "")
    p_cl = remove_punctuation(p or "")
    dist = editdistance.eval(t_cl.split(), p_cl.split())
    total_dist += dist
    total_words += len(t_cl.split())
    
    print(f'{lm_results["session"][len(lm_results["session"]) - len(lm_results["pred"]) + 0]}') # debug print
    print(f'True: {t_cl}')
    print(f'Pred: {p_cl}')
    print(f'WER: {dist / len(t_cl.split()):.2f}' if len(t_cl.split()) > 0 else 'N/A')
    print()

print(f"\nAggregate WER: {100 * total_dist / total_words:.2f}%" if total_words > 0 else "N/A")
df = pd.DataFrame(lm_results)
df.to_csv(os.path.join(args.model_path, f'results_{args.eval_type}.csv'), index=False)