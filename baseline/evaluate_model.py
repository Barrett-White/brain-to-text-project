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

from rnn_model import GRUDecoder
from evaluate_model_helpers import *

# argument parser for command line arguments
parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset.')
parser.add_argument('--model_path', type=str, default='../data/t15_pretrained_rnn_baseline',
                    help='Path to the pretrained model directory (relative to the current working directory).')
parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_final',
                    help='Path to the dataset directory (relative to the current working directory).')
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                    help='Evaluation type: "val" for validation set, "test" for test set. '
                         'If "test", ground truth is not available.')
parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv',
                    help='Path to the CSV file with metadata about the dataset (relative to the current working directory).')
parser.add_argument('--gpu_number', type=int, default=1,
                    help='GPU number to use for RNN model inference. Set to -1 to use CPU.')

# --- NEW DECODER ARGUMENTS ---
parser.add_argument('--lm_path', type=str, default='../language_model/pretrained_language_models/3gram.model',
                    help='Path to the KenLM n-gram model file.')
parser.add_argument('--lm_alpha', type=float, default=1.5,
                    help='Language Model weight (Alpha).')
parser.add_argument('--lm_beta', type=float, default=2.0,
                    help='Word Insertion Penalty (Beta).')
# -----------------------------

args = parser.parse_args()

# paths to model and data directories
model_path = args.model_path
data_dir = args.data_dir
eval_type = args.eval_type  

# load csv file
b2txt_csv_df = pd.read_csv(args.csv_path)

# load model args
model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))

# set up gpu device
gpu_number = args.gpu_number
if torch.cuda.is_available() and gpu_number >= 0:
    if gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
    device = f'cuda:{gpu_number}'
    device = torch.device(device)
    print(f'Using {device} for model inference.')
else:
    if gpu_number >= 0:
        print(f'GPU number {gpu_number} requested but not available.')
    print('Using CPU for model inference.')
    device = torch.device('cpu')

# define model
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

# load model weights
checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), map_location=device, weights_only=False)
for key in list(checkpoint['model_state_dict'].keys()):
    checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
    checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
model.load_state_dict(checkpoint['model_state_dict'])  

# add model to device
model.to(device) 
model.eval()

# load data for each session
test_data = {}
total_test_trials = 0
for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
    if f'data_{eval_type}.hdf5' in files:
        eval_file = os.path.join(data_dir, session, f'data_{eval_type}.hdf5')

        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data

        total_test_trials += len(test_data[session]["neural_features"])
        print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')
print(f'Total number of {eval_type} trials: {total_test_trials}')
print()


# ---------------------------------------------------------
# STEP 1: RNN INFERENCE (Generate Acoustic Logits)
# ---------------------------------------------------------
with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():

        data['logits'] = []
        data['pred_seq'] = []
        input_layer = model_args['dataset']['sessions'].index(session)
        
        for trial in range(len(data['neural_features'])):
            # get neural input for the trial
            neural_input = data['neural_features'][trial]

            # add batch dimension
            neural_input = np.expand_dims(neural_input, axis=0)

            # convert to torch tensor
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

            # run decoding step
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)

            pbar.update(1)
pbar.close()


# ---------------------------------------------------------
# STEP 2: LANGUAGE MODEL DECODING (Local C++ Decoder)
# ---------------------------------------------------------
print("\nInitializing C++ LM Decoder...")

# Prepare Vocabulary List
# LOGIT_TO_PHONEME is a dictionary {index: 'phoneme'}.
# The decoder typically expects a list where list[i] is the label for index i.
vocab_list = [LOGIT_TO_PHONEME[i] for i in range(len(LOGIT_TO_PHONEME))]

# Initialize the Scorer
# Note: We assume the C++ API handles (alpha, beta, model_path, vocabulary)
scorer = lm_decoder.Scorer(
    args.lm_alpha, 
    args.lm_beta, 
    args.lm_path, 
    vocab_list
)

print(f"Decoder initialized with Alpha={args.lm_alpha}, Beta={args.lm_beta}")

lm_results = {
    'session': [],
    'block': [],
    'trial': [],
    'true_sentence': [],
    'pred_sentence': [],
}

print("Running Beam Search Decoding...")
with tqdm(total=total_test_trials, desc='Decoding with LM', unit='trial') as pbar:
    for session in test_data.keys():
        for trial in range(len(test_data[session]['logits'])):
            
            # Get logits for this trial
            # data['logits'][trial] is a tensor of shape [1, Time, Classes]
            # We need a numpy array of shape [Time, Classes]
            logits_tensor = test_data[session]['logits'][trial]
            
            # Ensure it's on CPU and convert to numpy
            if torch.is_tensor(logits_tensor):
                logits_np = logits_tensor.squeeze(0).float().cpu().numpy()
            else:
                logits_np = logits_tensor[0]

            # --- DIRECT DECODE ---
            try:
                decoded_text = scorer.decode(logits_np)
            except Exception as e:
                print(f"Decoding error on {session} trial {trial}: {e}")
                decoded_text = ""
            # ---------------------

            # store results
            lm_results['session'].append(session)
            lm_results['block'].append(test_data[session]['block_num'][trial])
            lm_results['trial'].append(test_data[session]['trial_num'][trial])
            
            if eval_type == 'val':
                lm_results['true_sentence'].append(test_data[session]['sentence_label'][trial])
            else:
                lm_results['true_sentence'].append(None)
                
            lm_results['pred_sentence'].append(decoded_text)

            # update progress bar
            pbar.update(1)
pbar.close()


# ---------------------------------------------------------
# STEP 3: WER CALCULATION & OUTPUT
# ---------------------------------------------------------
if eval_type == 'val':
    total_true_length = 0
    total_edit_distance = 0

    lm_results['edit_distance'] = []
    lm_results['num_words'] = []

    for i in range(len(lm_results['pred_sentence'])):
        true_sentence = remove_punctuation(lm_results['true_sentence'][i] or "").strip()
        pred_sentence = remove_punctuation(lm_results['pred_sentence'][i] or "").strip()
        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())

        total_true_length += len(true_sentence.split())
        total_edit_distance += ed

        lm_results['edit_distance'].append(ed)
        lm_results['num_words'].append(len(true_sentence.split()))

        print(f'{lm_results["session"][i]} - Block {lm_results["block"][i]}, Trial {lm_results["trial"][i]}')
        print(f'True sentence:       {true_sentence}')
        print(f'Predicted sentence:  {pred_sentence}')
        # Handle zero-length sentences to avoid division by zero
        denom = len(true_sentence.split())
        if denom > 0:
            print(f'WER: {ed} / {denom} = {ed / denom:.2f}')
        else:
            print(f'WER: {ed} / 0 = N/A')
        print()

    print(f'Total true sentence length: {total_true_length}')
    print(f'Total edit distance: {total_edit_distance}')
    if total_true_length > 0:
        print(f'Aggregate Word Error Rate (WER): {100 * total_edit_distance / total_true_length:.2f}%')
    else:
        print('Aggregate WER: N/A (Total true length is 0)')


# write predicted sentences to a csv file. put a timestamp in the filename (YYYYMMDD_HHMMSS)
output_file = os.path.join(model_path, f'baseline_rnn_{eval_type}_predicted_sentences_{time.strftime("%Y%m%d_%H%M%S")}.csv')
ids = [i for i in range(len(lm_results['pred_sentence']))]
df_out = pd.DataFrame({'id': ids, 'text': lm_results['pred_sentence']})
df_out.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")