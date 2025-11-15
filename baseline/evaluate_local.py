import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import argparse
import torchaudio.functional as F  # Added torchaudio import for edit distance

from rnn_model import GRUDecoder
from evaluate_model_helpers import * # import helper functions

def main():
    # argument parser for command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset.')
    
    # NOTE: I changed the default model_path to the one we fixed
    parser.add_argument('--model_path', type=str, default='trained_models/baseline_rnn',
                        help='Path to the pretrained model directory (relative to the current working directory).')
    
    parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_final',
                        help='Path to the dataset directory (relative to the current working directory).')
    parser.add_argument('--eval_type', type=str, default='val', choices=['val', 'test'],
                        help='Evaluation type: "val" for validation set, "test" for test set. '
                             'If "test", ground truth is not available.')
    parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv',
                        help='Path to the CSV file with metadata about the dataset (relative to the current working directory).')
    parser.add_argument('--gpu_number', type=int, default=0,
                        help='GPU number to use for RNN model inference. Set to -1 to use CPU.')
    args = parser.parse_args()

    # paths to model and data directories
    model_path = args.model_path
    data_dir = args.data_dir

    # define evaluation type
    eval_type = args.eval_type
    if eval_type == 'test':
        print("Warning: eval_type is 'test'. True phoneme sequences are not available, so PER cannot be calculated.")


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
    checkpoint_file = os.path.join(model_path, 'checkpoint/best_checkpoint')
    print(f"Loading checkpoint from: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    
    # rename keys
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


    # put neural data through the pretrained model to get phoneme predictions (logits)
    with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
        for session, data in test_data.items():

            data['logits'] = []
            data['pred_seq'] = []
            input_layer = model_args['dataset']['sessions'].index(session)
            
            for trial in range(len(data['neural_features'])):
                neural_input = data['neural_features'][trial]
                neural_input = np.expand_dims(neural_input, axis=0)
                neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)

                # --- Run Inference ---
                with torch.no_grad():
                    logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
                data['logits'].append(logits)

                pbar.update(1)
    pbar.close()

    print("\n--- DECODED PHONEME SEQUENCES ---")
    
    # Initialize variables for PER calculation
    total_edit_distance = 0
    total_seq_length = 0
    
    # convert logits to phoneme sequences and print them out
    for session, data in test_data.items():
        data['pred_seq'] = []
        for trial in range(len(data['logits'])):
            logits = data['logits'][trial][0]
            
            # Get raw prediction (as ints)
            pred_seq_raw = np.argmax(logits, axis=-1)
            pred_seq_raw = [int(p) for p in pred_seq_raw if p != 0] # remove blanks
            pred_seq_raw = [pred_seq_raw[i] for i in range(len(pred_seq_raw)) if i == 0 or pred_seq_raw[i] != pred_seq_raw[i-1]] # remove consecutive
            
            # convert to phonemes for printing
            pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq_raw]
            data['pred_seq'].append(pred_seq)

            # print out the predicted sequences
            block_num = data['block_num'][trial]
            trial_num = data['trial_num'][trial]
            print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')
            
            if eval_type == 'val':
                sentence_label = data['sentence_label'][trial]
                
                # Get raw true sequence (as ints)
                true_seq_raw = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
                
                # Convert true phonemes to text for printing
                true_seq_text = [LOGIT_TO_PHONEME[p] for p in true_seq_raw]

                print(f'Sentence label:     {sentence_label}')
                print(f'True sequence:      {" ".join(true_seq_text)}') # Use the text version for printing
                
                # Calculate PER for this trial
                # We use the raw number sequences (lists of ints)
                edit_distance = F.edit_distance(pred_seq_raw, true_seq_raw)
                total_edit_distance += edit_distance
                total_seq_length += len(true_seq_raw)
                
            print(f'Predicted Sequence: {" ".join(pred_seq)}')
            print()

    # Add final block to print the average PER
    if eval_type == 'val' and total_seq_length > 0:
        avg_PER = (total_edit_distance / total_seq_length) * 100
        print("\n--- FINAL PHONEME ERROR RATE (PER) ---")
        print(f'Total Edit Distance:    {total_edit_distance}')
        print(f'Total Phoneme Length: {total_seq_length}')
        print(f'Average PER:          {avg_PER:.2f}%')
    elif eval_type == 'val':
        print("\n--- FINAL PHONEME ERROR RATE (PER) ---")
        print("No validation trials found or processed.")


if __name__ == "__main__":
    main()
    print("\n--- Evaluation Finished ---")
    print("NOTE: Redis was skipped. Word Error Rate (WER) and submission CSV were not calculated.")