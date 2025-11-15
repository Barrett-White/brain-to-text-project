import os

from omegaconf import OmegaConf

# Create training arguments for the baseline transformer model

args_path = "baseline/rnn_args_sampled.yaml"
args = OmegaConf.load(args_path)

print("Loaded config from:", args_path)

args["dataset"]["dataset_dir"] = (
    "/content/drive/MyDrive/STOR566Project/data/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final"
)

args["output_dir"] = "/content/drive/MyDrive/STOR566Project/models/transformer_model"
args["checkpoint_dir"] = os.path.join(args["output_dir"], "checkpoint")

os.makedirs(args["output_dir"], exist_ok=True)
os.makedirs(args["checkpoint_dir"], exist_ok=True)

args["mode"] = "train"
args["gpu_number"] = "1"
# Transformer Parameters
args["dataset"]["batch_size"] = 8
args["dataset"]["num_dataloader_workers"] = 2
args["save_val_logits"] = True
args["save_val_data"] = False
args["use_torch_compile"] = True
args["use_amp"] = True
args["model"]["n_units"] = 256
args["model"]["n_layers"] = 2
args["model"]["patch_size"] = 0
args["model"]["patch_stride"] = 0
args["model"]["rnn_dropout"] = 0.1
args["model"]["ff_dim"] = 512
args["model"]["num_heads"] = 4

with open("cnn_transformer_model/transformer_args.yaml", "w") as f:
    OmegaConf.save(args, f)

# Create training arguments for the CNN transformer model

args_path = "baseline/rnn_args_sampled.yaml"
args = OmegaConf.load(args_path)

print("Loaded config from:", args_path)

args["dataset"]["dataset_dir"] = (
    "/content/drive/MyDrive/STOR566Project/data/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final"
)

args["output_dir"] = (
    "/content/drive/MyDrive/STOR566Project/models/cnn_transformer_model"
)
args["checkpoint_dir"] = os.path.join(args["output_dir"], "checkpoint")

os.makedirs(args["output_dir"], exist_ok=True)
os.makedirs(args["checkpoint_dir"], exist_ok=True)

args["mode"] = "train"
args["gpu_number"] = "1"
# CNN Transformer Parameters
args["save_val_logits"] = True
args["save_val_data"] = False
args["use_torch_compile"] = True
args["use_amp"] = True
args["model"]["n_units"] = 256
args["model"]["n_layers"] = 2
args["model"]["patch_size"] = 0
args["model"]["patch_stride"] = 0
args["model"]["rnn_dropout"] = 0.1
args["model"]["cnn_repo_id"] = "PierreGtch/EEGNetv4"
args["model"]["cnn_modelpath"] = "EEGNetv4_Lee2019_MI/model-params.pkl"
args["model"]["transformer_name"] = "google/flan-t5-base"
args["dataset"]["temporal_bin"] = 20  # 20 ms

with open("cnn_transformer_model/cnn_transformer_args.yaml", "w") as f:
    OmegaConf.save(args, f)
