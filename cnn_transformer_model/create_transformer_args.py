import os

from omegaconf import OmegaConf

args_path = "baseline/rnn_args.yaml"
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

with open("cnn_transformer_model/transformer_args.yaml", "w") as f:
    OmegaConf.save(args, f)
