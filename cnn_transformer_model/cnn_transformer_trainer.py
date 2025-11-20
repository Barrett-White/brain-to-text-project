import json
import logging
import math
import os
import pathlib
import pickle
import random
import sys
import time

import numpy as np
import torch
import torchaudio.functional as taF
from braindecode.models import EEGNet
from mspca import mspca
from omegaconf import OmegaConf
from ssqueezepy import cwt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from array2image import array_to_image

from cnn_transformer_model.cnn_transformer_model import CNNTransformer
from cnn_transformer_model.dataset_transformer import (
    BrainToTextDataset,
    gauss_smooth,
    train_test_split_indicies,
)


class CNN_Transformer_Trainer:
    def __init__(self, args):
        self.args = args
        self.logger = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None

        self.best_val_PER = torch.inf
        self.best_val_loss = torch.inf

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.transform_args = self.args["dataset"]["data_transforms"]

        if args["mode"] == "train":
            os.makedirs(self.args["output_dir"], exist_ok=True)
        if (
            args["save_best_checkpoint"]
            or args["save_all_val_steps"]
            or args["save_final_model"]
        ):
            os.makedirs(self.args["checkpoint_dir"], exist_ok=True)

        self.logger = logging.getLogger("BrainToTextTrainer")
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="%(asctime)s: %(message)s")

        if args["mode"] == "train":
            fh = logging.FileHandler(
                str(pathlib.Path(self.args["output_dir"], "training_log"))
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        if torch.cuda.is_available():
            gpu_num = self.args.get("gpu_number", 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(
                    f"Invalid gpu_number value: {gpu_num}. Using 0 instead."
                )
                gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(
                    f"Requested GPU {gpu_num} not available. Using 0 instead."
                )
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                _ = torch.tensor([1.0]).to(self.device) * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f"Using device: {self.device}")

        if self.args["seed"] != -1:
            np.random.seed(self.args["seed"])
            random.seed(self.args["seed"])
            torch.manual_seed(self.args["seed"])

        eenet = EEGNet(
            n_chans=self.args["model"]["n_input_features"],
            n_outputs=self.args["model"]["n_units"],
            n_times=self.args["dataset"]["temporal_bin"],
        )

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.args["model"]["transformer_name"]
        )
        self.t5model = T5ForConditionalGeneration.from_pretrained(
            self.args["model"]["transformer_name"]
        )

        self.model = CNNTransformer(
            neural_dim=self.args["model"]["n_input_features"],
            n_units=self.args["model"]["n_units"],
            n_days=len(self.args["dataset"]["sessions"]),
            n_classes=self.args["dataset"]["n_classes"],
            rnn_dropout=self.args["model"]["rnn_dropout"],
            input_dropout=self.args["model"]["input_network"]["input_layer_dropout"],
            eenet_model=eenet,
            transformer_model=self.t5model,
            temporal_bin=self.args["dataset"]["temporal_bin"],
        )

        if self.args["use_torch_compile"]:
            self.model = torch.compile(self.model)

        self.logger.info("Initialized CNN transformer model")
        self.logger.info(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        day_params = 0
        for name, param in self.model.named_parameters():
            if "day" in name:
                day_params += param.numel()
        self.logger.info(
            f"Model has {day_params:,} day-specific parameters "
            f"| {((day_params / total_params) * 100):.2f}% of total parameters"
        )

        train_file_paths = [
            os.path.join(self.args["dataset"]["dataset_dir"], s, "data_train.hdf5")
            for s in self.args["dataset"]["sessions"]
        ]
        val_file_paths = [
            os.path.join(self.args["dataset"]["dataset_dir"], s, "data_val.hdf5")
            for s in self.args["dataset"]["sessions"]
        ]

        if len(set(train_file_paths)) != len(train_file_paths):
            raise ValueError("Duplicate sessions listed in the train dataset")
        if len(set(val_file_paths)) != len(val_file_paths):
            raise ValueError("Duplicate sessions listed in the val dataset")

        train_trials, _ = train_test_split_indicies(
            file_paths=train_file_paths,
            test_percentage=0,
            seed=self.args["dataset"]["seed"],
            bad_trials_dict=self.args["dataset"].get("bad_trials_dict", None),
        )
        _, val_trials = train_test_split_indicies(
            file_paths=val_file_paths,
            test_percentage=1,
            seed=self.args["dataset"]["seed"],
            bad_trials_dict=self.args["dataset"].get("bad_trials_dict", None),
        )

        with open(
            os.path.join(self.args["output_dir"], "train_val_trials.json"), "w"
        ) as f:
            json.dump({"train": train_trials, "val": val_trials}, f)

        feature_subset = None
        if ("feature_subset" in self.args["dataset"]) and self.args["dataset"][
            "feature_subset"
        ] is not None:
            feature_subset = self.args["dataset"]["feature_subset"]
            self.logger.info(f"Using only a subset of features: {feature_subset}")

        self.train_dataset = BrainToTextDataset(
            trial_indicies=train_trials,
            split="train",
            days_per_batch=self.args["dataset"]["days_per_batch"],
            n_batches=self.args["num_training_batches"],
            batch_size=self.args["dataset"]["batch_size"],
            must_include_days=None,
            random_seed=self.args["dataset"]["seed"],
            feature_subset=feature_subset,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=self.args["dataset"]["loader_shuffle"],
            num_workers=self.args["dataset"]["num_dataloader_workers"],
            pin_memory=True,
        )

        self.val_dataset = BrainToTextDataset(
            trial_indicies=val_trials,
            split="test",
            days_per_batch=None,
            n_batches=None,
            batch_size=self.args["dataset"]["batch_size"],
            must_include_days=None,
            random_seed=self.args["dataset"]["seed"],
            feature_subset=feature_subset,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        self.logger.info("Successfully initialized datasets")

        self.optimizer = self.create_optimizer()

        if self.args["lr_scheduler_type"] == "linear":
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.args["lr_min"] / self.args["lr_max"],
                total_iters=self.args["lr_decay_steps"],
            )
        elif self.args["lr_scheduler_type"] == "cosine":
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(
                self.optimizer
            )
        else:
            raise ValueError(
                f"Invalid lr_scheduler_type: {self.args['lr_scheduler_type']}"
            )

        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)

        if self.args["init_from_checkpoint"] and self.args["init_checkpoint_path"]:
            self.load_model_checkpoint(self.args["init_checkpoint_path"])

        # freeze specified model parameters
        for name, param in self.model.named_parameters():
            if "transformer_model" in name:
                if "lm_head" not in name:
                    param.requires_grad = False

        # Print out frozen info
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                self.logger.info(f"Frozen: {name}")

        self.model.to(self.device)

    def create_optimizer(self):
        bias_params = []
        day_params = []
        other_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            if "gru.bias" in name or "out.bias" in name:
                bias_params.append(p)
            elif "day" in name:
                day_params.append(p)
            else:
                other_params.append(p)

        param_groups = []
        if bias_params:
            param_groups.append(
                {"params": bias_params, "weight_decay": 0.0, "group_type": "bias"}
            )
        if day_params:
            param_groups.append(
                {
                    "params": day_params,
                    "lr": self.args["lr_max_day"],
                    "weight_decay": self.args["weight_decay_day"],
                    "group_type": "day_layer",
                }
            )
        if other_params:
            param_groups.append({"params": other_params, "group_type": "other"})

        try:
            optim = torch.optim.AdamW(
                param_groups,
                lr=self.args["lr_max"],
                betas=(self.args["beta0"], self.args["beta1"]),
                eps=self.args["epsilon"],
                weight_decay=self.args["weight_decay"],
                fused=True,
            )
        except TypeError:
            optim = torch.optim.AdamW(
                param_groups,
                lr=self.args["lr_max"],
                betas=(self.args["beta0"], self.args["beta1"]),
                eps=self.args["epsilon"],
                weight_decay=self.args["weight_decay"],
            )

        return optim

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args["lr_max"]
        lr_min = self.args["lr_min"]
        lr_decay_steps = self.args["lr_decay_steps"]

        lr_max_day = self.args["lr_max_day"]
        lr_min_day = self.args["lr_min_day"]
        lr_decay_steps_day = self.args["lr_decay_steps_day"]

        lr_warmup_steps = self.args["lr_warmup_steps"]
        lr_warmup_steps_day = self.args["lr_warmup_steps_day"]

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(
                    min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
                )
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps
                ),
                lambda step: lr_lambda(
                    step,
                    lr_min_day / lr_max_day,
                    lr_decay_steps_day,
                    lr_warmup_steps_day,
                ),
                lambda step: lr_lambda(
                    step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps
                ),
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps
                ),
                lambda step: lr_lambda(
                    step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps
                ),
            ]
        else:
            raise ValueError(
                f"Unexpected number of param groups: {len(optim.param_groups)}"
            )

        return LambdaLR(optim, lr_lambdas, -1)

    def load_model_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.learning_rate_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_PER = checkpoint.get("val_PER", torch.inf)
        self.best_val_loss = checkpoint.get("val_loss", torch.inf)

        self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info(f"Loaded model from checkpoint: {load_path}")

    def save_model_checkpoint(self, save_path, PER, loss):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.learning_rate_scheduler.state_dict(),
            "val_PER": PER,
            "val_loss": loss,
        }
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved model to checkpoint: {save_path}")

        args_save_path = os.path.join(self.args["checkpoint_dir"], "args.yaml")
        OmegaConf.save(config=self.args, f=args_save_path)

    def transform_data(self, features, n_time_steps, mode="train"):
        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        if mode == "train":
            if self.transform_args["static_gain_std"] > 0:
                warp_mat = torch.tile(
                    torch.unsqueeze(torch.eye(channels, device=self.device), dim=0),
                    (batch_size, 1, 1),
                )
                warp_mat += (
                    torch.randn_like(warp_mat) * self.transform_args["static_gain_std"]
                )
                features = torch.matmul(features, warp_mat)

            if self.transform_args["white_noise_std"] > 0:
                features += (
                    torch.randn(data_shape, device=self.device)
                    * self.transform_args["white_noise_std"]
                )

            if self.transform_args["constant_offset_std"] > 0:
                features += (
                    torch.randn((batch_size, 1, channels), device=self.device)
                    * self.transform_args["constant_offset_std"]
                )

            if self.transform_args["random_walk_std"] > 0:
                features += torch.cumsum(
                    torch.randn(data_shape, device=self.device)
                    * self.transform_args["random_walk_std"],
                    dim=self.transform_args["random_walk_axis"],
                )

            if self.transform_args["random_cut"] > 0:
                cut = np.random.randint(0, self.transform_args["random_cut"])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        if self.transform_args["smooth_data"]:
            features = gauss_smooth(
                inputs=features,
                device=self.device,
                smooth_kernel_std=self.transform_args["smooth_kernel_std"],
                smooth_kernel_size=self.transform_args["smooth_kernel_size"],
            )

        if self.transform_args["turn_into_image"]:
            # mspca function
            # turn features into numpy array
            features = features.cpu().numpy()
            # we have batches, so we need to loop over the batch dimension
            for i in tqdm(range(features.shape[0])):
                mymodel = mspca.MultiscalePCA()
                pca_temp = mymodel.fit_transform(
                    features[i, :, :], wavelet_func="db4", threshold=0.3
                )
                # TODO - fix the fact that we are simply averaging across all channels
                pca_temp = pca_temp.mean(0)
                Wx_k, scales = cwt(pca_temp, "gmw")

                image = array_to_image(Wx_k)

                # save to a new array with (batch, features, time)
                if i == 0:
                    temporary_data = np.zeros(
                        (
                            features.shape[0],
                            Wx_k.shape[0],
                            Wx_k.shape[1],
                        ),
                        dtype=np.float32,
                    )
                    temporary_data[i, :, :] = image
                else:
                    temporary_data[i, :, :] = image

            features = temporary_data

            # Turn into torch tensor
            features = torch.tensor(features, device=self.device)

            preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            features = preprocess(features)
        return features, n_time_steps

    def train(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Starting training with {trainable:,} trainable parameters")

        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []

        val_steps_since_improvement = 0

        save_best_checkpoint = self.args.get("save_best_checkpoint", True)
        early_stopping = self.args.get("early_stopping", False)
        early_stopping_val_steps = self.args["early_stopping_val_steps"]

        train_start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            self.model.train()
            self.optimizer.zero_grad()

            start_time = time.time()

            features = batch["input_features"].to(self.device)
            labels = batch["seq_class_ids"].to(self.device)
            n_time_steps = batch["n_time_steps"].to(self.device)
            phone_seq_lens = batch["phone_seq_lens"].to(self.device)
            day_indicies = batch["day_indicies"].to(self.device)

            with torch.autocast(
                device_type="cuda", enabled=self.args["use_amp"], dtype=torch.bfloat16
            ):
                features, n_time_steps = self.transform_data(
                    features, n_time_steps, "train"
                )

                logits = self.model(features, day_indicies)
                B, S, V = logits.shape

                max_target_len = phone_seq_lens.max().item()
                if max_target_len > S:
                    pad_T = max_target_len - S
                    last_step = logits[:, -1:, :].expand(B, pad_T, V)
                    logits = torch.cat([logits, last_step], dim=1)
                    S = max_target_len

                input_lengths = torch.full(
                    (B,),
                    S,
                    dtype=torch.long,
                    device=logits.device,
                )

                per_ex_loss = self.ctc_loss(
                    log_probs=logits.log_softmax(2).permute(1, 0, 2),
                    targets=labels,
                    input_lengths=input_lengths,
                    target_lengths=phone_seq_lens,
                )

                finite_mask = torch.isfinite(per_ex_loss)
                if not finite_mask.any():
                    self.logger.warning(
                        f"All CTC losses non-finite in train batch {i}. Skipping this batch."
                    )
                    continue

                loss = per_ex_loss[finite_mask].mean()

            loss.backward()

            grad_norm = None
            if self.args["grad_norm_clip_value"] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.args["grad_norm_clip_value"],
                    error_if_nonfinite=True,
                    foreach=True,
                )

            self.optimizer.step()
            self.learning_rate_scheduler.step()

            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            if i % self.args["batches_per_train_log"] == 0:
                self.logger.info(
                    f"Train batch {i}: loss={loss.detach().item():.3f} "
                    f"grad_norm={grad_norm if grad_norm is not None else 0:.2f} "
                    f"time={train_step_duration:.3f}"
                )

            if i % self.args["batches_per_val_step"] == 0 or i == (
                self.args["num_training_batches"] - 1
            ):
                self.logger.info(f"Running validation after training batch {i}")
                start_time = time.time()
                val_metrics = self.validation(
                    loader=self.val_loader,
                    return_logits=self.args["save_val_logits"],
                    return_data=self.args["save_val_data"],
                )
                val_step_duration = time.time() - start_time

                self.logger.info(
                    f"Val batch {i}: PER={val_metrics['avg_PER']:.4f} "
                    f"CTC loss={val_metrics['avg_loss']:.4f} time={val_step_duration:.3f}"
                )

                if self.args["log_individual_day_val_PER"]:
                    for d in val_metrics["day_PERs"].keys():
                        self.logger.info(
                            f"{self.args['dataset']['sessions'][d]} val PER: "
                            f"{val_metrics['day_PERs'][d]['total_edit_distance'] / val_metrics['day_PERs'][d]['total_seq_length']:.4f}"
                        )

                val_PERs.append(val_metrics["avg_PER"])
                val_losses.append(val_metrics["avg_loss"])
                val_results.append(val_metrics)

                new_best = False
                if val_metrics["avg_PER"] < self.best_val_PER:
                    self.logger.info(
                        f"New best val PER {self.best_val_PER:.4f} → {val_metrics['avg_PER']:.4f}"
                    )
                    self.best_val_PER = val_metrics["avg_PER"]
                    self.best_val_loss = val_metrics["avg_loss"]
                    new_best = True
                elif (
                    val_metrics["avg_PER"] == self.best_val_PER
                    and val_metrics["avg_loss"] < self.best_val_loss
                ):
                    self.logger.info(
                        f"New best val loss {self.best_val_loss:.4f} → {val_metrics['avg_loss']:.4f}"
                    )
                    self.best_val_loss = val_metrics["avg_loss"]
                    new_best = True

                if new_best:
                    if save_best_checkpoint:
                        self.logger.info("Checkpointing best model")
                        self.save_model_checkpoint(
                            os.path.join(
                                self.args["checkpoint_dir"], "best_checkpoint"
                            ),
                            self.best_val_PER,
                            self.best_val_loss,
                        )

                    if self.args["save_val_metrics"]:
                        with open(
                            os.path.join(
                                self.args["checkpoint_dir"], "val_metrics.pkl"
                            ),
                            "wb",
                        ) as f:
                            pickle.dump(val_metrics, f)

                    val_steps_since_improvement = 0
                else:
                    val_steps_since_improvement += 1

                if self.args["save_all_val_steps"]:
                    ckpt_path = os.path.join(
                        self.args["checkpoint_dir"], f"checkpoint_batch_{i}"
                    )
                    self.save_model_checkpoint(
                        ckpt_path, val_metrics["avg_PER"], val_metrics["avg_loss"]
                    )

                if early_stopping and (
                    val_steps_since_improvement >= early_stopping_val_steps
                ):
                    self.logger.info(
                        f"Val PER has not improved in {early_stopping_val_steps} validation steps. "
                        f"Stopping early at batch {i}."
                    )
                    break

        training_duration = time.time() - train_start_time
        self.logger.info(f"Best avg val PER achieved: {self.best_val_PER:.5f}")
        self.logger.info(f"Total training time: {training_duration / 60:.2f} minutes")

        if self.args["save_final_model"]:
            final_ckpt = os.path.join(
                self.args["checkpoint_dir"], f"final_checkpoint_batch_{i}"
            )
            self.save_model_checkpoint(final_ckpt, val_PERs[-1], val_losses[-1])

        train_stats = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_PERs": val_PERs,
            "val_metrics": val_results,
        }
        return train_stats

    def validation(self, loader, return_logits=False, return_data=False):
        self.model.eval()
        metrics = {}

        if return_logits:
            metrics["logits"] = []
            metrics["n_time_steps"] = []

        if return_data:
            metrics["input_features"] = []

        metrics["decoded_seqs"] = []
        metrics["true_seq"] = []
        metrics["phone_seq_lens"] = []
        metrics["transcription"] = []
        metrics["losses"] = []
        metrics["block_nums"] = []
        metrics["trial_nums"] = []
        metrics["day_indicies"] = []

        total_edit_distance = 0
        total_seq_length = 0

        day_per = {}
        for d in range(len(self.args["dataset"]["sessions"])):
            if self.args["dataset"]["dataset_probability_val"][d] == 1:
                day_per[d] = {"total_edit_distance": 0, "total_seq_length": 0}

        for i, batch in enumerate(loader):
            features = batch["input_features"].to(self.device)
            labels = batch["seq_class_ids"].to(self.device)
            n_time_steps = batch["n_time_steps"].to(self.device)
            phone_seq_lens = batch["phone_seq_lens"].to(self.device)
            day_indicies = batch["day_indicies"].to(self.device)

            day = day_indicies[0].item()
            if self.args["dataset"]["dataset_probability_val"][day] == 0:
                continue

            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda",
                    enabled=self.args["use_amp"],
                    dtype=torch.bfloat16,
                ):
                    features, n_time_steps = self.transform_data(
                        features, n_time_steps, "val"
                    )

                    logits = self.model(features, day_indicies)
                    B, S, V = logits.shape

                    max_target_len = phone_seq_lens.max().item()
                    if max_target_len > S:
                        pad_T = max_target_len - S
                        last_step = logits[:, -1:, :].expand(B, pad_T, V)
                        logits = torch.cat([logits, last_step], dim=1)
                        S = max_target_len

                    input_lengths = torch.full(
                        (B,),
                        S,
                        dtype=torch.long,
                        device=logits.device,
                    )

                    per_ex_loss = self.ctc_loss(
                        torch.permute(logits.log_softmax(2), [1, 0, 2]),
                        labels,
                        input_lengths,
                        phone_seq_lens,
                    )

                    finite_mask = torch.isfinite(per_ex_loss)
                    if not finite_mask.any():
                        self.logger.warning(
                            f"All CTC losses non-finite in val batch {i}. Skipping this batch."
                        )
                        continue

                    loss = per_ex_loss[finite_mask].mean()

                metrics["losses"].append(loss.cpu().detach().numpy())

                batch_edit_distance = 0
                decoded_seqs = []
                for b in range(logits.shape[0]):
                    T = input_lengths[b].item()
                    decoded_seq = torch.argmax(logits[b, :T, :], dim=-1)
                    decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                    decoded_seq = decoded_seq.cpu().detach().numpy()
                    decoded_seq = np.array([idx for idx in decoded_seq if idx != 0])

                    true_seq = np.array(labels[b][0 : phone_seq_lens[b]].cpu().detach())
                    batch_edit_distance += taF.edit_distance(decoded_seq, true_seq)
                    decoded_seqs.append(decoded_seq)

            day_per[day]["total_edit_distance"] += batch_edit_distance
            day_per[day]["total_seq_length"] += torch.sum(phone_seq_lens).item()

            total_edit_distance += batch_edit_distance
            total_seq_length += torch.sum(phone_seq_lens)

            if return_logits:
                metrics["logits"].append(logits.cpu().float().numpy())
                metrics["n_time_steps"].append(input_lengths.cpu().numpy())

            if return_data:
                metrics["input_features"].append(batch["input_features"].cpu().numpy())

            metrics["decoded_seqs"].append(decoded_seqs)
            metrics["true_seq"].append(batch["seq_class_ids"].cpu().numpy())
            metrics["phone_seq_lens"].append(batch["phone_seq_lens"].cpu().numpy())
            metrics["transcription"].append(batch["transcriptions"].cpu().numpy())
            metrics["block_nums"].append(batch["block_nums"].numpy())
            metrics["trial_nums"].append(batch["trial_nums"].numpy())
            metrics["day_indicies"].append(batch["day_indicies"].cpu().numpy())

        avg_PER = (total_edit_distance / total_seq_length).item()

        metrics["day_PERs"] = day_per
        metrics["avg_PER"] = avg_PER
        metrics["avg_loss"] = (
            float(np.mean(metrics["losses"]))
            if len(metrics["losses"]) > 0
            else float("inf")
        )

        return metrics

    def evaluate_full_validation(self):
        val_metrics = self.validation(
            loader=self.val_loader,
            return_logits=False,
            return_data=False,
        )

        self.logger.info("\n=== Full validation results from current model ===")
        self.logger.info(f"Average CTC loss: {val_metrics['avg_loss']:.4f}")
        self.logger.info(f"Average PER:      {val_metrics['avg_PER']:.4f}")

        for d, stats in val_metrics["day_PERs"].items():
            if stats["total_seq_length"] > 0:
                day_per = stats["total_edit_distance"] / stats["total_seq_length"]
                day_name = self.args["dataset"]["sessions"][d]
                self.logger.info(f"  {day_name}: PER = {day_per:.4f}")

        return val_metrics
