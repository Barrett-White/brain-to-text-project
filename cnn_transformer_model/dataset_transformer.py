# data_augmentations + dataset + train/val split

import math
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def gauss_smooth(
    inputs, device, smooth_kernel_std=2, smooth_kernel_size=100, padding="same"
):
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)

    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)
    gaussKernel = gaussKernel.repeat(C, 1, 1)

    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
    return smoothed.permute(0, 2, 1)


class BrainToTextDataset(Dataset):
    def __init__(
        self,
        trial_indicies,
        n_batches,
        split="train",
        batch_size=64,
        days_per_batch=1,
        random_seed=-1,
        must_include_days=None,
        feature_subset=None,
    ):
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        if split not in ["train", "test"]:
            raise ValueError(f'split must be "train" or "test", got {split}')

        self.split = split
        self.days_per_batch = days_per_batch
        self.batch_size = batch_size
        self.n_batches = n_batches

        self.trial_indicies = trial_indicies
        self.n_days = len(trial_indicies.keys())
        self.feature_subset = feature_subset
        self.n_trials = sum(len(trial_indicies[d]["trials"]) for d in trial_indicies)

        if must_include_days is not None and len(must_include_days) > days_per_batch:
            raise ValueError("must_include_days must be <= days_per_batch")

        if (
            must_include_days is not None
            and len(must_include_days) > self.n_days
            and split != "train"
        ):
            raise ValueError("must_include_days not valid for test data")

        if must_include_days is not None:
            for i, d in enumerate(must_include_days):
                if d < 0:
                    must_include_days[i] = self.n_days + d

        self.must_include_days = must_include_days

        if self.split == "train" and self.days_per_batch > self.n_days:
            raise ValueError(f"days_per_batch {days_per_batch} > n_days {self.n_days}")

        if self.split == "train":
            self.batch_index = self.create_batch_index_train()
        else:
            self.batch_index = self.create_batch_index_test()
            self.n_batches = len(self.batch_index.keys())

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        batch = {
            "input_features": [],
            "seq_class_ids": [],
            "n_time_steps": [],
            "phone_seq_lens": [],
            "day_indicies": [],
            "transcriptions": [],
            "block_nums": [],
            "trial_nums": [],
        }

        index = self.batch_index[idx]

        for d in index.keys():
            with h5py.File(self.trial_indicies[d]["session_path"], "r") as f:
                for t in index[d]:
                    try:
                        g = f[f"trial_{t:04d}"]

                        x = torch.from_numpy(g["input_features"][:])
                        if self.feature_subset:
                            x = x[:, self.feature_subset]

                        batch["input_features"].append(x)
                        batch["seq_class_ids"].append(
                            torch.from_numpy(g["seq_class_ids"][:])
                        )
                        batch["transcriptions"].append(
                            torch.from_numpy(g["transcription"][:])
                        )
                        batch["n_time_steps"].append(g.attrs["n_time_steps"])
                        batch["phone_seq_lens"].append(g.attrs["seq_len"])
                        batch["day_indicies"].append(int(d))
                        batch["block_nums"].append(g.attrs["block_num"])
                        batch["trial_nums"].append(g.attrs["trial_num"])
                    except Exception as e:
                        print(
                            f"Error loading trial {t} from {self.trial_indicies[d]['session_path']}: {e}"
                        )
                        continue

        batch["input_features"] = pad_sequence(
            batch["input_features"], batch_first=True, padding_value=0
        )
        batch["seq_class_ids"] = pad_sequence(
            batch["seq_class_ids"], batch_first=True, padding_value=0
        )

        batch["n_time_steps"] = torch.tensor(batch["n_time_steps"])
        batch["phone_seq_lens"] = torch.tensor(batch["phone_seq_lens"])
        batch["day_indicies"] = torch.tensor(batch["day_indicies"])
        batch["transcriptions"] = torch.stack(batch["transcriptions"])
        batch["block_nums"] = torch.tensor(batch["block_nums"])
        batch["trial_nums"] = torch.tensor(batch["trial_nums"])

        return batch

    def create_batch_index_train(self):
        batch_index = {}

        if self.must_include_days is not None:
            non_must_include_days = [
                d for d in self.trial_indicies.keys() if d not in self.must_include_days
            ]

        for batch_idx in range(self.n_batches):
            batch = {}

            if self.must_include_days is not None and len(self.must_include_days) > 0:
                days = np.concatenate(
                    (
                        self.must_include_days,
                        np.random.choice(
                            non_must_include_days,
                            size=self.days_per_batch - len(self.must_include_days),
                            replace=False,
                        ),
                    )
                )
            else:
                days = np.random.choice(
                    list(self.trial_indicies.keys()),
                    size=self.days_per_batch,
                    replace=False,
                )

            num_trials = math.ceil(self.batch_size / self.days_per_batch)

            for d in days:
                trial_idxs = np.random.choice(
                    self.trial_indicies[d]["trials"], size=num_trials, replace=True
                )
                batch[d] = trial_idxs

            extra_trials = (num_trials * len(days)) - self.batch_size
            while extra_trials > 0:
                d = np.random.choice(days)
                batch[d] = batch[d][:-1]
                extra_trials -= 1

            batch_index[batch_idx] = batch

        return batch_index

    def create_batch_index_test(self):
        batch_index = {}
        batch_idx = 0

        for d in self.trial_indicies.keys():
            num_trials = len(self.trial_indicies[d]["trials"])
            num_batches = (num_trials + self.batch_size - 1) // self.batch_size

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_trials)
                batch_trials = self.trial_indicies[d]["trials"][start_idx:end_idx]
                batch_index[batch_idx] = {d: batch_trials}
                batch_idx += 1

        return batch_index


def train_test_split_indicies(
    file_paths, test_percentage=0.1, seed=-1, bad_trials_dict=None
):
    if seed != -1:
        np.random.seed(seed)

    trials_per_day = {}
    for i, path in enumerate(file_paths):
        session = [
            s
            for s in path.split("/")
            if (s.startswith("t15.20") or s.startswith("t12.20"))
        ][0]
        good_trial_indices = []

        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                num_trials = len(list(f.keys()))
                for t in range(num_trials):
                    key = f"trial_{t:04d}"
                    block_num = f[key].attrs["block_num"]
                    trial_num = f[key].attrs["trial_num"]

                    if (
                        bad_trials_dict is not None
                        and session in bad_trials_dict
                        and str(block_num) in bad_trials_dict[session]
                        and trial_num in bad_trials_dict[session][str(block_num)]
                    ):
                        continue

                    good_trial_indices.append(t)

        trials_per_day[i] = {
            "num_trials": len(good_trial_indices),
            "trial_indices": good_trial_indices,
            "session_path": path,
        }

    train_trials, test_trials = {}, {}

    for day in trials_per_day.keys():
        num_trials = trials_per_day[day]["num_trials"]
        all_trial_indices = trials_per_day[day]["trial_indices"]
        session_path = trials_per_day[day]["session_path"]

        if test_percentage == 0:
            train_trials[day] = {
                "trials": all_trial_indices,
                "session_path": session_path,
            }
            test_trials[day] = {"trials": [], "session_path": session_path}
            continue
        if test_percentage == 1:
            train_trials[day] = {"trials": [], "session_path": session_path}
            test_trials[day] = {
                "trials": all_trial_indices,
                "session_path": session_path,
            }
            continue

        num_test = max(1, int(num_trials * test_percentage))
        test_indices = np.random.choice(
            all_trial_indices, size=num_test, replace=False
        ).tolist()
        train_indices = [idx for idx in all_trial_indices if idx not in test_indices]

        train_trials[day] = {"trials": train_indices, "session_path": session_path}
        test_trials[day] = {"trials": test_indices, "session_path": session_path}

    return train_trials, test_trials
