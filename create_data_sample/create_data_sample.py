import random
from pathlib import Path

import h5py


def load_h5py_file(file_path: str) -> dict:
    """Use offical website code to read .hdf5 files from Brain-to-Text.


    Args:
        file_path (str): Path from which to read the files

    Returns:
        dict: hdf5 dictionary
    """
    data = {
        "neural_features": [],
        "n_time_steps": [],
        "seq_class_ids": [],
        "seq_len": [],
        "transcriptions": [],
        "sentence_label": [],
        "session": [],
        "block_num": [],
        "trial_num": [],
    }
    # open the hdf5 file for that day
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())

        # each trail in the day
        for key in keys:
            g = f[key]

            neural_features = g["input_features"][:]
            n_time_steps = g.attrs["n_time_steps"]
            seq_class_ids = g["seq_class_ids"][:] if "seq_class_ids" in g else None
            seq_len = g.attrs["seq_len"] if "seq_len" in g.attrs else None
            transcription = g["transcription"][:] if "transcription" in g else None
            sentence_label = (
                g.attrs["sentence_label"][:] if "sentence_label" in g.attrs else None
            )
            session = g.attrs["session"]
            block_num = g.attrs["block_num"]
            trial_num = g.attrs["trial_num"]

            data["neural_features"].append(neural_features)
            data["n_time_steps"].append(n_time_steps)
            data["seq_class_ids"].append(seq_class_ids)
            data["seq_len"].append(seq_len)
            data["transcriptions"].append(transcription)
            data["sentence_label"].append(sentence_label)
            data["session"].append(session)
            data["block_num"].append(block_num)
            data["trial_num"].append(trial_num)
    return data

def load_data_by_day(base_directory, percent_of_days_to_read=100, specific_days=None):
    base_path = Path(base_directory)
    all_day_folders = [d for d in base_path.iterdir() if d.is_dir()]

    if not all_day_folders:
        print(f"No subdirectories found in '{base_path.resolve()}'")
        return None

    if specific_days is not None:
        if isinstance(specific_days, str):
            specific_days = [specific_days]
        folders_to_process = [d for d in all_day_folders if d.name in specific_days]
        print(f"Targeting {len(folders_to_process)} specific day(s): {specific_days}")
    else:
        if percent_of_days_to_read < 100:
            num_folders = int(len(all_day_folders) * (percent_of_days_to_read / 100.0))
            print(
                f"Found {len(all_day_folders)} total days. Randomly sampling {num_folders}"
            )
            folders_to_process = random.sample(all_day_folders, num_folders)
        else:
            print(f"Found {len(all_day_folders)} total days. Processing all...")
            folders_to_process = all_day_folders

    files_by_type = {"train": [], "val": [], "test": []}
    for folder in folders_to_process:
        for f in folder.glob("*.hdf5"):
            if "train" in f.name:
                files_by_type["train"].append(f)
            elif "val" in f.name:
                files_by_type["val"].append(f)
            elif "test" in f.name:
                files_by_type["test"].append(f)

    final_datasets = {}
    for data_type, file_list in files_by_type.items():
        if not file_list:
            print(f"No '{data_type}' files found in the selected days.")
            final_datasets[data_type] = None
            continue

        print(f"Processing {len(file_list)} '{data_type}' files")

        aggregated_data = {
            "neural_features": [],
            "n_time_steps": [],
            "seq_class_ids": [],
            "seq_len": [],
            "transcriptions": [],
            "sentence_label": [],
            "session": [],
            "block_num": [],
            "trial_num": [],
        }

        for file_path in file_list:
            data_from_file = load_h5py_file(file_path)
            for key in aggregated_data.keys():
                aggregated_data[key].extend(data_from_file[key])

        final_datasets[data_type] = aggregated_data

    print("\n Finished loading!")
    return final_datasets

def load_data_by_day_or_perc(
    base_directory: str,
    percent_of_days_to_read: int | None = 100,
    specific_days: str | None | list = None,
    seed: int | None = 1,
) -> dict:
    """Load either a percentage of days or specific day or days from some base directory with hdf5 files.

    Note: If specifying specific days, the percent of days to read will be ignored.

    Args:
        base_directory (str): Place from which to load all the data
        percent_to_read (int, optional): Percentage of data to read in. Defaults to 100.
        specific_days (str | None | list, optional): Name or list of names of specific days to use. Defaults to None.
        seed (int | None, optional): Random seed to use, if desired. Defaults to 1.

    Returns:
        dict: Data meeting the desired input criteria
    """
    if seed is not None:
        random.seed(seed)
    base_path = Path(base_directory)
    all_day_folders = [d for d in base_path.iterdir() if d.is_dir()]

    if not all_day_folders:
        print(f"No subdirectories found in '{base_path.resolve()}'")
        return None

    if specific_days is not None:
        if isinstance(specific_days, str):
            specific_days = [specific_days]
        folders_to_process = [d for d in all_day_folders if d.name in specific_days]
        print(f"Targeting {len(folders_to_process)} specific day(s): {specific_days}")
    else:
        if percent_of_days_to_read < 100:
            num_folders = int(len(all_day_folders) * (percent_of_days_to_read / 100.0))
            print(
                f"Found {len(all_day_folders)} total days. Randomly sampling {num_folders}"
            )
            folders_to_process = random.sample(all_day_folders, num_folders)
        else:
            print(f"Found {len(all_day_folders)} total days. Processing all...")
            folders_to_process = all_day_folders

    files_by_type = {"train": [], "val": [], "test": []}
    for folder in folders_to_process:
        for f in folder.glob("*.hdf5"):
            if "train" in f.name:
                files_by_type["train"].append(f)
            elif "val" in f.name:
                files_by_type["val"].append(f)
            elif "test" in f.name:
                files_by_type["test"].append(f)

    final_datasets = {}
    for data_type, file_list in files_by_type.items():
        if not file_list:
            print(f"No '{data_type}' files found in the selected days.")
            final_datasets[data_type] = None
            continue

        print(f"Processing {len(file_list)} '{data_type}' files")

        aggregated_data = {
            "neural_features": [],
            "n_time_steps": [],
            "seq_class_ids": [],
            "seq_len": [],
            "transcriptions": [],
            "sentence_label": [],
            "session": [],
            "block_num": [],
            "trial_num": [],
        }

        for file_path in file_list:
            data_from_file = load_h5py_file(file_path)
            for key in aggregated_data.keys():
                aggregated_data[key].extend(data_from_file[key])

        final_datasets[data_type] = aggregated_data

    print("\n Finished loading!")
    return final_datasets


if __name__ == "__main__":
    import os

    import pandas as pd

    seed = 8

    base_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..\data\hdf5_data_final"
    )
    data_set = load_data_by_day_or_perc(
        base_directory, percent_of_days_to_read=20, seed=seed
    )
    train_data = pd.DataFrame(data_set["train"])
    test_data = pd.DataFrame(data_set["test"])
    val_data = pd.DataFrame(data_set["val"])

    # Now, we can save all of these in our own data subfile
    new_dirs = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..\data", "sampled_dataset"
    )
    os.makedirs(
        new_dirs,
        exist_ok=True,
    )

    train_data.to_csv(os.path.join(new_dirs, "train_data.csv"))
    test_data.to_csv(os.path.join(new_dirs, "test_data.csv"))
    val_data.to_csv(os.path.join(new_dirs, "val_data.csv"))
