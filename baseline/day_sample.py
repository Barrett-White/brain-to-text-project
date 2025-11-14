import os
import random
from pathlib import Path

SEED = 8
PERCENT_TO_READ = 20
# Get the 'baseline' folder path
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up to the main project root
project_root = os.path.dirname(script_dir)

# Point to the data folder from the root
BASE_DIRECTORY = os.path.join(project_root, "data", "hdf5_data_final")


def get_sampled_day_names(base_dir, percent, seed):
    """
    Runs the exact sampling logic from your 'load_data_by_day_or_perc'
    function to get a reproducible list of day folder names.
    """

    print(f"Finding sampled days using Seed={seed} and Percent={percent}%...")

    # 1. Set the seed (This ensures the sample is reproducible)
    random.seed(seed)

    # 2. Find all day folders
    base_path = Path(base_dir)
    all_day_folders = [d for d in base_path.iterdir() if d.is_dir()]

    if not all_day_folders:
        print(f"Error: No day folders found in {base_dir}")
        return []

    # 3. CRITICAL STEP: Sort the list before sampling.
    # This guarantees that 'random.sample' sees the list in the
    # same order on every machine, making the sample reproducible.
    all_day_folders.sort()

    # 4. Calculate number of folders to sample
    num_folders = int(len(all_day_folders) * (percent / 100.0))
    num_folders = max(1, num_folders)  # Ensure at least 1 folder is sampled

    print(f"Found {len(all_day_folders)} total days. Sampling {num_folders}...")

    # 5. Run the sampling
    sampled_folders = random.sample(all_day_folders, num_folders)

    # 6. Get just the names from the Path objects
    sampled_names = [folder.name for folder in sampled_folders]

    return sampled_names


if __name__ == "__main__":
    sampled_day_list = get_sampled_day_names(BASE_DIRECTORY, PERCENT_TO_READ, SEED)

    if sampled_day_list:
        print(f"\n--- Found {len(sampled_day_list)} sampled days ---")
        print("--- Copy and paste this into your rnn_args.yaml 'sessions:' list ---")

        # Sort for easy reading in the YAML file
        for name in sorted(sampled_day_list):
            print(f"  - {name}")
