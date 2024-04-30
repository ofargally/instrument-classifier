import os
import random
import shutil
from typing import List, Tuple

def get_files(directory: str, extension: str) -> List[str]:
    """Retrieve files with the specified extension from the given directory."""
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]

def split_data(files: List[str], subset_percent: float, train_split: float) -> Tuple[List[str], List[str]]:
    """First subset data and then split into training and validation sets."""
    random.shuffle(files)
    subset_index = int(len(files) * subset_percent)
    subset_files = files[:subset_index]
    train_index = int(len(subset_files) * train_split)
    return subset_files[:train_index], subset_files[train_index:]

def copy_files(file_paths: List[str], target_dir: str):
    """Copy files to a new directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file_path in file_paths:
        shutil.copy(file_path, target_dir)

def create_subsets(csv_dir: str, subset_percent: float, train_split: float):
    """Create and copy subsets of CSV files for training and validation."""
    csv_files = get_files(csv_dir, '.csv')
    train_set, validation_set = split_data(csv_files, subset_percent, train_split)
    train_dir = os.path.join(csv_dir, 'train')
    validation_dir = os.path.join(csv_dir, 'validation')
    copy_files(train_set, train_dir)
    copy_files(validation_set, validation_dir)
    print("Total files in CSV directory:", len(csv_files))
    print("Files in training directory:", len(train_set))
    print("Files in validation directory:", len(validation_set))
    return train_set, validation_set

# Example usage
csv_directory = input("Enter the path to the directory with CSV files: ")
subset_percentage = float(input("Enter the subset percentage (e.g., 50 for 50%): ")) / 100
train_percentage = float(input("Enter the training set percentage within the subset (e.g., 90 for 90%): ")) / 100

train_files, validation_files = create_subsets(csv_directory, subset_percentage, train_percentage)