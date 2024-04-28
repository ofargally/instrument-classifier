import os
import random
import shutil
from typing import List, Tuple

def get_files(directory: str, extension: str) -> List[str]:
    """Retrieve files with the specified extension from the given directory."""
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]

def match_files(wav_files: List[str], csv_files: List[str]) -> List[Tuple[str, str]]:
    """Match WAV files with their corresponding CSV files."""
    base_names = {os.path.splitext(os.path.basename(wav))[0]: wav for wav in wav_files}
    matched_files = [(base_names[os.path.splitext(os.path.basename(csv))[0]], csv) 
                     for csv in csv_files if os.path.splitext(os.path.basename(csv))[0] in base_names]
    return matched_files

def split_data(files: List[Tuple[str, str]], subset_percent: float, train_split: float) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """First subset data and then split into training and validation sets."""
    random.shuffle(files)
    subset_index = int(len(files) * subset_percent)
    subset_files = files[:subset_index]
    train_index = int(len(subset_files) * train_split)
    return subset_files[:train_index], subset_files[train_index:]

def copy_files(file_pairs: List[Tuple[str, str]], target_dir: str):
    """Copy files to a new directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for wav, csv in file_pairs:
        shutil.copy(wav, target_dir)
        shutil.copy(csv, target_dir)

def create_subsets(audio_dir: str, label_dir: str, subset_percent: float, train_split: float):
    """Create and copy subsets of WAV and CSV files for training and validation."""
    wav_files = get_files(audio_dir, '.wav')
    csv_files = get_files(label_dir, '.csv')
    matched_files = match_files(wav_files, csv_files)
    train_set, validation_set = split_data(matched_files, subset_percent, train_split)
    copy_files(train_set, os.path.join(audio_dir, 'train'))
    copy_files(validation_set, os.path.join(audio_dir, 'validation'))
    return train_set, validation_set

# User inputs
audio_directory = input("Enter the path to the directory with WAV files: ")
label_directory = input("Enter the path to the directory with CSV files: ")
subset_percentage = float(input("Enter the subset percentage (e.g., 50 for 50%): ")) / 100
train_percentage = float(input("Enter the training set percentage within the subset (e.g., 90 for 90%): ")) / 100

train_files, validation_files = create_subsets(audio_directory, label_directory, subset_percentage, train_percentage)
print("Training Files:", train_files)
print("Validation Files:", validation_files)
