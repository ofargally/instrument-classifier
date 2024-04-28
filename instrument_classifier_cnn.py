import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os
import pandas as pd

# Path to the directory containing CSV files
directory = './mfcc_post_processing'

# Function to get sequence length from a CSV file
def get_sequence_length(filename):
    df = pd.read_csv(filename)
    return len(df)

# Iterate through each CSV file in the directory
def get_max_sequence_length(directory):
    max_seq_len = 0
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            seq_len = get_sequence_length(file_path)
            max_seq_len = max(max_seq_len, seq_len)
    return max_seq_len

class Net(nn.Module):
    def __init__(self, max_seq_len):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * (max_seq_len // 2), 128)
        self.fc2 = nn.Linear(128, 12)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * (self.max_seq_len // 2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
get_max_sequence_length('./mfcc_post_processing')
