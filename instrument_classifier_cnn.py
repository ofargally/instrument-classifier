import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Dataset Class
class MFCCDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        self.labels = []
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()

        for filename in os.listdir(directory):
            df = pd.read_csv(os.path.join(directory, filename), names=['Coefficients', 'Instruments'], skiprows=1)
            df = df.dropna()  # Drop rows with missing labels
            if not df.empty:
                instruments = df['Instruments'].apply(lambda x: tuple(map(int, x.split(';')))).tolist()
                coefficients = df['Coefficients'].tolist()
               
                # Normalize MFCC coefficients
                coefficients = np.array(coefficients).reshape(-1, 1)
                coefficients = self.scaler.fit_transform(coefficients).flatten()

                self.data.append(coefficients)
                self.labels.append(instruments)
       
        # Fit MultiLabelBinarizer to all available labels
        self.labels = self.mlb.fit_transform(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Pad or truncate the sequence to a fixed size, e.g., 500
        max_length = 500
        feature = np.zeros(max_length)
        length = min(max_length, len(self.data[idx]))
        feature[:length] = self.data[idx][:length]
        return torch.tensor(feature, dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.float32)

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * 250, 128)  # Adjust size according to your padding/stride
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Setup
directory = '/path/to/mfcc_post_processing'
dataset = MFCCDataset(directory)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
num_classes = len(dataset.mlb.classes_)
model = CNN(num_classes)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model():
    model.train()
    for data, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('Training complete')

train_model()