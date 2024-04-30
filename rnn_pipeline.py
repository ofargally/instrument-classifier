import glob
import os
import torch
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

# Load CSV files and concatenate into a DataFrame
directory_path_train = './mfcc_training_final'
file_pattern = "*.csv"
csv_files_train = glob.glob(os.path.join(directory_path_train, file_pattern))
sample_percentage = 1
num_files_to_sample = int(len(csv_files_train) * (sample_percentage / 100.0))
csv_files_train = random.sample(csv_files_train, num_files_to_sample)
dataframes_train = []
for file in csv_files_train:
    df = pd.read_csv(file)
    dataframes_train.append(df)
pd_train = pd.concat(dataframes_train, ignore_index=True)

# Process labels for multi-label classification
def process_labels(value):
    return list(map(int, value.split(';')))

labels = pd_train['Instruments'].apply(process_labels)
mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(labels)

# Prepare features
features = pd_train.drop('Instruments', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)

# Convert to tensors
train_features = torch.tensor(X_train).float()
train_labels = torch.tensor(y_train).float()
val_features = torch.tensor(X_val).float()
val_labels = torch.tensor(y_val).float()

# DataLoader setup
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# RNN Model
class RNNInstrumentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNInstrumentClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
    def forward(self, x):
        # Initialize hidden state and cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNInstrumentClassifier(input_size=train_features.shape[1], hidden_size=100, num_layers=2, num_classes=train_labels.shape[1])
model.to(device)

# Training setup
criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

# Validation function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

# Training loop
for epoch in range(30):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/30 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model and scaler
torch.save(model.state_dict(), './rnn_model_state.pth')
torch.save(model, './rnn_model.pth')

print('Finished Training')
