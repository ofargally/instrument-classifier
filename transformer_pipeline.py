import glob
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
#Combine all csv files into one dataframe
directory_path_train = './mfcc_post_processing/' #MODIFY THIS PATH
file_pattern = "*.csv"
csv_files_train = glob.glob(os.path.join(directory_path_train, file_pattern))
dataframes_train = []
for file in csv_files_train:
    df = pd.read_csv(file)
    dataframes_train.append(df)
pd_train = pd.concat(dataframes_train, ignore_index=True)
print(pd_train.head()) 
print(pd_train.shape)
# Split the training data into features and labels
features = pd_train.drop('Instruments', axis=1)
labels = pd_train['Instruments']
print(features.head())
print(labels.head())
# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
labels_encoded = encoder.fit_transform(labels.values.reshape(-1, 1))
print(labels_encoded.shape)
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
# Convert to PyTorch tensors
train_features = torch.tensor(X_train.values).float()
train_labels = torch.tensor(y_train).float()
val_features = torch.tensor(X_val.values).float()
val_labels = torch.tensor(y_val).float()

# Create TensorDatasets
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)
# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Define the model
class InstrumentClassifier(nn.Module):
    def __init__(self, num_features, num_classes, dim_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(InstrumentClassifier, self).__init__()
        self.embedding = nn.Linear(num_features, dim_model)
        encoder_layers = TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(dim_model, num_classes)

    def forward(self, src):
        embedded = self.embedding(src)
        transformer_output = self.transformer_encoder(embedded)
        if transformer_output.dim() == 3:
            output = self.output_layer(transformer_output[:, 0, :])  # Correct if seq_length dimension exists
        else:
            output = self.output_layer(transformer_output)  # Use directly if no seq_length dimension
        return output

# Create a model instance
model = InstrumentClassifier(num_features=train_features.shape[1], num_classes=labels_encoded.shape[1])
# Define the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# Define the training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training and validation phases
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward and optimize
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Number of epochs
num_epochs = 30

# Loop over the dataset multiple times
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print('Finished Training')

torch.save(model.state_dict(), './transformer_v1_state.pth')
torch.save(model, './transformer_v1.pth')

