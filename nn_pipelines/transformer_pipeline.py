import glob
import os
import torch
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import joblib  # For saving the scaler
random.seed(666)
# Load CSV files and concatenate into a DataFrame
#directory_path_train = './mfcc_training_final' #for tturing
directory_path_train = './mfcc_post_processing'
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
print(pd_train.head())
print(pd_train.shape)

# Split the training data into features and labels
features = pd_train.drop('Instruments', axis=1)
labels = pd_train['Instruments'].astype(str)

# Prepare the data scaler and encode labels
scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=False)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train.values)

# Transform the validation data
X_val_scaled = scaler.transform(X_val.values)

# Save the scaler to disk for later use
joblib.dump(scaler, 'scaler.joblib')

# One-hot encode the labels
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_val_encoded = encoder.transform(y_val.values.reshape(-1, 1))

# Convert to PyTorch tensors
train_features = torch.tensor(X_train_scaled).float()
train_labels = torch.tensor(y_train_encoded).float()
val_features = torch.tensor(X_val_scaled).float()
val_labels = torch.tensor(y_val_encoded).float()
print("passed the train validation pytorch tensor creation")
# DataLoaders
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("passed the dataloader stuff")
# Define the model
class InstrumentClassifier(nn.Module):
    def __init__(self, num_features, num_classes, dim_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(InstrumentClassifier, self).__init__()
        self.embedding = nn.Linear(num_features, dim_model)
        encoder_layers = TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(dim_model, num_classes)

    def forward(self, src):
        embedded = self.embedding(src)
        transformer_output = self.transformer_encoder(embedded)
        output = self.output_layer(transformer_output[:, 0, :] if transformer_output.dim() == 3 else transformer_output)
        return output

# Initialize and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InstrumentClassifier(num_features=train_features.shape[1], num_classes=train_labels.shape[1]).to(device)
print("passed the model instance creation")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("passed the optimizier")
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
num_epochs = 30
for epoch in range(num_epochs):
    print("epoch number: " + str(epoch))
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model and scaler
torch.save(model.state_dict(), './transformer_v1_state.pth')
torch.save(model, './transformer_v1.pth')

print('Finished Training')
