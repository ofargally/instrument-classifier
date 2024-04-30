import optuna
import torch
import os
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pandas as pd
import numpy as np
import glob
import joblib
import json
from torch.nn import TransformerEncoderLayer, TransformerEncoder

# Define the model with dynamic hyperparameters
class InstrumentClassifier(nn.Module):
    def __init__(self, num_features, num_classes, dim_model, nhead, num_layers, dropout):
        super(InstrumentClassifier, self).__init__()
        self.embedding = nn.Linear(num_features, dim_model)
        encoder_layers = TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(dim_model, num_classes)

    def forward(self, src):
        embedded = self.embedding(src)
        transformer_output = self.transformer_encoder(embedded)
        output = self.output_layer(transformer_output[:, 0, :] if transformer_output.dim() == 3 else transformer_output)
        return output

def objective(trial):
    # Load and prepare the data
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
    # Function to process labels into a list of integers
    def process_labels(value):
        value = str(value)
        return [int(v) for v in value.split(';')]
    
    features = pd_train.drop('Instruments', axis=1)
    labels = pd_train['Instruments'].apply(process_labels)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)  # Apply MultiLabelBinarizer

    # Hyperparameters to tune
    dim_model = trial.suggest_categorical('dim_model', [64, 128, 256])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)  # Updated to use suggest_float with log=True
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Data scaling
    scaler = StandardScaler()
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    train_features = torch.tensor(X_train_scaled).float()
    train_labels = torch.tensor(y_train).float()
    val_features = torch.tensor(X_val_scaled).float()
    val_labels = torch.tensor(y_val).float()

    # Dataloaders
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InstrumentClassifier(train_features.shape[1], train_labels.shape[1], dim_model, nhead, num_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    num_epochs = 10  # Adjusted for example
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)

    return avg_loss  # Objective: minimize average loss

# Running the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Adjust the number of trials based on time/resources

print("Best hyperparameters: ", study.best_params)

def save_hyperparameters(best_params, directory_path, filename='best_hyperparameters_transformer.json'):
    file_path = os.path.join(directory_path, filename)  # Full path to the file
    with open(file_path, 'w') as file:
        json.dump(best_params, file, indent=4)  # Save JSON with pretty print
    print(f"Hyperparameters saved to {file_path}")

save_hyperparameters(study.best_params, './finetuning_nn')
