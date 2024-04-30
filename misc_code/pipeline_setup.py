import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import os

def load_data(directory):
    """ Load all CSV files into a single DataFrame """
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    df_list = []
    for file in files:
        df = pd.read_csv(os.path.join(directory, file), names=['Coefficients', 'Instruments'], skiprows=1)
        df['Instruments'] = df['Instruments'].apply(lambda x: tuple(map(int, x.split(';'))) if isinstance(x, str) else ())
        df_list.append(df)
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def preprocess_data(df):
    """ Normalize MFCC features and encode labels """
    scaler = StandardScaler()
    df['Coefficients'] = scaler.fit_transform(df[['Coefficients']])

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['Instruments'].values)
    return df['Coefficients'].values, labels, mlb.classes_

def create_datasets(features, labels, batch_size=32):
    """ Convert arrays to tensor and create DataLoader """
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Load and preprocess data
directory = './mfcc_post_processing'
dataframe = load_data(directory)
features, labels, classes = preprocess_data(dataframe)

# Create DataLoader
data_loader = create_datasets(features, labels)

print(data_loader.dataset.tensors[0].shape)
print(data_loader.dataset.tensors[0])
print(data_loader.dataset)