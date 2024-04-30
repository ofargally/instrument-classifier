import glob
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def process_labels(value):
    value = str(value)
    return [int(v) for v in value.split(';')]

# Load the test data
directory_path_test = './mfcc_post_processing_test/'
csv_files_test = glob.glob(os.path.join(directory_path_test, '*.csv'))
dataframes_test = []
for file in csv_files_test:
    df = pd.read_csv(file)
    dataframes_test.append(df)

# Encode features and labels
pd_test = pd.concat(dataframes_test, ignore_index=True)
labels = pd_test['Instruments'].apply(process_labels)
mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(labels)

features = pd_test.drop('Instruments', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)