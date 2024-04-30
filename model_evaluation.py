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

# Load the test data
directory_path_test = './mfcc_post_processing_test/'
csv_files_test = glob.glob(os.path.join(directory_path_test, file_pattern))
dataframes_test = []
for file in csv_files_test:
    df = pd.read_csv(file)
    dataframes_test.append(df)
pd_test = pd.concat(dataframes_test, ignore_index=True)
print(pd_test.head())
print(pd_test.shape)
#Split the test data into features and labels
features_test = pd_test.drop('Instruments', axis=1)
labels_test = pd_test['Instruments']
print(features_test.head())
print(labels_test.head())
#One-hot encode the labels
labels_encoded_test = encoder.fit_transform(labels_test.values.reshape(-1, 1))
print(labels_encoded_test.shape)
class_names = [str(name) for name in encoder.categories_[0]]