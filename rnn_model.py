import os
import pandas as pd
import numpy as np
import torch
from torch import nn



csv_folder = './mfcc_post_processing'
mfccNames = [os.path.join(csv_folder, file) for file in os.listdir(csv_folder) if file.endswith('.csv')]

class InstrumentClassifierRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(InstrumentClassifierRNN, self).__init__()
        
        # Define LSTM layer(s)
        self.lstm = nn.LSTM(
            input_size=input_size,  # Number of MFCC features
            hidden_size=hidden_size,  # Size of the hidden state
            num_layers=num_layers,  # Number of layers in the LSTM
            batch_first=True,  # Use batch-first format (batch_size, seq_length, input_size)
            dropout=dropout  # Dropout for regularization (0 means no dropout)
        )
        
        # Fully connected (linear) layer for classification
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Forward pass through the LSTM layer
        # Returns the final hidden state (hn) from the LSTM
        _, (hn, _) = self.lstm(x)
        
        # Use the last hidden state from the last layer as the final representation
        # hn shape: (num_layers, batch_size, hidden_size)
        # We take hn[-1], which is the last layer's hidden state
        final_representation = hn[-1]  # Shape: (batch_size, hidden_size)
        
        # Forward pass through the fully connected layer
        # Output shape: (batch_size, output_size)
        out = self.fc(final_representation)
        
        return out
