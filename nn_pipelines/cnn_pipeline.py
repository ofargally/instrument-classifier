import os
import torch
import glob
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Prepare the Data
def tensorConvert(value): #necessary since some of the labels have semicolons and some are ints or strings...
    if isinstance(value, str):
        parts = value.split(';')
        int_parts = [int(part) for part in parts]
        tensor = torch.tensor(int_parts, dtype=torch.long)
    else: 
        tensor = torch.tensor([value], dtype=torch.long)
    return tensor

def load_data(filename):
    data = pd.read_csv(filename)
    data['Instruments'] = data['Instruments'].apply(tensorConvert)
    features = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    #print(data['Instruments'])
    #print(type(data.iloc[:, -1].values[1])) #for some reason some of the instrument labels are strings while others are ints.
    print(filename)
    labels = data['Instruments'] #torch.tensor(data.iloc[:, -1].values, dtype=torch.long)
    return features, labels

def get_all_files(directory):
    files = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            files.append(f)
    return files
# Assuming you have a list of CSV file paths
directory_path_train = './mfcc_post_processing/train'
file_pattern = "*.csv"
csv_files_train = glob.glob(os.path.join(directory_path_train, file_pattern))
dataframes_train = []

for file in csv_files_train:
    df = pd.read_csv(file)
    dataframes_train.append(df)
pd_train = pd.concat(dataframes_train, ignore_index=True)

# Process labels for multi-label classification
def process_labels(value):
    value = str(value)
    return [int(v) for v in value.split(';')]
    
labels = pd_train['Instruments'].apply(process_labels)
mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(labels)

# Prepare features
features = pd_train.drop('Instruments', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)

# Convert to tensors
train_features = torch.tensor(X_train).float()
train_labels = torch.tensor(y_train).float()
val_features = torch.tensor(X_val).float()
val_labels = torch.tensor(y_val).float()

train_data = TensorDataset(train_features, train_labels)
val_data = TensorDataset(val_features, val_labels)
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

#print(data[1])

# Pad or truncate sequences to a fixed length
# You may need to adjust this based on your data
# For example, you can use torch.nn.utils.rnn.pad_sequence

# Step 2: Define the CNN Model
class CNN(nn.Module): ##THIS NEEDS TO BE HEAVILY EDITTED IDK WHAT IM DOING >< 
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN, self).__init__()
        # Define your CNN architecture here
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(64, hidden_size, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(9, 12)  
    def forward(self, x):
        # Define the forward pass of your CNN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x

# Step 3: Training Loop
model = CNN(input_size=train_features.shape[1], hidden_size=32, num_layers=2, num_classes=train_labels.shape[1]).to(torch.device("cpu")) 
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {loss.item():.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}, "
          f"Val Accuracy: {(correct / total) * 100:.2f}%")