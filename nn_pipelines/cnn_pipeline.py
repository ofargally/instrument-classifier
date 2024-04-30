import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Prepare the Data
def tensorConvert(value):
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
csv_files = get_all_files('./mfcc_post_processing')

data = [load_data(filename) for filename in csv_files]

print(data[1])

# Pad or truncate sequences to a fixed length
# You may need to adjust this based on your data
# For example, you can use torch.nn.utils.rnn.pad_sequence

# Step 2: Define the CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define your CNN architecture here

    def forward(self, x):
        # Define the forward pass of your CNN
        return x

# Step 3: Training Loop
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 50

# Split data into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.2)

# Assuming data is already padded or truncated and converted into tensors
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

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