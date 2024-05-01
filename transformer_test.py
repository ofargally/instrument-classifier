import os
import torch
import glob
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import random
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Process labels for multi-label classification
def process_labels(value):
    value = str(value)
    return [int(v) for v in value.split(';')]

# Load test data
directory_path_test = './mfcc_post_processing_test'
file_pattern = "*.csv"
csv_files_test = glob.glob(os.path.join(directory_path_test, file_pattern))
dataframes_test = []

for file in csv_files_test:
    df = pd.read_csv(file)
    dataframes_test.append(df)
pd_test = pd.concat(dataframes_test, ignore_index=True)

test_labels = pd_test['Instruments'].apply(process_labels)
all_labels = list(range(0, 12))
mlb = MultiLabelBinarizer(classes = all_labels)
labels_encoded = mlb.fit_transform(test_labels)

# Standardize features
features = pd_test.drop('Instruments', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Convert to PyTorch tensors
test_features = torch.tensor(features_scaled).float()
test_labels = torch.tensor(labels_encoded).float()

# Create DataLoader
test_data = TensorDataset(test_features, test_labels)
batch_size = 32
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

# Load the Transformer model
class InstrumentClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(InstrumentClassifier, self).__init__()
        # Hyperparameters from the finetuning results
        dim_model = 64  # Dimension of the model
        nhead = 2       # Number of heads in multiheadattention
        num_layers = 3  # Number of transformer layers
        dropout = 0.20573244238175206  # Dropout rate

        self.embedding = nn.Linear(num_features, dim_model)
        encoder_layers = TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(dim_model, num_classes)

    def forward(self, src):
        embedded = self.embedding(src)
        transformer_output = self.transformer_encoder(embedded)
        output = self.output_layer(transformer_output[:, 0, :] if transformer_output.dim() == 3 else transformer_output)
        return output
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InstrumentClassifier(num_features=test_features.shape[1], num_classes=len(all_labels)).to(device)
model_path = "./model_dump/100percent_trans/transformer_v1_state.pth"  # Update with the actual model file path
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

true_labels = []
model_predictions = []
model_predictions_prob = []
# Evaluate the model
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted_labels = torch.sigmoid(outputs)  # Apply sigmoid to convert to probabilities
        model_predictions_prob.extend(predicted_labels.cpu().numpy())
        predicted_labels = (predicted_labels > 0.5).float()  # Binarize the output
        true_labels.extend(labels.cpu().numpy())
        model_predictions.extend(predicted_labels.cpu().numpy())

# Calculate performance metrics
hamming = hamming_loss(true_labels, model_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, model_predictions, average='macro', zero_division = 0)


print(f'Hamming Loss: {hamming:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')


model_predictions_prob = np.array(model_predictions_prob)
true_labels = np.array(true_labels)

for i in range(model_predictions_prob.shape[1]):
    fpr, tpr, thresholds = roc_curve(true_labels[:, i], model_predictions_prob[:, i])
    plt.plot(fpr, tpr, label=f'Label {i}')

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Label')
plt.legend(loc='best')

# Display the plot
plt.show()
