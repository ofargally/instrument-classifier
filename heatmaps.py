import seaborn as sns
import matplotlib.pyplot as plt

data = [
    [1-0.4075, 0.0925, 0.4167, 0.1423],  # High Recall Model 100% CNN BCE scores: [Hamming Loss, Precision, Recall, F1]
    [1-0.0835, 0.1458, 0.1683, 0.1370],#Low Hamming Loss, 50% CNN BCELogitLoss
    [1-0.1682, 0.1515, 0.3811, 0.2098] # Transformer Model 100%: [Hamming Loss, Precision, Recall, F1]
]




sns.heatmap(data, font_scale=1.5, annot=True, cmap='coolwarm', xticklabels=['1-Hamming Loss', 'Precision', 'Recall', 'F1'], yticklabels=['CNN Model - BCE', 'CNN Model - BCELogitLoss', 'Transformer Model'])



# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Models')
plt.title('Heatmap of Model Performance Metrics')

# Display the heatmap
plt.show()