# Instrument-classifier - Musical Instrument Classification Using Deep Learning

A full Poster Presentation for this Project can be found using this link:
https://docs.google.com/presentation/d/18IBeH0pW_7lbweRn_8ji28IWwwTs_9tHs45XLeV4qJg/edit?usp=sharing


![Project Banner]()

## Table of Contents

- [Introduction](#introduction)
  - [Research Question](#research-question)
  - [Hypothesis](#hypothesis)
  - [Why This Task](#why-this-task)
  - [Previous Works](#previous-works)
- [Background](#background)
- [Dataset](#dataset)
  - [MusicNet](#musicnet)
  - [MFCCs](#mfccs)
- [Experimental Set-Up](#experimental-set-up)
  - [Preprocessing](#preprocessing)
  - [Model 1: Convolutional Neural Network (CNN)](#model-1-convolutional-neural-network-cnn)
  - [Model 2: Transformers](#model-2-transformers)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
  - [Model Performance](#model-performance)
  - [Figures](#figures)
- [Summary and Conclusion](#summary-and-conclusion)
- [Future Directions](#future-directions)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Accurately identifying the musical instruments playing in a segment of music is a complex task, especially in polyphonic compositions where multiple instruments perform simultaneously. This project explores the feasibility of training deep learning models to classify the instruments present in `.wav` audio files by leveraging sound properties to learn and identify hidden features of the music.

### Research Question

**Can we train deep learning models to classify the musical instrument(s) playing in a segment of music?** Specifically, given a `.wav` file, can our model utilize sound properties to learn hidden features of the music and accurately identify the instruments present?

### Hypothesis

We hypothesize that our deep learning models will accurately predict the most represented classes of instruments, particularly **keyboards** and **strings**, due to their prominent presence in the dataset.

### Why This Task

Identifying musical instruments accurately requires a trained ear and extensive experience with a wide range of instruments. Automating this process can significantly aid in tasks such as **music transcription**, **reproduction**, and **musicological analysis**, making it more accessible and efficient.

### Previous Works

While instrument classification has been extensively explored for various datasets using CNNs, **no benchmark exists for instrument classification within the MusicNet dataset**. This project fills that gap by providing a comprehensive study and baseline for future research in this area.

**References:**
1. Reddy, C. K. A., Gopa, V., Dubey, H., Matusevych, S., Cutler, R., & Aichner, R. (2022). *MusicNet: Compact Convolutional Neural Network for Real-time Background Music Detection* (arXiv:2110.04331). arXiv.
2. Han, Y., Kim, J., & Lee, K. (2017). *Deep Convolutional Neural Networks for Predominant Instrument Recognition in Polyphonic Music*. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 25(1), 208–221.
3. Haidar-Ahmad, L. (2019). *Music and instrument classification using deep learning technics*. Recall, 67(37.00), 80-00.

## Background

Musical instrument classification in polyphonic music presents unique challenges, as multiple instruments can overlap in their sound production. Traditional machine learning models struggle to isolate individual instruments in such settings. To address this, we consolidated over 120 instrument types from the MusicNet dataset into 12 broader classes, simplifying the classification task and improving model performance.

## Dataset

### MusicNet

- **Description:** MusicNet is a curated dataset consisting of over 300 annotated classical music recordings.
- **Content:** Contains more than 120 instrument types, which we have grouped into **12 predefined classes** for this classification task.
- **Annotations:** Each `.wav` file is meticulously annotated note by note, including start time, end time, and the instrument playing each note.
- **Source:** [MusicNet Dataset](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset)

### MFCCs

- **Description:** Mel-Frequency Cepstral Coefficients (MFCCs) are coefficients that make up a mel-frequency cepstrum, representing the short-term spectral shape of sound.
- **Purpose:** Serve as the primary features for our prediction models.

## Experimental Set-Up

### Preprocessing

The CSV files provided by the MusicNet dataset are processed using Python’s `librosa` library. We extract **13 MFCCs** with a hop length of **11.6 ms**. Each row in the processed data represents an MFCC frame synchronized with the corresponding time intervals in the audio file.

### Model 1: Convolutional Neural Network (CNN)

- **Architecture:**
  - **Convolutional Layers:** Features a two-layer convolutional network, transitioning from 32 to 64 channels, followed by ReLU activations and MaxPooling layers with a stride of 1.
  - **Fully Connected Layer:** Converts the flattened features from the convolutional layers to 12 output classes corresponding to the predefined instrument groups.
- **Objective:** Extract spatial features from the MFCC frames to classify the presence of instruments.

### Model 2: Transformers

- **Architecture:**
  - **Transformer Encoder:** Utilizes a single-layer Transformer encoder with 128-dimensional embeddings and 4 attention heads.
  - **Dropout:** Applies a dropout rate of approximately 0.429 to prevent overfitting.
  - **Output Layer:** Maps the transformer outputs to the 12 instrument classes.
- **Objective:** Leverage the attention mechanism to capture temporal dependencies and complex relationships within the MFCC features for accurate classification.

## Evaluation Metrics

To assess the performance of our models, we employed the following metrics:

- **ROC Curve and AUC (Area Under the Curve):** Measures the trade-off between true positive rate (sensitivity) and false positive rate (1 - specificity), indicating the model’s ability to distinguish between classes.
- **Hamming Loss:** Calculates the fraction of incorrectly predicted labels, suitable for multi-label classification tasks. Lower values indicate better performance.
- **Precision, Recall, F1 Score:** Evaluate the accuracy of the positive predictions, the model’s ability to find all relevant instances, and the harmonic mean of precision and recall, respectively.

## Results

### Model Performance

- **CNN Model:**
  - **Recall:** Achieved a surprisingly high recall, indicating the model's strong ability to identify instruments when they are present.
  - **Precision:** Exhibited low precision, implying frequent misclassifications where the model incorrectly identified instruments that were not present.
  - **Effect of Loss Function:** Using `BCEWithLogitLoss` reduced the Hamming Loss but also led to a decrease in recall.

- **Transformer Model:**
  - **Performance:** Demonstrated the best overall performance with decently high recall and lower Hamming Loss compared to the CNN models.

### Figures

- **Figure 1:** Heatmap of Model Performance Metrics comparing CNN models with different loss functions and the Transformer model. We take the inverse of Hamming Loss for consistency, so scores near 1.0 are preferable for that metric.

- **Figure 2:** ROC Curve for Transformer Model showcasing the model’s ability to predict each instrument class effectively.

![Figure 1: Heatmap of Model Performance Metrics](figure1_heatmap.png)
*Figure 1. Heatmap of Model Performance Metrics.*

![Figure 2: ROC Curve for Transformer Model]
*Figure 2. ROC Curve for Transformer Model.
*<img width="975" alt="ROC_Curve" src="https://github.com/user-attachments/assets/f23ddf7d-4a72-47e6-94de-f03ba2cedf47">


## Summary and Conclusion

Classifying multiple instruments playing simultaneously remains a challenging task. Our experiments revealed that while the CNN model achieved high recall, it struggled with precision, leading to frequent misclassifications. The Transformer model outperformed the CNNs by balancing recall and Hamming Loss more effectively.

**Key Insights:**
- **Difficulty in Feature Isolation:** The overlapping sounds of different instruments make it hard for models to learn distinct features from MFCC coefficients.
- **Model Generalizability:** The current models are likely tailored to classical and romantic era music, showing reasonable performance on datasets containing works from composers like Mozart, Bach, and Beethoven.

## Future Directions

To enhance the performance and generalizability of our models, future work will focus on:

- **Expanding Training Data:** Training on the entire dataset to capture a more diverse range of instrument classes and improve model robustness.
- **Hyperparameter Tuning:** Conducting more extensive training with additional epochs and fine-tuning hyperparameters to optimize model performance.
- **Advanced Feature Extraction:** Exploring alternative feature extraction methods beyond MFCCs to better capture the nuances of polyphonic music.
- **Class Performance Analysis:** Identifying which instrument classes are best predicted and which ones require improvement, potentially addressing class imbalances and refining model architectures accordingly.

## References

1. Reddy, C. K. A., Gopa, V., Dubey, H., Matusevych, S., Cutler, R., & Aichner, R. (2022). *MusicNet: Compact Convolutional Neural Network for Real-time Background Music Detection* (arXiv:2110.04331). arXiv.
2. Han, Y., Kim, J., & Lee, K. (2017). *Deep Convolutional Neural Networks for Predominant Instrument Recognition in Polyphonic Music*. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 25(1), 208–221.
3. Haidar-Ahmad, L. (2019). *Music and instrument classification using deep learning technics*. Recall, 67(37.00), 80-00.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**
    ```bash
    git checkout -b feature/YourFeature
    ```
3. **Commit Your Changes**
    ```bash
    git commit -m "Add YourFeature"
    ```
4. **Push to the Branch**
    ```bash
    git push origin feature/YourFeature
    ```
5. **Open a Pull Request**

Please ensure your code adheres to the project's coding standards and passes all tests.


