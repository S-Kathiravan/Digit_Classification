# MNIST Digit Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

## Data Preprocessing
- **Load and Reshape:**
  - Load the MNIST dataset.
  - Reshape images to include a channel dimension (28x28x1).
- **Normalization:**
  - Scale image pixel values to the [0, 1] range.
- **One-Hot Encoding:**
  - Convert labels into one-hot encoded vectors.

## Model Architecture
- **Conv2D Layers:**
  - First layer: 32 filters, 3x3 kernel, ReLU activation, MaxPooling (2x2).
  - Second layer: 64 filters, 3x3 kernel, ReLU activation, MaxPooling (2x2).
- **Dense Layer:**
  - Fully connected layer with 128 neurons and ReLU activation.
  - Dropout with 0.5 probability to prevent overfitting.
- **Output Layer:**
  - Dense layer with 10 neurons and softmax activation for digit classification.

## Training
- **Compilation:**
  - Model compiled with Adam optimizer and categorical cross-entropy loss.
- **Epochs:**
  - Trained for 10 epochs with a batch size of 128.
  - Tracked accuracy and loss during training.

## Results
- **Evaluation:**
  - Achieved `X%` accuracy on the test set.
- **Visualization:**
  - Plotted accuracy and loss curves over epochs.
  - Displayed model predictions on random test images alongside actual labels.

## Prediction
- **Display:**
  - Random test images are shown with their predicted and actual labels.
