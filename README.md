#**MNIST Digit Classification using CNN**
  ####This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

  ##Data Preprocessing:
    -Load and Reshape: Load the MNIST dataset and reshape the images to include a channel dimension.
  ##Normalization: Scale image pixel values to the [0, 1] range.
    -One-Hot Encoding: Convert labels into one-hot encoded vectors.
  ##Model Architecture:
    -Conv2D Layers: Two convolutional layers with ReLU activation, each followed by MaxPooling to reduce dimensionality.
    -Dense Layer: A fully connected layer with 128 neurons and ReLU activation, followed by Dropout to prevent overfitting.
    -Output Layer: A dense layer with 10 neurons and softmax activation to classify the digits.
  ##Training:
    -Compilation: Model compiled with Adam optimizer and categorical cross-entropy loss.
    -Epochs: Trained for 10 epochs with batch size 128, tracking accuracy and loss.
  ##Results:
    -Evaluation: Achieved X% accuracy on the test set.
    -Visualization: Plotted accuracy/loss curves and displayed model predictions on sample test images.
  ##Prediction:
    -Display: Predicted labels of random test images are shown alongside actual labels.
