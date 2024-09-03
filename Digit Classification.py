#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

#Loading the Data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#Reshaping the Data
train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)

# Normalize the pixel values
train_X = train_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0

# One-hot encode the labels
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

#Display the Sample Inputs
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_X[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(train_y[i])}")
    plt.axis('off')
plt.show()

#Build the CNN model
model = Sequential()

# First convolutional layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output and add a dense layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer with 10 neurons (one for each digit)
model.add(Dense(10, activation='softmax'))

#Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Train the model
history = model.fit(train_X, train_y,
                    validation_data=(test_X, test_y),epochs=10,
                    batch_size=128,verbose=2)

# Plot the accuracy and loss over epochs
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_X, test_y, verbose=0)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Make predictions on the test set
predictions = model.predict(test_X)

# Select a few random samples to display predictions
num_samples = 9
indices = np.random.choice(len(test_X), num_samples, replace=False)

# Plot the sample predictions
plt.figure(figsize=(10, 10))
for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[idx].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[idx])
    actual_label = np.argmax(test_y[idx])
    plt.title(f"Predicted: {predicted_label}, Actual: {actual_label}")
    plt.axis('off')
plt.show()
