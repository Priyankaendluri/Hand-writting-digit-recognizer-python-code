import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on some test images and display
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(f"Predicted: {predicted_label}, True: {true_label}", color=color)

# Pick a test image index
idx = 0
predictions = model.predict(test_images)
plt.figure(figsize=(6,3))
plot_image(idx, predictions[idx], test_labels, test_images)
plt.show()

# To predict a single image (28x28 grayscale numpy array):
def predict_digit(img_array):
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch size 1
    prediction = model.predict(img_array)
    return np.argmax(prediction)

# Example usage:
# digit = predict_digit(test_images[0])
# print(f"Predicted digit: {digit}")

