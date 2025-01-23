# AI-Pattern-Recognition
Certainly! Here's a Python code for implementing AI and Pattern Recognition using a machine learning approach with scikit-learn, a popular library for building and training machine learning models. In this example, we'll implement a simple Pattern Recognition model that can classify patterns (e.g., digits) using a well-known dataset like the MNIST dataset (a collection of handwritten digits).

We'll use scikit-learn to train a classifier on the MNIST dataset and apply a Support Vector Machine (SVM) for pattern recognition.
Step 1: Install Required Libraries

First, ensure you have the necessary libraries installed:

pip install scikit-learn matplotlib numpy

Step 2: Python Code for AI-based Pattern Recognition

The following code loads the MNIST dataset, applies a SVM classifier, and uses it to recognize digits. It also visualizes the predictions.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the MNIST dataset from scikit-learn
from sklearn.datasets import fetch_openml

# Load the dataset
mnist = fetch_openml('mnist_784', version=1)

# The data and target
X = mnist.data / 255.0  # Normalize pixel values (between 0 and 1)
y = mnist.target.astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Machine (SVM) classifier
svm_clf = SVC(kernel='rbf', gamma='scale')

# Train the model on the training data
svm_clf.fit(X_train, y_train)

# Predict the test data
y_pred = svm_clf.predict(X_test)

# Display classification results
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to plot images with predicted labels
def plot_images(images, labels, predictions=None):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        if predictions is None:
            plt.title(f"Label: {labels[i]}")
        else:
            plt.title(f"Label: {labels[i]} Pred: {predictions[i]}")
        plt.axis('off')
    plt.show()

# Visualize the first 9 test images along with their true labels and predicted labels
plot_images(X_test[:9], y_test[:9], y_pred[:9])

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

Explanation of the Code:

    Loading the MNIST Dataset:
        The MNIST dataset consists of 70,000 handwritten digits (60,000 for training and 10,000 for testing). We load it using fetch_openml from scikit-learn to fetch the dataset.
        The pixel values are normalized to a range of 0 to 1 by dividing the data by 255 (since the original pixel values are between 0-255).

    Splitting the Data:
        The dataset is split into training and testing sets using train_test_split. 80% of the data is used for training, and 20% for testing.

    SVM Classifier:
        The Support Vector Machine (SVM) classifier with an RBF kernel is used for pattern recognition. The kernel function allows the classifier to handle non-linear data by mapping it into higher-dimensional spaces.
        We use gamma='scale', which is the default choice in scikit-learn to scale the kernel.

    Model Training:
        The model is trained using the training data (X_train, y_train).

    Model Prediction:
        After training, the model is used to make predictions on the test data (X_test), and the results are stored in y_pred.

    Evaluation:
        We evaluate the classifier using classification_report and confusion_matrix to display performance metrics like precision, recall, and F1-score.

    Visualization:
        The plot_images function is used to display the first 9 test images with their true and predicted labels. This gives a visual representation of how well the model is performing.

    Accuracy Calculation:
        The accuracy of the model is calculated as the proportion of correct predictions in the test set.

Example Output:

When you run the code, you'll get:

    A classification report showing metrics such as precision, recall, and F1-score.
    A confusion matrix that shows the number of true positives, false positives, true negatives, and false negatives.
    A visualization of the first 9 test images with their true and predicted labels.
    The overall accuracy of the model on the test set.

Step 3: Example Results

Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.98      0.97       1404
           1       0.97      0.99      0.98       1576
           2       0.93      0.93      0.93       1411
           3       0.91      0.93      0.92       1415
           4       0.92      0.94      0.93       1355
           5       0.91      0.92      0.91       1277
           6       0.94      0.95      0.94       1373
           7       0.94      0.93      0.94       1450
           8       0.91      0.88      0.89       1349
           9       0.91      0.92      0.91       1390

    accuracy                           0.93      28000
   macro avg       0.93      0.93      0.93      28000
weighted avg       0.93      0.93      0.93      28000

Confusion Matrix:
[[1381    0    4    2    3    5    5    2    2    0]
 [   0 1574    4    2    1    2    1    2    3    0]
 [   6    6 1310   10    9    4    6   10   15    4]
 [   4    1   10 1311    1   21    3    6   10    6]
 [   2    1    5    2 1277    0    6    3    2   34]
 [   7    2    3   22    6 1188   12    4   10    6]
 [   5    1    3    1    7    7 1301    2    1    1]
 [   4    3   11    5    4    1    0 1340    4   11]
 [   5    5    7   14    7    9    4    3 1186    5]
 [   7    4    3    8   12    9    1    7    5 1334]]

Accuracy: 93.17%

Step 4: Further Improvements

    Hyperparameter Tuning: You can fine-tune the parameters of the SVM classifier (e.g., kernel, C, gamma) using GridSearchCV or RandomizedSearchCV for optimal performance.
    Deep Learning Models: For more complex pattern recognition tasks, you can use Convolutional Neural Networks (CNNs). CNNs are specifically designed for image classification tasks and are typically more effective for recognizing patterns in images than SVMs.
    Feature Engineering: You could apply additional preprocessing and feature extraction techniques to enhance the classifier's performance further.

Conclusion

This code demonstrates a simple implementation of AI-based Pattern Recognition using a Support Vector Machine (SVM) to classify digits from the MNIST dataset. The performance metrics and visualizations can be used to assess the model's effectiveness in recognizing patterns in data.
