#!/usr/bin/env python
# coding: utf-8

# # Assignment 1-1: K-Nearest Neighbors (k-NN)

# In this notebook you will implement a K-Nearest Neighbors classifier on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
# 
# Recall that the K-Nearest Neighbor classifier does the following:
# - During training, the classifier simply memorizes the training data
# - During testing, test images are compared to each training image; the predicted label is the majority vote among the K nearest training examples.
# 
# After implementing the K-Nearest Neighbor classifier, you will use *cross-validation* to find the best value of K.
# 
# The goals of this exercise are to go through a simple example of the data-driven image classification pipeline, and also to practice writing efficient, vectorized code in [PyTorch](https://pytorch.org/).

# ## Downloading the CIFAR-10 dataset 
import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
download_dir = "./data"
download.maybe_download_and_extract(url,download_dir)


## Loading raw files and reading them as training and testing datasets

cifar10_dir = './data/cifar-10-batches-py'
X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir)

# Checking the size of the training and testing data
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# ## Visualizing dataset samples
# To give you a sense of the nature of the images in CIFAR-10, this cell visualizes some random examples from the training set.

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


# ## Subsample the dataset
# When implementing machine learning algorithms, it's usually a good idea to use a small sample of the full dataset. This way your code will run much faster, allowing for more interactive and efficient development. Once you are satisfied that you have correctly implemented the algorithm, you can then rerun with the entire dataset.

# We will subsample the data to use only 500 training examples and 250 test examples:

# Memory error prevention by subsampling data

num_training = 500
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 250
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# reshaping data and placing into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)


# # K-Nearest Neighbors (k-NN)

# Now that we have examined and prepared our data, it is time to implement the kNN classifier. We can break the process down into two steps:
# 
# 1. Perform k-Nearest neighbours algorithm on the CiFAR-10 dataset to classify test images. 
# 2. Perform k-fold cross validation and plot the trend line with error bars that correspond to standard deviation to find the best value of the 'k' hyper parameter and best accuracy on the dataset.
# 3. Select the best value for k, and rerun the classifier on our full 5000 set of training examples.
# 4. Discussion: Discuss your understanding.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

# Print the classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_test[i].reshape(32, 32, 3).astype('uint8'))
    plt.title(f'Predicted: {classes[y_pred[i]]}')
    plt.axis('off')
plt.show()

## k-fold cross validation

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

k_values = range(1, 21)  # Test k values from 1 to 20
mean_accuracies = []
std_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(knn, X_train_flat, y_train, cv=kfold, scoring='accuracy')
    mean_accuracies.append(np.mean(cv_scores))
    std_accuracies.append(np.std(cv_scores))


plt.errorbar(k_values, mean_accuracies, yerr=std_accuracies, fmt='-o', ecolor='red', capsize=5)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Mean Accuracy')
plt.title('KNN Cross-Validation Accuracy with Error Bars')
plt.show()

best_k = k_values[np.argmax(mean_accuracies)]
print(f'Best value of k: {best_k}')


knn_best = KNeighborsClassifier(n_neighbors=best_k) 
knn_best.fit(X_train, y_train) 
y_pred_best = knn_best.predict(X_test)
print(classification_report(y_test, y_pred_best))

