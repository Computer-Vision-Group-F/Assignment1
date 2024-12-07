import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
download_dir = "./data"
download.maybe_download_and_extract(url,download_dir)

cifar10_dir = './data/cifar-10-batches-py'
X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir)

# Checking the size of the training and testing data
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


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

import matplotlib.pyplot as plt
import numpy as np

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)

fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
for i, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == i)
    mean_image = np.mean(X_train[idxs], axis=0).astype('uint8')
    axes[i].imshow(mean_image)
    axes[i].axis('off')
    axes[i].set_title(cls)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 5))
plt.hist(y_train, bins=np.arange(num_classes + 1) - 0.5, ec='black', alpha=0.7)
plt.xticks(np.arange(num_classes), classes)
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Histogram of Class Distribution')
plt.show()


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


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize the k-NN classifier
k = 5  # You can change this number to see different results
classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = classifier.predict(X_test)

# Print out the predictions
print("First 10 Predicted labels: ", y_pred[:10])
print("First 10 Actual labels: ", y_test[:10])

# Evaluate the classifier
print("Classification report:\n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Set the range of k to test
k_values = range(1, 26)  # Testing k from 1 to 25
mean_scores = []
std_dev_scores = []

# Perform 5-fold cross-validation for each k
for k in k_values:
    classifier = KNeighborsClassifier(n_neighbors=k)
    # Obtain scores for 5-fold cross-validation
    scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    mean_scores.append(np.mean(scores))
    std_dev_scores.append(np.std(scores))

# Plotting the trend line with error bars
plt.errorbar(k_values, mean_scores, yerr=std_dev_scores, fmt='-o', ecolor='red', capsize=5)
plt.title('k-NN Varying number of neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.grid(True)
plt.show()


best_k_index = np.argmax(mean_scores)
best_k = k_values[best_k_index]
print(f"Best k: {best_k} with an average cross-validation accuracy of {mean_scores[best_k_index]:.3f}")

# Assuming you have a function or a way to load more data
# For this example, let's say you reload and combine multiple batches to have more training data
# Ensure you have up to 5000 samples for training
X_train, y_train, _, _ = data_utils.load_CIFAR10(cifar10_dir)  # Adjust this line if needed to get more data

# Subsample 5000 examples for training
if X_train.shape[0] > 5000:
    indices = np.random.choice(X_train.shape[0], 5000, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

# Reshape the data into rows again
X_train = np.reshape(X_train, (X_train.shape[0], -1))

# Initialize and train the k-NN classifier with the best k
classifier = KNeighborsClassifier(n_neighbors=best_k)
classifier.fit(X_train, y_train)

# If you also have a larger test set, use it; otherwise, use the original
# Let's assume X_test and y_test are already defined and preprocessed
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set with k={best_k}: {accuracy:.3f}")

