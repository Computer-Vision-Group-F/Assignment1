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

from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# as we cannot  give images to knn we need to reshape them 
X_train_reshape = X_train.reshape(X_train.shape[0], -1)
X_test_reshape = X_test.reshape(X_test.shape[0], -1)

# Training the k-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_reshape, y_train)

# now we will predict using the model we just trained 
Y_pred = knn.predict(X_test_reshape)

# Calculate the accuracy of the k-NN model on the test set
accuracy = np.mean(Y_pred == y_test)

print("Accuracy of k-NN model on test set is:", accuracy)

# we will chak for all values of K from 1 to 21
k_values = range(1, 21)

cv_performance = []   # we will store the preformance here

# Perform k-fold cross-validation for each k
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_reshape, y_train, cv=kf, scoring='accuracy')
    cv_performance.append(scores)

cv_means = [np.mean(scores) for scores in cv_performance]    # calculating mean 
cv_stds = [np.std(scores) for scores in cv_performance]      # calculation stander diviation

# Plot the results
plt.errorbar(k_values, cv_means, yerr=cv_stds, fmt='-o')
plt.title('k-NN vs values of K')
plt.xlabel('Number of Neighbors k')
plt.ylabel('k-fold Cross-Validated Accuracy')
plt.show()

best_k_index = np.argmax(cv_means)
best_k = k_values[best_k_index]
best_accuracy = cv_means[best_k_index]

print("The best k value in this case is:", best_k)
print("Accuracy for that K is:", best_accuracy)

# Training the k-NN model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_reshape, y_train)

# now we will predict using the model we just trained 
Y_pred = knn.predict(X_test_reshape)

# Calculate the accuracy of the k-NN model on the test set
accuracy = np.mean(Y_pred == y_test)

print("Accuracy of k-NN model on test set is:", accuracy)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, Y_pred)
plt.figure(figsize=(10, 8))
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()         # plotting confusion matrix

# increased the subsample from 500 to a higher number .

# In order to further Improve the KNN model I am going to perform the following steps
# 1. Normalize the pixel values of the images to scale them between 0 and 1.
# 2. KNN might struggle with high-dimensional data. Reducing the dimensionality using PCA.
# 3. Try different values of K and experiment with different distance metrics. (Grid Search))
# 4. plot the data to see the clusters.

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

# Normalize the pixel values to [0, 1] range
X_train = X_train / 255.0
X_test = X_test / 255.0

num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 2500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# reshaping data and placing into rows
X_train_flat = np.reshape(X_train, (X_train.shape[0], -1))
X_test_flat = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train_flat.shape, X_test_flat.shape)

#Reducing the dimensionality using PCA.
# Reduce dimensions to 50 principal components
pca = PCA(n_components=30)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

#Checking the shape of the data after dimensionality reduction

print(X_train_pca.shape, X_test_pca.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_reshape, y_train)

Y_pred = knn.predict(X_test_reshape)

# Calculate the accuracy of the k-NN model on the test set with 2500 images
accuracy = np.mean(Y_pred == y_test)

print("Accuracy of k-NN model on test set is:", accuracy)
#accuracy = 0.2628

# checking class imbalance 
plt.hist(y_train, bins=10)
plt.show()
# 4. plot the data to see the clusters.

# Plot the first two principal components
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='tab10', alpha=0.7)

# Add a colorbar for class labels
legend = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend)

plt.title('Data Visualization after PCA (First Two Principal Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
#(Grid Search)

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Set up KNN with grid search for different K values and distance metrics
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    'metric': ['euclidean', 'manhattan', 'cosine']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_pca, y_train)
print("Best parameters:", grid_search.best_params_)

# Use the best model
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_pca)
from sklearn.metrics import accuracy_score

# Calculate the accuracy of the k-NN model on the test set with 3000 images
# Predict on the training set
y_train_pred = best_knn.predict(X_train_pca)
# Predict on the test set
y_test_pred = best_knn.predict(X_test_pca)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
# Print both accuracies
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# accuracy = 44.1%


# we will chak for all values of K from 1 to 21
k_values = range(1, 21)

cv_performance = []   # we will store the preformance here

# Perform k-fold cross-validation for each k
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_reshape, y_train, cv=kf, scoring='accuracy')
    cv_performance.append(scores)
    
cv_means = [np.mean(scores) for scores in cv_performance]    # calculating mean 
cv_stds = [np.std(scores) for scores in cv_performance]      # calculation stander diviation

# Plot the results
plt.errorbar(k_values, cv_means, yerr=cv_stds, fmt='-o')
plt.title('k-NN vs values of K')
plt.xlabel('Number of Neighbors k')
plt.ylabel('k-fold Cross-Validated Accuracy')
plt.show()

# We saw a huge improvement in the performance of the model 
# after increasing the total number if images. 
# Model accuracy improved from 22% to 44% after implementing
# the 5 steps we listed above.