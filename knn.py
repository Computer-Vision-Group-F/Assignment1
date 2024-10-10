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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
knn_sklearn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_sklearn.fit(X_train, y_train)

print("k-NN classifier has been trained.")

y_pred_sklearn = knn_sklearn.predict(X_test)

cm1 = confusion_matrix(y_test, y_pred_sklearn)
plt.figure(figsize=(10, 8))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for k-NN using Euclidean')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

accuracy = accuracy_score(y_test, y_pred_sklearn)
print(f"Accuracy of k-NN classifier: {accuracy:.4f}")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("\
Classification Report:")
print(classification_report(y_test, y_pred_sklearn, target_names=class_names))
print(f"Accuracy with Euclidean distance: {accuracy:.4f}")

knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

# Train the classifier
knn_manhattan.fit(X_train, y_train)

print("k-NN classifier using Manhattan distance has been trained.")

y_pred_manhattan = knn_manhattan.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_manhattan)
print(f"Accuracy of k-NN classifier with Manhattan distance: {accuracy:.4f}")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("\
Classification Report:")
print(classification_report(y_test, y_pred_manhattan, target_names=class_names))

cm2 = confusion_matrix(y_test, y_pred_manhattan)
plt.figure(figsize=(10, 8))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for k-NN with Manhattan Distance')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(f"Accuracy with Manhattan distance: {accuracy:.4f}")

knn_minkowski = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)

# Train the classifier
knn_minkowski.fit(X_train, y_train)

print("k-NN classifier using Minkowski distance has been trained.")
y_pred_minkowski = knn_minkowski.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_minkowski)
print(f"Accuracy of k-NN classifier with Minkowski distance: {accuracy:.4f}")

print("\
Classification Report:")
print(classification_report(y_test, y_pred_minkowski, target_names=class_names))

cm3 = confusion_matrix(y_test, y_pred_minkowski)
plt.figure(figsize=(10, 8))
sns.heatmap(cm3, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for k-NN with Minkowski Distance')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("\
Comparison:")
print(f"Accuracy with Euclidean distance: {0.1880:.4f}")
print(f"Accuracy with Manhattan distance: {0.2040:.4f}")
print(f"Accuracy with Minkowski distance: {accuracy:.4f}")

from sklearn.model_selection import KFold
from tqdm import tqdm

# Define the range of k values to test
k_values = range(1, 21)

# Initialize lists to store results
mean_scores = []
std_scores = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for k in tqdm(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = []
    
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        knn.fit(X_train_fold, y_train_fold)
        y_pred = knn.predict(X_val_fold)
        scores.append(accuracy_score(y_val_fold, y_pred))
    
    mean_scores.append(np.mean(scores))
    std_scores.append(np.std(scores))

print("Cross-validation completed.")

plt.figure(figsize=(10, 6))
plt.errorbar(k_values, mean_scores, yerr=std_scores, fmt='-o', capsize=5)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('k-NN Performance with Error Bars')
plt.grid(True)

plt.show()

best_k = k_values[np.argmax(mean_scores)]
best_accuracy = np.max(mean_scores)

print(f"Best k value: {best_k}")
print(f"Best accuracy: {best_accuracy:.4f}")

X_train_full, y_train_full, X_test_full, y_test_full = load_CIFAR10(cifar10_dir)

# Subsample to 5000 training examples
num_training_full = 5000
mask_full = list(range(num_training_full))
X_train_full = X_train_full[mask_full]
y_train_full = y_train_full[mask_full]

# Reshape the full training and testing datasets into rows
X_train_full = np.reshape(X_train_full, (X_train_full.shape[0], -1))
X_test_full = np.reshape(X_test_full, (X_test_full.shape[0], -1))

print('Full training data shape: ', X_train_full.shape)
print('Full test data shape: ', X_test_full.shape)

from sklearn.model_selection import cross_val_score

# Define the range of k values to test
k_values = range(1, 21)

# Initialize lists to store results
mean_scores = []
std_scores = []

# Perform k-fold cross-validation
for k in tqdm(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_full, y_train_full, cv=5)
    mean_scores.append(np.mean(scores))
    std_scores.append(np.std(scores))

print("Cross-validation completed.")

# Plot the results
plt.figure(figsize=(10, 6))
plt.errorbar(k_values, mean_scores, yerr=std_scores, fmt='-o', capsize=5)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('k-NN Performance with Error Bars on Full CIFAR-10 Subset (5000 samples)')
plt.grid(True)
plt.show()

best_k = k_values[np.argmax(mean_scores)]
best_accuracy = np.max(mean_scores)

print(f"Best k value: {best_k}")
print(f"Best cross-validation accuracy: {best_accuracy:.4f}")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_full, y_train_full)

test_accuracy = best_knn.score(X_test_full, y_test_full)
print(f"Test set accuracy: {test_accuracy:.4f}")