from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

#--------------------------------------Feature engineering method 1----------------------------------------------
################################   HOG As Feature Extracting Method   ########################################
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage import color
import numpy as np

## Use RGB color
def extract_hog_features(data):
    hog_features = []
    for image in data:
        image = image.reshape((32, 32, 3))
        fd = hog(image, orientations=12, pixels_per_cell=(4, 4), cells_per_block=(1, 1), channel_axis=-1)
        hog_features.append(fd)
    return np.array(hog_features)

# Extract HOG features
x_train_data_fe = extract_hog_features(x_train)
x_test_data_fe = extract_hog_features(x_test)

print("train data: ", x_train_data_fe.shape)
print("test data: ", x_test_data_fe.shape)

############## visualize feature ###################
from skimage import exposure
import matplotlib.pyplot as plt
# Choose a sample image from the dataset
sample_image = x_train[0]
# Make sure the image has the right shape
image = sample_image.reshape((32, 32, 3))
# Extract HOG features and the HOG image
hog_features, hog_image = hog(image, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)

# Rescale the HOG image for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Display the original image and the HOG image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input Image')

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

plt.show()

######################### Standardize data   ##########################
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# x_train_data_standard = scaler.fit_transform(x_train_data_fe)
# x_test_data_standard = scaler.transform(x_test_data_fe)

# print("train data: ", x_train_data_standard.shape)
# print("test data: ", x_test_data_standard.shape)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train_data_standard = scaler.fit_transform(x_train_data_fe)
x_test_data_standard = scaler.transform(x_test_data_fe)
print("train data: ", x_train_data_standard.shape)
print("test data: ", x_test_data_standard.shape)

#################   PCA for dimension reduction   ##################
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(0.8)  # adjust when testing
# pca = PCA(n_components = 3) 
x_train_pca = pca.fit_transform(x_train_data_standard)
x_test_pca = pca.transform(x_test_data_standard)
print(pca.n_components_)

print("train data: ", x_train_pca.shape)
print("test data: ", x_test_pca.shape)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, edgecolor='none', alpha=0.5)
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()
# plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x_train_pca[:, 0], x_train_pca[:, 1], x_train_pca[:, 2], 
            c=y_train, edgecolor='k', alpha=0.6)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.title('3D PCA plot')
plt.colorbar(scatter)
plt.show()

#--------------------------------------Feature engineering method 2----------------------------------------------
###################   Color Histogram As Feature Extracting Method (alternative of HOG)  ##################
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage import color
import numpy as np
import cv2

def extract_color_histograms(data, bins=(8, 8, 8)):
    histogram_features = []
    for image in data:
        image = image.reshape((32, 32, 3))
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        histogram_features.append(hist.flatten())
    return np.array(histogram_features)

# Extract color histogram features
x_train_data_fe = extract_color_histograms(x_train)
x_test_data_fe = extract_color_histograms(x_test)

print("train data: ", x_train_data_fe.shape)
print("test data: ", x_test_data_fe.shape)

############## visualize feature ###################
from skimage import exposure
import matplotlib.pyplot as plt
# Choose a sample image from the dataset
sample_image = x_train[0]
# Make sure the image has the right shape
image = sample_image.reshape((32, 32, 3))
# Compute color histogram for the image
hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
cv2.normalize(hist, hist)

# Flatten the histogram 
hist = hist.flatten()

# Plot the color histogram
plt.figure()
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

######################### Standardize data   ##########################
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_data_standard = scaler.fit_transform(x_train_data_fe)
x_test_data_standard = scaler.transform(x_test_data_fe)

print("train data: ", x_train_data_standard.shape)
print("test data: ", x_test_data_standard.shape)
# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
# x_train_data_standard = scaler.fit_transform(x_train_data_fe)
# x_test_data_standard = scaler.transform(x_test_data_fe)
# print("train data: ", x_train_data_standard.shape)
# print("test data: ", x_test_data_standard.shape)


#################   PCA for dimension reduction   ##################
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(0.8)  # adjust when testing,variance to keep
x_train_pca = pca.fit_transform(x_train_data_standard)
x_test_pca = pca.transform(x_test_data_standard)
print("train data: ", x_train_pca.shape)
print("test data: ", x_test_pca.shape)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, edgecolor='none', alpha=0.5)
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()
# plt.show()

# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, edgecolor='none', alpha=0.5)
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()
# plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x_train_pca[:, 0], x_train_pca[:, 1], x_train_pca[:, 2], 
            c=y_train, edgecolor='k', alpha=0.6)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.title('3D PCA plot')
plt.colorbar(scatter)
plt.show()


#--------------------------------------Feature engineering method 3----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Display the dimensions of the training data and test data
print("Shape of training data:")
print(train_images.shape)
print(train_labels.shape)
print("Shape of test data:")
print(test_images.shape)
print(test_labels.shape)

# Map the labels (which are integers from 0 to 9) to the actual class names
lable_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

# Display images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
# The category tag is a list containing one element that needs to be extracted
    plt.xlabel(lable_names[train_labels[i][0]])
plt.show()

# Flatten the image into one-dimensional vector
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# Convert to floating point and normalize
train_images_flat = train_images_flat.astype('float32') / 255.0
test_images_flat = test_images_flat.astype('float32') / 255.0

# Create PCA object and fit the data
pca = PCA()
pca.fit(train_images_flat)
# Get eigenvalues
eigenvalues = pca.explained_variance_

# Plot out the eigenvalue spectrum
plt.plot(range(1, min(len(eigenvalues)+1, 101)), eigenvalues[:100], marker='o')
plt.xlabel('principal component')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalue spectrum')
plt.show()

# Create PCA objects and fit training set data
pca = PCA(n_components=100)  # Set the number of principal components to be retained
pca.fit(train_images_flat)

# Apply PCA transform
train_images_pca = pca.transform(train_images_flat)
test_images_pca = pca.transform(test_images_flat)

# Plotting scatter plots
plt.scatter(train_images_pca[:, 0], train_images_pca[:, 1], c=train_labels.ravel(), cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot')
plt.colorbar()
plt.show()

############################ Apply SVM(SVC) #############################
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Initialize the SVC model with desired parameters
clf = svm.SVC(kernel='poly', C=10, cache_size=10000)

print(x_train_pca.shape, y_train.shape)
# Train the model using the training data
clf.fit(x_train_pca, y_train)

# Use the model to predict the labels of the test data
pred_labels = clf.predict(x_test_pca)

# Calculate accuracy
accuracy = accuracy_score(y_test, pred_labels)
print(f'Accuracy: {accuracy*100}%')

# Print confusion matrix
cm = confusion_matrix(y_test, pred_labels)
print('Confusion Matrix:')
disp = ConfusionMatrixDisplay(cm, display_labels=['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks'])
disp.plot()
plt.xticks(rotation=45, ha='right')
plt.show()

############################ Apply Random Forest (alternative of SVC) #############################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)

# Train the Random Forest
clf.fit(x_train_pca, y_train)

# Make predictions on the test set
predictions = clf.predict(x_test_pca)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy*100}%')

# Print confusion matrix
cm = confusion_matrix(y_test, predictions)
print('Confusion Matrix:')
disp = ConfusionMatrixDisplay(cm, display_labels=['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks'])
disp.plot()
plt.xticks(rotation=45, ha='right')
plt.show()

# Now, we can check the training and validation accuracy
# rf_accuracy_train = rf.score(x_train_pca, y_train)
# rf_accuracy_val = rf.score(x_val_pca, y_val)
# print(f"Random Forest Validation Accuracy after PCA: {rf_accuracy_val}")

# Define the random forest classifier
rf = RandomForestClassifier()

# Define the parameter grid to be searched
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15]
}

# Define the cross-validation object
cv = 5 # Set the number of folds for cross-validation according to requirements
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv)
grid_search.fit(train_images_pca, train_labels)
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

# Create random forest classifier objects
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=15)

# Training with PCA-transformed features``
rf_classifier.fit(train_images_pca, train_labels.ravel())

# Prediction of the test set using the trained classifier
predictions = rf_classifier.predict(test_images_pca)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(test_labels, predictions)
print(f'Accuracy: {accuracy*100}%')

# Calculate the confusion matrix
cm = confusion_matrix(test_labels, predictions)
print('Confusion Matrix:')
disp = ConfusionMatrixDisplay(cm, display_labels=['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks'])
disp.plot()
plt.xticks(rotation=45, ha='right')
plt.show()

# Calculate precision, recall and F1 values
precision = precision_score(test_labels, predictions, average='macro')
recall = recall_score(test_labels, predictions, average='macro')
f1 = f1_score(test_labels, predictions, average='macro')

# Print Results
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)






