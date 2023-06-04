# Import necessary libraries
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import LeakyReLU


# Load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert class vectors to binary class matrices
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
model = Sequential()
model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))  # Increase number of filters
model.add(LeakyReLU(alpha=0.1))  # use LeakyReLU activation function
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))  # Increase number of filters
model.add(LeakyReLU(alpha=0.1))  # use LeakyReLU activation function
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))  # Increase number of filters
model.add(LeakyReLU(alpha=0.1))  # use LeakyReLU activation function
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))  # Increase number of filters
model.add(LeakyReLU(alpha=0.1))  # use LeakyReLU activation function
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))  # Increase number of filters
model.add(LeakyReLU(alpha=0.1))  # use LeakyReLU activation function
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), kernel_initializer='he_uniform', padding='same'))  # Increase number of filters
model.add(LeakyReLU(alpha=0.1))  # use LeakyReLU activation function
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.6))

model.add(Flatten())  # Flatten the tensor output from the previous layer
model.add(Dense(64, kernel_initializer='he_uniform'))  # Increase number of neurons
model.add(LeakyReLU(alpha=0.1))  # use LeakyReLU activation function
model.add(BatchNormalization())
model.add(Dropout(0.7))
model.add(Dense(num_classes, activation='softmax'))  # num_classes is the number of categories you have


# compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set a learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=8, 
                                            verbose=1, 
                                            factor=0.4, 
                                            min_lr=0.00001)

# Create a ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.04,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=False,
)
datagen.fit(X_train)

batch_size = 128
epochs = 120  # Increased the number of epochs


history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    callbacks=[learning_rate_reduction])

# Test the model
score = model.evaluate(X_test, y_test, verbose=1)

# Print the scores
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Predict the values from the test dataset
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# Plot the confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True, fmt="d");

# Show some misclassified examples
misclassified_idx = np.where(Y_pred_classes != Y_true)[0]
for i in range(5):
    plt.figure()
    idx = misclassified_idx[i]
    plt.imshow(X_test[idx])
    plt.title(f"True label: {Y_true[idx]} Predicted: {Y_pred_classes[idx]}");
plt.show()

# Save the trained model
test_accuracy = round(score[1] * 100, 2)
model.save(f'model_test_accuracy_{test_accuracy}.h5')