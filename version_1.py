# Import necessary libraries
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from functions import evaluate_and_plot, evaluate_model
from keras.layers import LeakyReLU



# Check for GPU and set visible devices
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
deep_path = os.path.join(desktop, 'deepLearning-Claasification-with-CIFAR-10-dataset')  # add the 'deep' folder
model_path = os.path.join(deep_path, 'models')
os.makedirs(model_path, exist_ok=True)

# Load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize inputs
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert class vectors to binary class matrices
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the model
model = Sequential()
# model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))  # Increase number of filters
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))  # Increase number of filters
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))  # Increase number of filters
# model.add(BatchNormalization())
# model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))  # Increase number of filters
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.4))

# model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))  # Increase number of filters
# model.add(BatchNormalization())
# model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))  # Increase number of filters
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))

# model.add(Flatten())
# model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))  # Increase number of neurons
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(num_classes, activation='softmax'))
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
model.add(Dropout(0.5))

model.add(Flatten())  # Flatten the tensor output from the previous layer
model.add(Dense(64, kernel_initializer='he_uniform'))  # Increase number of neurons
model.add(LeakyReLU(alpha=0.1))  # use LeakyReLU activation function
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))  # num_classes is the number of categories you have


# compile model
optimizer = Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set a learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=8, 
                                            verbose=1, 
                                            factor=0.7, 
                                            min_lr=0.00001)

# Create a ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.12,
    zoom_range=0.12,
    width_shift_range=0.12,
    height_shift_range=0.12,
    horizontal_flip=True,
    vertical_flip=False,
)
datagen.fit(X_train)

# Batch size and number of epochs
batch_size = 256
epochs = 120



# Check if a model already exists
existing_models = os.listdir(model_path)
if len(existing_models) > 0:
    print("Existing models: ", existing_models)
    answer = input("Would you like to load an existing model? (y/n): ")
    if answer.lower() == 'y':
        model_name = input("Please enter the model name: ")
        model = load_model(os.path.join(model_path, model_name + '.h5'))
        evaluate_model(model, X_test, y_test, num_classes)
        # Call the function to fine-tune the model
        fine_tune_answer = input("Would you like to fine tune this model? (y/n): ")
        if fine_tune_answer.lower() == 'y':
            # Call the function to fine-tune the model
            fine_tune_batch_size = 64
            fine_tune_epochs = 50
            fine_tune_lr = 0.0001
            # Set a learning rate reduction
            fine_tune_learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                                        patience=5, 
                                                        verbose=1, 
                                                        factor=0.2, 
                                                        min_lr=0.00001)
            # Compile model with fine-tune learning rate
            model.compile(optimizer=Adam(learning_rate=fine_tune_lr), loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(datagen.flow(X_train, y_train, batch_size=fine_tune_batch_size),
                                epochs=fine_tune_epochs,  # use the fine_tune_epochs variable
                                validation_data=(X_test, y_test),
                                steps_per_epoch=X_train.shape[0] // batch_size,
                                callbacks=[fine_tune_learning_rate_reduction])
        
            evaluate_and_plot(history, model, X_test, y_test, num_classes)

            # Evaluate the fine-tuned model
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

            # Save the fine-tuned model
            ft_model_name = "ft_" + str(round(accuracy*100, 2))
            filename = os.path.join(model_path, ft_model_name + '.h5')
            model.save(filename)
            
            # Evaluate the fine-tuned model
            evaluate_model(model, X_test, y_test, num_classes)
    else:
       # If no, then we train a new model
        print("Training a new model...")
        history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            callbacks=[learning_rate_reduction])

        evaluate_and_plot(history, model, X_test, y_test, num_classes)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        # Save the trained model
        model_name = str(round(accuracy*100, 2))  # round the accuracy to 2 decimal places
        filename = os.path.join(model_path, model_name + '.h5')
        model.save(filename)

else:
    print("No existing model found, training a new model...")
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        callbacks=[learning_rate_reduction])
    
    evaluate_and_plot(history, model, X_test, y_test, num_classes)
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Save the trained model
    model_name = str(round(accuracy*100, 2))  # round the accuracy to 2 decimal places
    filename = os.path.join(model_path, model_name + '.h5')
    model.save(filename)



   









                                           
