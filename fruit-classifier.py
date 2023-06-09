import os
import tensorflow as tf
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Set the paths to your dataset directories
train_data_dir = "C:/Users/marij/Documents/Image-analysis-foto/Fruit/train"
test_data_dir = "C:/Users/marij/Documents/Image-analysis-foto/Fruit/test"
modelpath = "C:/Users/marij/Documents/GitHub/Image-analysis-2/model-2.h5"
num_classes = len(os.listdir(train_data_dir))
image_size = (100, 100)
batch_size = 128
epochs = 5
validation_split = 0.2

# Create data generators with data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Create the model
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

modelpath = "C:/Users/marij/Documents/GitHub/Image-analysis-2/model-3.h5"
model.save(modelpath)
print("Model saved successfully.")

model = tf.keras.models.load_model(modelpath)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


