import os
import tensorflow as tf
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as plt

# Set the paths to your dataset directories
# Set the paths to your dataset directories
train_data_dir = "C:/Users/marij/Documents/Image-analysis-foto/Fruit/train"
test_data_dir = "C:/Users/marij/Documents/Image-analysis-foto/Fruit/test"
num_classes = len(os.listdir(train_data_dir))
image_size = (100, 100)
batch_size = 128
epochs = 5
validation_split = 0.2

# # Print all files in the training directory
# for root, dirs, files in os.walk(train_data_dir):
#     for file in files:
#         print(os.path.join(root, file).replace('\\', '/'))

# for root, dirs, files in os.walk(test_data_dir):
#     for file in files:
#         print(os.path.join(root, file).replace('\\', '/'))

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
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
modelpath = "C:/Users/Marijn/Documents/GitHub/Image-analysis-2/model.h5"
model.save(model_path)
print("Model saved successfully.")
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

# Create the figure with the specified size
plt.figure(figsize=(8, 8))

# Subplot 1: Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Subplot 2: Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Display the figure
plt.show()


predicted_probs = model.predict(test_ds)
predicted_labels = np.argmax(predicted_probs, axis=1)  # Convert probabilities to class labels

true_labels = []
for _, labels_batch in test_ds:
    true_labels.extend(labels_batch.numpy())

# Calculate the TPR and FPR for different classification thresholds
sorted_indices = np.argsort(predicted_probs[:, 1])[::-1]  # Sort by probabilities of the positive class
sorted_labels = np.array(true_labels)[sorted_indices]
sorted_probs = predicted_probs[:, 1][sorted_indices]  # Assuming the positive class is at index 1

tpr = [0.0]
fpr = [0.0]

num_positives = np.sum(true_labels)
num_negatives = len(true_labels) - num_positives

for i in range(len(sorted_labels)):
    if sorted_labels[i] == 1:
        tpr.append(tpr[-1] + 1 / num_positives)
        fpr.append(fpr[-1])
    else:
        tpr.append(tpr[-1])
        fpr.append(fpr[-1] + 1 / num_negatives)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show(block=False)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)