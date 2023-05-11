import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from keras.models import load_model
from keras import layers
from keras.models import Sequential

def main():
    # Set the necessary variables
    train_dir = "foto/train"
    val_dir = "foto/validation"
    test_dir = "foto/test"
    modelpath = "C:/Users/Marijn/Documents/GitHub/Image-analysis-2/model.h5"
    batch_size = 32
    img_height = 800
    img_width = 895
    epochs = 3

    # Call the modular functions
    check_gpu_availability()
    train_ds, val_ds, test_ds, class_names, num_classes = load_data(train_dir, val_dir, test_dir, batch_size, img_height, img_width)
    normalized_train_ds, normalized_val_ds = preprocess_data(train_ds, val_ds)
    
    #building a new model
    model = build_model(img_height, img_width, num_classes)

    # #load a trained model
    # model = load_model(modelpath)

    history = train_model(model, normalized_train_ds, normalized_val_ds, epochs)
    plot_training_history(history, epochs)
    calculate_roc_auc(model,test_ds)
    # Save the model
    # save_model(model)
    

def check_gpu_availability():
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available.")
        try:
            # Select the first GPU device (if multiple are available)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            # Limit GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("Error occurred while configuring GPU:", e)
    else:
        print("GPU is not available.")


def load_data(train_dir, val_dir, test_dir, batch_size, img_height, img_width):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        class_names=None
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    return train_ds, val_ds, test_ds, class_names, num_classes


def preprocess_data(train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1./255)

    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return normalized_train_ds, normalized_val_ds


def build_model(img_height, img_width, num_classes):
    
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train_model(model, train_ds, val_ds, epochs):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return history


def plot_training_history(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show(block=False)

def calculate_roc_auc(model, test_ds):
    # Calculate predicted probabilities
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

def save_model(model):
    wd = str(os.getcwd())
    model_path = wd+"\model.h5"

    # Save the model
    model.save(model_path)
    print("Model saved successfully.")


def load_model(model_path):
    # Load the saved model
    loaded_model = load_model(model_path)
    print("Model loaded successfully.")
    return load_model

main()