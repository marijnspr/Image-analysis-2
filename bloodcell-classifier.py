import os
import numpy as np
from scipy import optimize
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import layers
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 16
img_height = 240
img_width = 320

def main():
    # Set the necessary variables
    dir = "C:/Users/marij/Documents/Image-analysis-foto/Bloodcells/"
    modelpath = "C:/Users/Marijn/Documents/GitHub/Image-analysis-2/model.h5"
    epochs = 10

    # Call the modular functions
    # check_gpu_availability()
    train_ds, val_ds, class_names, num_classes, test_ds = create_validation(dir+"TRAIN", dir+"TEST")
    normalized_train_ds, normalized_val_ds = preprocess_data(train_ds, val_ds)

    model = build_model(img_height, img_width, num_classes)

    history = train_model(model, normalized_train_ds, normalized_val_ds, epochs, num_classes)
    plot_training_history(history, epochs)
    calculate_roc_auc(model, test_ds)
    save_model(model)

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
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=420,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        validation_split=0.2,
        subset="validation",
        seed=420,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("Number of classes:", num_classes)
    print("Class names:", class_names)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names, num_classes



def preprocess_data(train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE

    # Load and preprocess train dataset
    normalized_train_ds = train_ds.map(
        lambda x, y: (load_and_preprocess_image(x), y),
        num_parallel_calls=AUTOTUNE
    )
    normalized_train_ds = normalized_train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    # Load and preprocess validation dataset
    normalized_val_ds = val_ds.map(
        lambda x, y: (load_and_preprocess_image(x), y),
        num_parallel_calls=AUTOTUNE
    )
    normalized_val_ds = normalized_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return normalized_train_ds, normalized_val_ds


def load_and_preprocess_image(image_path):
    # Load image
    image = tf.io.read_file(image_path)
    # Decode and preprocess image
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = image / 255.0  # Normalize image
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    return image
 



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


def train_model(model, train_ds, val_ds, epochs, num_classes):
    # Define loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Define metrics
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    @tf.function

    @tf.function
    def train_step(images, labels):
        labels = tf.strings.to_number(labels, out_type=tf.int32)  # Convert labels to integer
        labels = tf.one_hot(labels, num_classes)
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss_value = loss_fn(labels, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric(labels, logits)
        return loss_value





    @tf.function
    def test_step(images, labels):
        val_logits = model(images, training=False)
        val_acc_metric(labels, val_logits)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        print("\nStart of epoch", epoch + 1)
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            loss_value = train_step(x_batch_train, y_batch_train)

            if step % 10 == 0:
                print("Training loss (for one batch) at step", step, ":", float(loss_value))

        train_acc = train_acc_metric.result()
        print("Training accuracy over epoch: %.4f" % (float(train_acc),))

        for x_batch_val, y_batch_val in val_ds:
            test_step(x_batch_val, y_batch_val)

        val_acc = val_acc_metric.result()
        print("Validation accuracy: %.4f" % (float(val_acc),))

        history["loss"].append(float(loss_value))
        history["accuracy"].append(float(train_acc))
        history["val_loss"].append(float(loss_value))
        history["val_accuracy"].append(float(val_acc))

        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

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
    plt.show()
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
    plt.show()
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

def create_validation(train_dir, test_dir):
    image_paths = []
    labels = []
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            image_path = os.path.join(root, file)
            image_paths.append(image_path)
            
            # Extract the label from the subdirectory name
            label = os.path.basename(root)
            labels.append(label)
    # Get the unique class labels and count the number of classes
    class_labels = list(set(labels))
    num_classes = len(class_labels)
    # Split the data into training and validation sets
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))
    test_data_gen = ImageDataGenerator(rescale=1./255)

    test_ds = test_data_gen.flow_from_directory(
        test_dir,
        target_size=(240, 320),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return train_dataset, val_dataset, class_labels, num_classes, test_ds


main()