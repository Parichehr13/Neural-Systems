#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Exercises on supervised learning: Training deep neural networks
# Improved full version
# ============================================================

from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path

utils_dir = Path(__file__).resolve().parents[1] / "utilities" / "Python"
if utils_dir.exists() and str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))
from load_my_digit import load_my_digit

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# Helper functions
# ============================================================
def to_one_hot(y_dense, C):
    """
    Custom one-hot encoder.
    y_dense: ndarray (n_examples,) with values in {0, ..., C-1}
    """
    y_dense = np.asarray(y_dense).reshape(-1)
    return np.eye(C, dtype=np.int32)[y_dense]


def plot_history(history):
    train_loss = history.history["loss"]
    train_acc = history.history["accuracy"]
    valid_loss = history.history["val_loss"]
    valid_acc = history.history["val_accuracy"]
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, linewidth=2, label="training")
    plt.plot(epochs, valid_loss, linewidth=2, label="validation")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.title("Training history")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc, linewidth=2, label="training")
    plt.plot(epochs, valid_acc, linewidth=2, label="validation")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred, split_name="Test"):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # Raw confusion matrix
    cmtx = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cmtx)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"{split_name} confusion matrix")
    plt.tight_layout()
    plt.show()

    # Normalized confusion matrix
    cmtx_norm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cmtx_norm)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
    plt.title(f"{split_name} normalized confusion matrix")
    plt.tight_layout()
    plt.show()


def evaluate_split(model, x_data, labels_data, split_name="Test", show_confusion=True):
    proba = model.predict(x_data, verbose=0)
    y_pred = np.argmax(proba, axis=-1)
    labels_data = np.asarray(labels_data).reshape(-1)

    acc = np.mean(labels_data == y_pred)
    print("#" * 10 + f" {split_name} set")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(labels_data, y_pred, digits=4))

    if show_confusion:
        plot_confusion(labels_data, y_pred, split_name=split_name)

    return y_pred, proba


def plot_prediction_examples(x_data, labels_true, y_pred, n_examples=12):
    labels_true = np.asarray(labels_true).reshape(-1)
    wrong_idx = np.where(labels_true != y_pred)[0]

    if len(wrong_idx) == 0:
        print("No misclassified examples found.")
        return

    n_examples = min(n_examples, len(wrong_idx))
    chosen = wrong_idx[:n_examples]

    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(chosen, start=1):
        plt.subplot(3, 4, i)
        img = np.squeeze(x_data[idx])

        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)

        plt.title(f"T:{labels_true[idx]}  P:{y_pred[idx]}")
        plt.axis("off")

    plt.suptitle("Misclassified examples")
    plt.tight_layout()
    plt.show()


def plot_cnn_kernels_and_features(model, x_example):
    conv_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]

    if len(conv_layers) == 0:
        print("No Conv2D layers found in the current model.")
        return

    # ---- First conv layer kernels
    first_conv = conv_layers[0]
    weights, biases = first_conv.get_weights()
    n_kernels = weights.shape[-1]

    plt.figure(figsize=(12, 8))
    n_cols = 4
    n_rows = int(np.ceil(n_kernels / n_cols))
    for i in range(n_kernels):
        plt.subplot(n_rows, n_cols, i + 1)

        # show first channel of kernel
        kernel_img = weights[:, :, 0, i]
        plt.imshow(kernel_img, cmap="gray")
        plt.title(f"Kernel {i}")
        plt.axis("off")

    plt.suptitle("First convolutional layer kernels")
    plt.tight_layout()
    plt.show()

    # ---- Feature maps from real forward pass
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=[layer.output for layer in conv_layers]
    )

    feature_maps = feature_extractor.predict(x_example, verbose=0)

    for layer_idx, fmap in enumerate(feature_maps):
        n_maps = fmap.shape[-1]
        n_show = min(n_maps, 16)

        plt.figure(figsize=(12, 8))
        for i in range(n_show):
            plt.subplot(4, 4, i + 1)
            plt.imshow(fmap[0, :, :, i], cmap="gray")
            plt.title(f"L{layer_idx + 1} map {i}")
            plt.axis("off")

        plt.suptitle(f"Feature maps from Conv Layer {layer_idx + 1}")
        plt.tight_layout()
        plt.show()


# ============================================================
# Dataset choice
# ============================================================
dataset_name = "mnist"   # 'mnist', 'fashion_mnist', 'cifar10'
model_type = "dense"     # 'dense' or 'cnn'

# ============================================================
# Load dataset
# ============================================================
if dataset_name == "mnist":
    (x_train, labels_train), (x_test, labels_test) = keras.datasets.mnist.load_data()
    H, W, channels = 28, 28, 1
    C = len(np.unique(labels_train))

elif dataset_name == "fashion_mnist":
    (x_train, labels_train), (x_test, labels_test) = keras.datasets.fashion_mnist.load_data()
    H, W, channels = 28, 28, 1
    C = len(np.unique(labels_train))

elif dataset_name == "cifar10":
    (x_train, labels_train), (x_test, labels_test) = keras.datasets.cifar10.load_data()
    labels_train = labels_train.reshape(-1)
    labels_test = labels_test.reshape(-1)
    H, W, channels = 32, 32, 3
    C = len(np.unique(labels_train))

else:
    raise ValueError("Undefined dataset name.")

print("Input data type:", x_train.dtype)
print("Min value:", x_train.min())
print("Max value:", x_train.max())
print("Shape of the training examples:", x_train.shape)
print("Shape of the test examples:", x_test.shape)

# scale images to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Input data type (after scaling):", x_train.dtype)
print("Min value (after scaling):", x_train.min())
print("Max value (after scaling):", x_train.max())

# make sure images have explicit channel dimension
if channels == 1:
    n_examples_train = x_train.shape[0]
    n_examples_test = x_test.shape[0]
    x_train = x_train.reshape((n_examples_train, H, W, 1))
    x_test = x_test.reshape((n_examples_test, H, W, 1))
else:
    n_examples_train = x_train.shape[0]
    n_examples_test = x_test.shape[0]

print("Shape of the training examples (after reshaping):", x_train.shape)
print("Shape of the test examples (after reshaping):", x_test.shape)

labels_train = np.asarray(labels_train).reshape(-1)
labels_test = np.asarray(labels_test).reshape(-1)

print("Shape of the training labels:", labels_train.shape)
print("Shape of the test labels:", labels_test.shape)

# ============================================================
# Train / validation split
# ============================================================
x_train, x_valid, labels_train, labels_valid = train_test_split(
    x_train,
    labels_train,
    test_size=0.1,
    random_state=42,
    stratify=labels_train
)

y_train = to_one_hot(labels_train, C)
y_valid = to_one_hot(labels_valid, C)
y_test = to_one_hot(labels_test, C)

print("Shape of the training labels (after one-hot encoding):", y_train.shape)
print("Shape of the validation labels (after one-hot encoding):", y_valid.shape)
print("Shape of the test labels (after one-hot encoding):", y_test.shape)

# ============================================================
# Plot one example
# ============================================================
target_idx = np.random.randint(low=0, high=x_train.shape[0], size=1)
target_img = np.squeeze(x_train[target_idx, :, :, :])
target_lbl = np.squeeze(y_train[target_idx])

plt.figure(figsize=(6, 6))
if target_img.ndim == 2:
    plt.imshow(target_img, cmap="gray")
else:
    plt.imshow(target_img)
plt.title("One-hot label: " + str(target_lbl))
plt.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# Model design
# ============================================================
input_shape = (H, W, channels)

if model_type == "dense":
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Flatten(),

            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.25),

            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.25),

            keras.layers.Dense(C, activation="softmax"),
        ]
    )

elif model_type == "cnn":
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),

            keras.layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Flatten(),
            keras.layers.Dense(C, activation="softmax"),
        ]
    )

else:
    raise ValueError("Undefined model type.")

# Inspect model
model.summary()

# ============================================================
# Compile model
# ============================================================
lr = 0.001
momentum = 0.9
optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

# ============================================================
# Callbacks
# ============================================================
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="best_mdl.keras",
    monitor="val_loss",
    save_best_only=True
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

# ============================================================
# Training
# ============================================================
batch_size = 128
max_epochs = int(os.getenv("EX11_MAX_EPOCHS", "100"))

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(x_valid, y_valid),
    callbacks=[model_checkpoint_callback, early_stop],
    verbose=1
)

# ============================================================
# Plot training curves
# ============================================================
plot_history(history)

# ============================================================
# Load best model and evaluate
# ============================================================
model = keras.models.load_model("best_mdl.keras")

# Training set: accuracy/report only, no confusion matrix clutter
y_pred_train, proba_train = evaluate_split(
    model, x_train, labels_train, "Training", show_confusion=False
)

# Validation and test: keep confusion matrices
y_pred_valid, proba_valid = evaluate_split(
    model, x_valid, labels_valid, "Validation", show_confusion=True
)

y_pred_test, proba_test = evaluate_split(
    model, x_test, labels_test, "Test", show_confusion=True
)

# Show wrong predictions on test set
plot_prediction_examples(x_test, labels_test, y_pred_test, n_examples=12)

# ============================================================
# Evaluate on personal digit (only meaningful for grayscale 28x28 datasets)
# ============================================================
if channels == 1 and H == 28 and W == 28:
    sample_digit_path = str(utils_dir / "sample_digit.jpg")
    my_digit = load_my_digit(sample_digit_path)

    plt.figure(figsize=(4, 4))
    plt.imshow(my_digit, cmap="gray")
    plt.title("My handwritten digit")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("Input data type:", my_digit.dtype)
    print("Shape of my digit:", my_digit.shape)
    print("Min value:", my_digit.min())
    print("Max value:", my_digit.max())

    my_digit = my_digit.astype("float32") / 255.0
    print("Input data type (after scaling):", my_digit.dtype)
    print("Min value (after scaling):", my_digit.min())
    print("Max value (after scaling):", my_digit.max())

    my_digit = my_digit.reshape((1, H, W, 1))
    print("Shape of my digit (after reshaping):", my_digit.shape)

    proba = model.predict(my_digit, verbose=0)
    y_pred_my_digit = np.argmax(proba, axis=-1)

    print("Predicted class of my digit:", int(y_pred_my_digit[0]))
    print("Class probabilities:", np.round(proba[0], 4))

# ============================================================
# CNN figures: kernels and feature maps
# ============================================================
if model_type == "cnn":
    idx = 0
    x_to_viz = x_train[idx:idx + 1]  # keep batch dimension

    plt.figure(figsize=(4, 4))
    img = np.squeeze(x_to_viz[0])
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.title("Example under investigation")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plot_cnn_kernels_and_features(model, x_to_viz)
