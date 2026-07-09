import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report


# -------------------------------------------------------------------------------------------
# Results папкасын жасау

os.makedirs("results", exist_ok=True)


# -------------------------------------------------------------------------------------------
# Dataset дайындау

train_dir = "dataset/train"
test_dir = "dataset/test"

img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

print("Class indices:", train_data.class_indices)
print("Number of classes:", train_data.num_classes)


# -------------------------------------------------------------------------------------------
# MobileNetV2 Transfer Learning

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -------------------------------------------------------------------------------------------
# Training

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)


# -------------------------------------------------------------------------------------------
# Accuracy графигі

plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("results/training_validation_accuracy.png", dpi=300)
plt.show()

print("Saved: results/training_validation_accuracy.png")


# -------------------------------------------------------------------------------------------
# Loss графигі

plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("results/training_validation_loss.png", dpi=300)
plt.show()

print("Saved: results/training_validation_loss.png")


# -------------------------------------------------------------------------------------------
# Fine-tuning

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)


# -------------------------------------------------------------------------------------------
# Fine-tuning accuracy графигі

plt.figure(figsize=(8, 6))
plt.plot(fine_tune_history.history["accuracy"], label="Fine-tune Train Accuracy")
plt.plot(fine_tune_history.history["val_accuracy"], label="Fine-tune Validation Accuracy")
plt.legend()
plt.title("Fine-tuning Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("results/fine_tuning_accuracy.png", dpi=300)
plt.show()

print("Saved: results/fine_tuning_accuracy.png")


# -------------------------------------------------------------------------------------------
# Fine-tuning loss графигі

plt.figure(figsize=(8, 6))
plt.plot(fine_tune_history.history["loss"], label="Fine-tune Train Loss")
plt.plot(fine_tune_history.history["val_loss"], label="Fine-tune Validation Loss")
plt.legend()
plt.title("Fine-tuning Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("results/fine_tuning_loss.png", dpi=300)
plt.show()

print("Saved: results/fine_tuning_loss.png")


# -------------------------------------------------------------------------------------------
# Final Evaluation

test_loss, test_accuracy = model.evaluate(test_data)

print("Final Test Loss:", test_loss)
print("Final Test Accuracy:", test_accuracy)
print("Class indices:", train_data.class_indices)


# -------------------------------------------------------------------------------------------
# Prediction жасау

test_data.reset()

y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = test_data.classes

class_names = list(test_data.class_indices.keys())


# -------------------------------------------------------------------------------------------
# Classification Report

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

print("\nClassification Report:")
print(report_df)

report_df.to_csv("results/classification_report.csv")

plt.figure(figsize=(10, 6))
sns.heatmap(
    report_df.iloc[:-3, :-1],
    annot=True,
    cmap="Blues",
    fmt=".2f"
)
plt.title("Classification Report")
plt.tight_layout()
plt.savefig("results/classification_report.png", dpi=300)
plt.show()

print("Saved: results/classification_report.csv")
print("Saved: results/classification_report.png")


# -------------------------------------------------------------------------------------------
# Confusion Matrix

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()

print("Saved: results/confusion_matrix.png")


# -------------------------------------------------------------------------------------------
# Save model

model.save("fashion_mobilenetv2_model.keras")
print("Model saved as fashion_mobilenetv2_model.keras")


# -------------------------------------------------------------------------------------------
# Final message

print("\nAll results saved successfully!")
print("Check the 'results' folder.")