import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------------------------------------
# Model және image path

model_path = "fashion_mobilenetv2_model.keras"
img_path = "dataset/test/Accessories/4716.jpg"

img_size = (224, 224)

class_names = [
    "Accessories",
    "Apparel",
    "Footwear",
    "Personal Care"
]

# ------------------------------------------------------------
# Дайын модельді жүктеу

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully!")

# ------------------------------------------------------------
# Суретті дайындау

img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# ------------------------------------------------------------
# Prediction

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]
confidence = prediction[0][predicted_index] * 100

print("Image:", img_path)
print("Predicted class:", predicted_class)
print(f"Confidence: {confidence:.2f}%")

# ------------------------------------------------------------
# Grad-CAM функциясы

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    if max_value == 0:
        return heatmap.numpy()

    heatmap = heatmap / max_value

    return heatmap.numpy()

# ------------------------------------------------------------
# Grad-CAM жасау

last_conv_layer_name = "Conv_1"

heatmap = make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    predicted_index
)

# ------------------------------------------------------------
# Heatmap-ты original image үстіне қою

original_img = cv2.imread(img_path)

if original_img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)


original_img = cv2.resize(original_img, img_size)

heatmap_resized = cv2.resize(heatmap, img_size)
heatmap_resized = np.uint8(255 * heatmap_resized)

heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

alpha = 0.4
gradcam_img = heatmap_color * alpha + original_img
gradcam_img = np.uint8(gradcam_img)

# ------------------------------------------------------------
# Нәтижені көрсету

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gradcam_img)
plt.title(f"Grad-CAM: {predicted_class} ({confidence:.2f}%)")
plt.axis("off")

os.makedirs("results", exist_ok=True)

plt.tight_layout()
plt.savefig("results/gradcam_example.png", dpi=300)
plt.show()

print("Grad-CAM saved: results/gradcam_example.png")