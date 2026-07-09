import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


st.set_page_config(
    page_title="Fashion Product Image Classification",
    page_icon="🧥",
    layout="centered"
)

st.title("Fashion Product Image Classification")
st.write("Upload a fashion product image and the model will classify it into one of four categories.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fashion_mobilenetv2_model.keras")

class_names = [
    "Accessories",
    "Apparel",
    "Footwear",
    "Personal Care"
]

img_size = (224, 224)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded image", use_container_width=True)

    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.subheader("Prediction Result")
    st.write(f"**Predicted class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.progress(float(confidence / 100))

    st.subheader("Class probabilities")

    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i] * 100:.2f}%")