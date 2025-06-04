"""Streamlit app for product classification."""

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Product Classifier", layout="wide")


def load_model_and_labels(model_path: Path, data_dir: Path):
    """Load trained model and class labels."""
    try:
        model = tf.keras.models.load_model(model_path)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir, image_size=(224, 224), batch_size=32
        )
        class_names = train_ds.class_names
        logger.info("Model and labels loaded successfully")
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        logger.error(f"Load failed: {e}")
        return None, None


def predict_image(model, image, class_names):
    """Predict class for uploaded image."""
    img = cv2.resize(image, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = prediction[np.argmax(prediction)]
    return predicted_class, confidence


def main():
    """Run the Streamlit app."""
    st.title("Product Image Classifier")
    st.markdown("Upload an image to classify the product.")

    model_path = Path("output/model.h5")
    data_dir = Path("data/raw")
    model, class_names = load_model_and_labels(model_path, data_dir)
    if model is None:
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file:
        image = cv2.imdecode(
            np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR
        )
        st.image(image, channels="BGR", caption="Uploaded Image")
        predicted_class, confidence = predict_image(model, image, class_names)
        st.write(f"Prediction: **{predicted_class}** (Confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
