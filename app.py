
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Page settings
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "handwritten_digit_model.keras"
    )

model = load_model()

# App heading
st.title("Handwritten Digit Recognition")
st.write(
    "Upload an image of a handwritten digit "
    "and let the neural network predict the number."
)

st.info("Upload a clear image containing one digit (0–9).")

# Image upload
uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.subheader("Uploaded image")
    st.image(image, width=250)

    if st.button("Recognize Digit"):
        # MNIST images have light backgrounds and dark digits.
        # Invert images with predominantly dark backgrounds.
        image_array = np.array(image)
        if image_array.mean() < 127:
            image = ImageOps.invert(image)

        # Resize to 28 x 28 pixels
        image = image.resize((28, 28))

        # Normalize pixel values
        image_array = np.array(image).astype("float32") / 255.0

        # Add batch dimension: (1, 28, 28)
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        probabilities = model.predict(
            image_array, verbose=0
        )[0]

        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100

        st.success(
            f"Predicted Digit: {predicted_digit}"
        )
        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Probabilities for each digit")
        st.bar_chart({
            str(i): float(probabilities[i])
            for i in range(10)
        })

st.caption(
    "Built using Python, TensorFlow, Keras and Streamlit."
)

