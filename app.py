
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFilter

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("handwritten_digit_model.keras")

model = load_model()

st.title("Handwritten Digit Recognition")
st.write("Upload a clear image of one handwritten digit (0–9).")

uploaded_file = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Original image", width=250)

    
if st.button("Recognize Digit"):
    # Read the uploaded image
    image = Image.open(uploaded_file).convert("L")

    # Convert to a NumPy array
    arr = np.array(image)

    # Convert to white digit on black background.
    # MNIST uses bright digits on a dark background.
    if arr.mean() > 127:
        arr = 255 - arr

    # Find the digit pixels
    mask = arr > 30

    if not mask.any():
        st.error("No clear digit found. Upload a darker, clearer digit.")
    else:
        ys, xs = np.where(mask)

        # Crop closely around the digit
        cropped = arr[
            ys.min():ys.max() + 1,
            xs.min():xs.max() + 1
        ]

        # Preserve aspect ratio and fit inside 20x20
        digit_image = Image.fromarray(cropped.astype("uint8"))
        digit_image.thumbnail((20, 20), Image.Resampling.LANCZOS)

        # Center on a 28x28 black canvas
        canvas = Image.new("L", (28, 28), 0)
        x = (28 - digit_image.width) // 2
        y = (28 - digit_image.height) // 2
        canvas.paste(digit_image, (x, y))

        # Normalize exactly like the training images
        model_input = np.array(canvas).astype("float32") / 255.0
        model_input = model_input.reshape(1, 28, 28, 1)

        # Display the actual input to the model
        st.image(
            canvas,
            caption="Processed image sent to CNN",
            width=200
        )

        # Predict
        probabilities = model.predict(model_input, verbose=0)[0]
        digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100

        st.success(f"Predicted digit: {digit}")
        st.metric("Confidence", f"{confidence:.2f}%")

        st.bar_chart({
            str(i): float(probabilities[i])
            for i in range(10)
        })

