import streamlit as st
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import time
import matplotlib.pyplot as plt

# Title of the app
st.title("Image Classification with Streamlit")

# Sidebar for model parameters and options
st.sidebar.header("Model Parameters")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.0)
st.sidebar.write("Adjust the confidence threshold to filter predictions.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image and make predictions
    with st.spinner("Classifying..."):
        model = MobileNetV2(weights="imagenet")
        img_resized = image.resize((224, 224))
        img_array = preprocess_input(np.array(img_resized)[np.newaxis, ...])

        # Simulate a long process for progress bar demonstration
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress.progress(i + 1)

        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=5)[0]  # Get the top 5 predictions

        # Filter predictions by confidence threshold
        filtered_preds = [(label, desc, prob) for (label, desc, prob) in decoded_preds if prob >= confidence_threshold]

        # Display the most probable prediction
        if filtered_preds:
            most_probable_pred = max(filtered_preds, key=lambda x: x[2])
            st.write(f"Most Probable Prediction: **{most_probable_pred[1]}** with a probability of **{most_probable_pred[2] * 100:.2f}%**")
        else:
            st.write("No predictions met the confidence threshold.")

# Container to display graphs and visualizations
with st.container():
    st.header("Model Predictions Visualization")
    if uploaded_file is not None and filtered_preds:
        # Sort predictions by probability (in case filtering changed the order)
        filtered_preds = sorted(filtered_preds, key=lambda x: x[2])
        # Take the last 5 predictions
        labels, probabilities = zip(*[(desc, prob * 100) for (_, desc, prob) in filtered_preds[-5:]])

        fig, ax = plt.subplots()
        ax.barh(labels, probabilities, color='skyblue')
        ax.set_xlabel("Probability (%)")
        ax.set_title("Bottom 5 Predictions Confidence Levels")
        st.pyplot(fig)
    else:
        st.write("Upload an image and ensure predictions are available.")
