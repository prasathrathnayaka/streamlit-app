Image Classification with Streamlit

This project is a web application built using Streamlit that allows users to upload an image and classify it using the MobileNetV2 model, which is pre-trained on the ImageNet dataset. The app includes features such as setting a confidence threshold, displaying the most probable prediction, and visualizing the confidence levels of predictions.

Features

Image Upload: Users can upload an image in JPG, JPEG, or PNG format.
Image Classification: The app uses the MobileNetV2 model to classify the uploaded image.
Confidence Threshold: Users can adjust the confidence threshold to filter predictions.
Visualization: The app visualizes the confidence levels of the bottom 5 predictions using a horizontal bar chart.

Requirements

To run this application, you need the following Python libraries:

streamlit: For building the web application.
PIL (Pillow): For image processing.
tensorflow: For loading and using the MobileNetV2 model.
numpy: For handling numerical operations.
matplotlib: For creating visualizations.

How It Works

Upload an Image: The uploaded image is resized and processed.
Classify: Predictions are made using MobileNetV2.
Filter: Predictions are filtered by the selected confidence threshold.
Display: The most probable prediction is shown, and confidence levels are visualized.

App Components

Title: The app's title, "Image Classification with Streamlit", is displayed at the top.

Sidebar: A sidebar is provided for users to adjust model parameters, specifically the confidence threshold for filtering predictions.

Image Upload: Users can upload an image using the st.file_uploader widget. The app accepts images in JPG, JPEG, or PNG format.

Image Display: Once an image is uploaded, it is displayed on the main page using st.image.

Classification Process:

The uploaded image is resized to 224x224 pixels to match the input size expected by MobileNetV2.
The image is preprocessed using preprocess_input before feeding it into the model for prediction.
A progress bar is shown to simulate a longer processing time.
Displaying Predictions:

The app uses the decode_predictions function to convert the model's raw predictions into human-readable labels.
Predictions are filtered based on the user-defined confidence threshold.
The most probable prediction is displayed on the screen.
Visualization:

If predictions are available and meet the confidence threshold, the bottom 5 predictions are visualized using a horizontal bar chart created with Matplotlib.
The bar chart displays the probability percentages for each prediction.
Container for Graphs
The app includes a container where the confidence levels of the bottom 5 predictions are visualized. This container will only display content if an image is uploaded and predictions are available after applying the confidence threshold.
