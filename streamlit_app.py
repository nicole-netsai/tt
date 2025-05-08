# prompt: create a streamlit app for this system

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Load the trained model
model = keras.models.load_model('Campus Parking App.keras')

# Define image dimensions
img_height = 180
img_width = 180

# Define parking lot categories (replace with your actual categories)
data_cat = ['Empty', 'Occupied'] # Example categories, replace with your actual categories

st.title("Campus Parking Availability")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_arr = tf.keras.utils.array_to_img(image, size=(img_height, img_width))
    img_bat = tf.expand_dims(img_arr, 0)

    # Make a prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    predicted_class = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100

    st.write(f"The parking lot in the image is {predicted_class} with a confidence of {confidence:.2f}%")
