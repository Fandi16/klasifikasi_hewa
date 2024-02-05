# app.py
import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the Keras model
model = load_model('models/mdl_wt.hdf5')

# Function to preprocess and make predictions
def predict(image):
    # Resize the image to the required input size of the model
    resized_image = image.resize((224, 224))  # Adjust the size as needed

    # Convert the image to a NumPy array
    image_array = np.array(resized_image)

    # Normalize the pixel values to be between 0 and 1
    normalized_image_array = image_array / 255.0

    # Expand the dimensions to match the input shape expected by the model
    input_data = np.expand_dims(normalized_image_array, axis=0)

    # Make a prediction using the loaded model
    prediction_array = model.predict(input_data)

    return prediction_array

# Streamlit app
def main():
    st.title("Clasification Animal")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image animal...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image ", use_column_width=True)

        # Make prediction using the model
        if st.button("Predict"):
            prediction_array = predict(image)

            # Define class labels
            class_labels = [
                "Dog", 
                "horse",
                "elephant",
                "butterfly", 
                "chicken",
                "cat",
                "cow",
                "spider",
                "squirrel", 
            ]

            # Get the predicted class (index with highest probability)
            predicted_class_index = np.argmax(prediction_array)
            
            # Get the class label from the model
            predicted_class = class_labels[predicted_class_index]

            st.write("Prediction Animal :", predicted_class)
            st.write("Prediction Array :", prediction_array)

if __name__ == "__main__":
    main()
