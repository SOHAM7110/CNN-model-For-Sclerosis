# Importing necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

import sys
sys.stdout.reconfigure(encoding='utf-8')


# Load the saved CNN model
model = load_model('model_1.keras')

# Class names (update based on your dataset classes)
class_names = ['CIS','Multiple Sclerosis','No Lesion', 'PPMS','RRMS','SPMS']  # Example: Binary classification

# Function to predict image
def predict_image(img_path):
    # Load image and preprocess it
    img = image.load_img(img_path, target_size=(150, 150))  # Change (64, 64) if required
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if required (based on training)

    # Predict
    prediction = model.predict(img_array)

    # For multi-class
    predicted_class = np.argmax(prediction[0])

    # print(f"Image: {img_path}")
    print(f"Predicted Class: {class_names[predicted_class]}")
    # print(f"Prediction Probabilities: {prediction}")

# Example usage
folder_path = 'test_images/'  # Folder containing images to predict

for img_file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_file)
    predict_image(img_path)
