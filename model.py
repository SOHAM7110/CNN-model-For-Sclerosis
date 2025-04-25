#run this in you terminal before you ru the program : chcp 65001

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.preprocessing import image
import numpy as np

from tensorflow.keras.saving import save_model

# Path to your dataset
dataset_path = 'C:/Users/SOHAM/Desktop/Scelrosis Detection using CNN model/data'

# Create an ImageDataGenerator for loading the data
datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]

# Load images from the folder
data_flow = datagen.flow_from_directory(
    dataset_path,              # Path to the dataset
    target_size=(150, 150),     # Resize images to 150x150
    batch_size=32,              # Number of images per batch
    class_mode='categorical',   # 'categorical' for multi-class classification
    shuffle=True                # Shuffle the images
)

# Display class labels
print("Class Labels:", data_flow.class_indices)

"""------------------------------------------------------------------------------------------------------------------Model"""

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # First Conv Layer
    MaxPooling2D((2, 2)),  # Max Pooling Layer

    Conv2D(64, (3, 3), activation='relu'),  # Second Conv Layer
    MaxPooling2D((2, 2)),  # Max Pooling Layer

    Conv2D(128, (3, 3), activation='relu'),  # Third Conv Layer
    MaxPooling2D((2, 2)),  # Max Pooling Layer

    Flatten(),  # Flatten the 3D output to 1D

    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout to prevent overfitting
    
    Dense(len(data_flow.class_indices), activation='softmax')  # Output layer with softmax
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
# model.summary()

"""------------------------------------------------------------------------------------------------------------------train Model"""

# Train the model
model.fit(
    data_flow,  # The data generator
    steps_per_epoch=data_flow.samples // data_flow.batch_size,  # Number of steps per epoch
    epochs=10  # Number of epochs
)
from tensorflow.keras.saving import save_model
# model.summary()
# Save the model using keras.saving.save_model()
save_model(model, 'model_1.keras')


# """--------------------------------------------Prediction"""



# # Load the first image from the directory (modify path accordingly)
# img_path = 'C:/Users/SOHAM/Desktop/Major Project/data/1/IM000001.jpg'  # Change this path to the actual image path

# # Load the image with target size (same as model input)
# img = image.load_img(img_path, target_size=(150, 150))

# # Convert the image to a numpy array
# img_array = image.img_to_array(img)

# # Expand the dimensions to match the model's input shape (batch_size, height, width, channels)
# img_array = np.expand_dims(img_array, axis=0)

# # Rescale the image (same preprocessing as during training)
# img_array = img_array / 255.0



# # Predict the class probabilities for the image
# predictions = model.predict(img_array)

# # Get the index of the highest probability class
# predicted_class_index = np.argmax(predictions, axis=1)

# # Map the index to the class name using class_indices
# class_names = list(data_flow.class_indices.keys())
# predicted_class_name = class_names[predicted_class_index[0]]

# # Output the predicted class name
# print(f'Predicted Class: {predicted_class_name}')

