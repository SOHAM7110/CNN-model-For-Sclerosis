# paste these to line in terminal before you run this code
# set PYTHONIOENCODING=utf-8
# python your_script.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set path to your test dataset
test_data_path = './data'

# Load Test Data
input_shape = (150, 150)
batch_size = 32

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_path,
    labels='inferred',
    label_mode='categorical',  # Important for One-Hot
    image_size=input_shape,
    batch_size=batch_size,
    shuffle=False
)

# Class Labels
print("Class Labels:", test_dataset.class_names)

# Load Model
model = tf.keras.models.load_model('model_1.keras')

# Evaluate Model
loss, accuracy = model.evaluate(test_dataset)
print(f'Model Accuracy on Test Data: {accuracy * 100:.2f}%')

# Predictions
y_pred_probs = model.predict(test_dataset)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# True Labels → Fix Here
y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])
y_true = np.argmax(y_true, axis=1)   # Convert One-Hot to Integers

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_labels, target_names=test_dataset.class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_labels)
plt.figure(figsize=(5, 6))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=test_dataset.class_names, yticklabels=test_dataset.class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
