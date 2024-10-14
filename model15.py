import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from sklearn.utils import class_weight
import os
from PIL import Image

# Supported formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

# Define the path to your dataset
dataset_directory = r"C:\Users\sminc\OneDrive\Desktop\train_data"

# Function to check image format
def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Verify if it's an actual image
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")
        return True
    except (IOError, ValueError, SyntaxError) as e:
        print(f"Invalid image file '{file_path}': {e}")
        return False

# Function to filter invalid images
def filter_invalid_images(directory):
    invalid_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_valid_image(file_path):
                invalid_images.append(file_path)
    return invalid_images

# Function to convert invalid images to JPEG
def convert_image_to_jpeg(image_path, output_path):
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert('RGB')  # Convert to RGB for saving as JPEG
            rgb_img.save(output_path, format='JPEG')
            print(f"Converted {image_path} to JPEG format.")
    except Exception as e:
        print(f"Failed to convert {image_path}: {e}")

# Perform boundary testing for invalid images
invalid_images = filter_invalid_images(dataset_directory)

if invalid_images:
    print(f"Found {len(invalid_images)} invalid images:")
    for img in invalid_images:
        print(f"Attempting to convert {img} to JPEG format.")
        new_path = os.path.splitext(img)[0] + ".jpg"
        convert_image_to_jpeg(img, new_path)
else:
    print("All images are in valid formats or have been successfully converted.")

# Handle empty dataset directory
if len(os.listdir(dataset_directory)) == 0:
    raise ValueError(f"The dataset directory '{dataset_directory}' is empty or contains no valid images.")

# Load the dataset (no augmentation yet)
try:
    raw_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_directory,
        image_size=(224, 224),
        batch_size=32,
        label_mode='int',
        validation_split=0.2,
        subset='training',
        seed=123
    )
except Exception as e:
    raise ValueError(f"Failed to load the dataset: {e}")

# Get class names before applying the map function
class_names = raw_train_dataset.class_names
print(f"Class names: {class_names}")

# Class distribution analysis
class_counts = {i: 0 for i in range(len(class_names))}
for _, labels in raw_train_dataset:
    for label in labels.numpy():
        class_counts[label] += 1
print("Class distribution:")
for i, class_name in enumerate(class_names):
    print(f"Class {class_name} has {class_counts[i]} samples.")

# Define augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# Apply augmentation
def apply_augmentation(image, label):
    image = data_augmentation(image)
    return image, label

train_dataset = raw_train_dataset.map(apply_augmentation)

# Validation dataset
try:
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_directory,
        image_size=(224, 224),
        batch_size=32,
        label_mode='int',
        validation_split=0.2,
        subset='validation',
        seed=123
    )
except Exception as e:
    raise ValueError(f"Failed to load the validation dataset: {e}")

# Calculate class weights
y_train = np.concatenate([y.numpy() for _, y in raw_train_dataset])
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Adjust class weights
class_weights[0] = class_weights[0] * 1.5
class_weights = dict(enumerate(class_weights))

print(f"Class weights: {class_weights}")

# Load MobileNetV2
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Unfreeze some layers
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Define model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=50,
    class_weight=class_weights
)

# Load and preprocess an image for prediction
def load_and_preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise ValueError(f"Image path does not exist: {image_path}")
    
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {ext}")
    
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict image class with validation
def predict_image(image_path):
    try:
        preprocessed_image = load_and_preprocess_image(image_path)
        predictions = model.predict(preprocessed_image)
        
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class_name = class_names[predicted_class[0]]
        
        print(f"Predicted class: {predicted_class[0]} ({predicted_class_name})")
        print(f"Prediction confidence for each class: {predictions[0]}")
    except ValueError as e:
        print(f"Error: {e}")

# Continuous user input for image prediction
while True:
    image_path = input("Enter the path to your image (or 'exit' to quit): ").strip()
    
    if image_path.lower() == 'exit':
        print("Exiting prediction loop.")
        break

    if not os.path.exists(image_path):
        print(f"Invalid path: {image_path}")
    else:
        predict_image(image_path)
