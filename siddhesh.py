import os 
import shutil
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the pre-trained model
HybridCnn = load_model('./skin_cancerr.h5')  # Adjust the path to your model

# Define class labels
classes = {
    4: 'Nevus',
    6: 'Melanoma',
    2: 'Seborrheic Keratosis',
    1: 'Basal Cell Carcinoma',
    5: 'Vascular Lesion',
    0: 'Actinic Keratosis',
    3: 'Dermatofibroma',
    7: 'NORMAL CLASS'
}

# Image preprocessing function
def preprocess_image(image, target_size=(28, 28)):
    # Convert RGBA images to RGB by removing the alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Convert grayscale images (L mode) to RGB by repeating the grayscale values across 3 channels
    elif image.mode == 'L':  # 'L' mode is for grayscale images
        image = image.convert('RGB')
        
    image = image.resize(target_size)  # Resize to the target size expected by the model
    image_array = np.array(image) / 255.0  # Normalize pixel values
    if image_array.shape[-1] != 3:  # Ensure the image has 3 color channels (RGB)
        raise ValueError("Input image must have 3 color channels (RGB).")
    
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Find and store images per class
def organize_images(input_folder, output_folder, images_per_class=5):
    # Create a counter for each class to keep track of saved images
    class_count = {class_name: 0 for class_name in classes.values()}

    # Create output folders for each class if they don't exist
    for class_name in classes.values():
        class_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # Preprocess and predict class
            try:
                img_array = preprocess_image(img)
                prediction = HybridCnn.predict(img_array)  # Changed from DermaNet to HybridCnn
                pred_label = np.argmax(prediction, axis=1)[0]
                pred_class = classes[pred_label]
            except ValueError as e:
                print(f"Skipping image {filename}: {e}")
                continue

            # Check if the class already has the required number of images
            if class_count[pred_class] < images_per_class:
                # Save the image to the corresponding class folder
                dest_path = os.path.join(output_folder, pred_class, filename)
                shutil.copy(img_path, dest_path)
                class_count[pred_class] += 1

            # Stop if all classes have the required number of images
            if all(count >= images_per_class for count in class_count.values()):
                print("Collected required images for all classes.")
                break

    print("Image organization completed.")

# Paths and configuration
input_folder = './HAM10000_images_part_1'  # Folder containing images to classify
output_folder = './organized_images2'  # Folder to store organized images

# Call the function to organize images
organize_images(input_folder, output_folder)
