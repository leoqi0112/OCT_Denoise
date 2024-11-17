import os
import numpy as np
from PIL import Image

# Define the folder containing the images
input_folder = "Denoising_Input_Folder"

# Create an empty list to store the loaded image arrays
image_arrays = []

# Loop through all files in the folder
for filename in os.listdir(input_folder):
    # Construct the full file path
    file_path = os.path.join(input_folder, filename)
    
    # Check if it's an image (optional: filter by extension)
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Load the image
        image = Image.open(file_path)
        
        # Convert the image to a NumPy array
        image_array = np.array(image)
        
        # Append the array to the list
        image_arrays.append(image_array)

# Convert the list of arrays to a single NumPy array if all images are of the same size
try:
    # Stack images into a single array (num_images, height, width, channels)
    image_arrays = np.array(image_arrays)
except ValueError:
    # If images have different sizes, they cannot be stacked directly
    print("Warning: Images have different sizes and cannot be stacked into a single NumPy array.")
    pass

# Save the array as an NPY file
output_file = "images_dataset.npy"
np.save(output_file, image_arrays)

# Confirmation
print(f"Saved all images to {output_file}.")
