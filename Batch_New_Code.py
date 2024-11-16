import cv2
import os
import numpy as np
import pandas as pd
import openpyxl

# Function to reduce spiking by limiting the difference between consecutive points
def reduce_spikes(contour, max_diff=5):
    smoothed_contour = [contour[0]]
    for i in range(3, len(contour)):
        prev_point = smoothed_contour[-1][0]
        curr_point = contour[i][0]
        if abs(curr_point[1] - prev_point[1]) > max_diff:
            curr_point[1] = prev_point[1] + np.sign(curr_point[1] - prev_point[1]) * max_diff
        smoothed_contour.append([curr_point])
    return np.array(smoothed_contour)

# Function to extract red and green contour information and export to Excel
def extract_red_box_info(image, threshold_value, output_folder):
    # Red color detection using thresholding
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([100, 100, 255])
    red_mask = cv2.inRange(image, lower_red, upper_red)

    height, width = red_mask.shape
    top_positions = np.full(width, -1)
    bottom_positions = np.full(width, -1)
    differences = np.full(width, -1)
    image_with_circles = np.copy(image)

    # Find top and bottom positions for each column
    for x in range(width):
        red_column = red_mask[:, x]
        if np.any(red_column):
            top_positions[x] = np.argmax(red_column)
            bottom_positions[x] = height - 1 - np.argmax(red_column[::-1])
            differences[x] = (bottom_positions[x] - top_positions[x]) * 2  # Multiply by two
            if differences[x] > 80:
                cv2.circle(image_with_circles, (x, int(top_positions[x])), 3, (0, 255, 0), -1)  # Green circle for top
                cv2.circle(image_with_circles, (x, int(bottom_positions[x])), 3, (255, 0, 0), -1)  # Blue circle for bottom

    avg_difference = np.mean(differences[differences != -1])
    differences_above_80 = differences[differences > 80]
    avg_diff_above_80 = np.mean(differences_above_80) if len(differences_above_80) > 0 else 0
    count_above_80 = np.sum(differences > 80)
    product_count_avg_diff_above_80 = count_above_80 * avg_diff_above_80

    # Prepare data for Excel export
    data = {
        'Pixel (Width)': range(len(top_positions)),
        'Top Position': top_positions,
        'Bottom Position': bottom_positions,
        'Thickness': differences,
        'Average Thickness': [avg_difference] * len(top_positions),
        'Average Thickness > 80 pixels': [avg_diff_above_80] * len(top_positions),
        'Count > 80': [count_above_80] * len(top_positions),
        'Area': [product_count_avg_diff_above_80] * len(top_positions),
    }

    df = pd.DataFrame(data)
    excel_path = os.path.join(output_folder, f'red_box_info_threshold_{threshold_value}.xlsx')
    df.to_excel(excel_path, index=False)

    # Save the image with circles
    image_with_circles_path = os.path.join(output_folder, f'image_with_circles_threshold_{threshold_value}.png')
    cv2.imwrite(image_with_circles_path, image_with_circles)

    print(f"Information exported to {excel_path}")

    # Return the extracted values for the summary
    return avg_diff_above_80, count_above_80, product_count_avg_diff_above_80

# Main function to process all images in a folder
def process_images_in_folder(input_folder, output_base_folder):
    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        # Load the image
        image = cv2.imread(image_path)

        # Create a subfolder for each image
        folder_name = os.path.splitext(image_file)[0]
        output_folder = os.path.join(output_base_folder, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # Define different threshold values to be used
        threshold_values = [35, 45, 50, 55, 60, 65, 75]

        for threshold_value in threshold_values:
            # Convert the image to grayscale and apply thresholding
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

            # Remove noise from the top and bottom
            height, width = binary_image.shape
            binary_image[:int(height * 0.1), :] = 0
            binary_image[int(height * 0.85):, :] = 0

            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 0.1 * cv2.contourArea(main_contour)]

                # Apply spike reduction
                smoothed_contours = [reduce_spikes(contour) for contour in filtered_contours]

                # Draw smoothed contours
                outlined_image_smoothed = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(outlined_image_smoothed, smoothed_contours, -1, (0, 0, 255), 2)

                # Extract red box information, save to Excel, and get the summary data
                extract_red_box_info(outlined_image_smoothed, threshold_value, output_folder)

        print(f"All threshold images and Excel files saved in folder: {output_folder}")

# Call the function with the input and output folder paths
input_folder = 'Denoising_Input_Folder'
output_base_folder = 'Denoising_Output_Folder'
process_images_in_folder(input_folder, output_base_folder)
