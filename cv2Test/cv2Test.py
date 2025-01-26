import cv2
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def calculate_average_thickness(top_boundary, lower_boundary):
    """
    Calculates the average vertical thickness between top and bottom boundaries.
    Assumes both boundaries have the same length and aligned x-coordinates.
    """
    if top_boundary.shape != lower_boundary.shape:
        raise ValueError("Boundaries must have the same shape to calculate thickness.")

    thicknesses = lower_boundary[:,1] - top_boundary[:,1]
    return np.mean(thicknesses)

def pixels_to_micrometers(img, height, width, measurement):
    """
    Convert a 'measurement' in pixels to micrometers based on a scale in the bottom row.
    Adjust the row index if your scale is located elsewhere.
    """
    pixels_per_100_micrometers = 0
    # Example: scanning row = (height - 1 - 67)
    for i in range(width - 1, -1, -1):
        if img[height - 1 - 67, i] >= 240:
            pixels_per_100_micrometers += 1

    if pixels_per_100_micrometers == 0:
        return 0.0

    # measurement is in pixels; scale it to micrometers
    return (float(measurement) / float(pixels_per_100_micrometers)) * 100.0

def reduce_spikes(boundary, max_diff=5):
    smoothed_boundary = [boundary[0]]
    for i in range(1, len(boundary)):
        prev_point = smoothed_boundary[-1]
        curr_point = boundary[i].copy()

        if abs(curr_point[1] - prev_point[1]) > max_diff:
            direction = np.sign(curr_point[1] - prev_point[1])
            curr_point[1] = prev_point[1] + direction * max_diff

        smoothed_boundary.append(curr_point)
    return np.array(smoothed_boundary, dtype=boundary.dtype)

def smooth_boundary(boundary_points, window_size=11, poly_order=2):
    xs = boundary_points[:,0]
    ys = boundary_points[:,1]
    # Ensure window size is valid
    if len(ys) < window_size:
        # Make window size smaller if needed
        window_size = len(ys) if (len(ys) % 2 == 1) else (len(ys) - 1)
    ys_smooth = savgol_filter(ys, window_length=max(3, window_size), polyorder=poly_order)
    return np.column_stack((xs, ys_smooth))

def find_subtle_boundary(img, top_boundary, offset=25, search_depth=50, gradient_thresh=10):
    """
    Find a subtle lower boundary starting at least 'offset' pixels below top_boundary
    by searching for strong vertical gradient.
    """
    h, w = img.shape
    lower_boundary = []
    # Slight blur to reduce noise
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_y_abs = cv2.convertScaleAbs(grad_y)

    for (x, y_top) in top_boundary:
        x_int = int(x)
        y_start = int(min(y_top + offset, h-1))
        y_end = min(y_start + search_depth, h-1)

        column_grad = grad_y_abs[y_start:y_end, x_int]
        candidates = np.where(column_grad > gradient_thresh)[0]
        if len(candidates) > 0:
            boundary_y = y_start + candidates[0]
        else:
            boundary_y = y_end

        lower_boundary.append([x_int, boundary_y])

    lower_boundary = np.array(lower_boundary)
    lower_boundary_smooth = smooth_boundary(lower_boundary, window_size=15, poly_order=2)
    return lower_boundary_smooth

def extract_top_boundary_from_mask(mask):
    """
    For each column in the mask, find the topmost (smallest y) white pixel.
    Returns Nx2 array of [x, y].
    """
    h, w = mask.shape
    boundary_points = []
    for x in range(w):
        col = mask[:, x]
        whites = np.where(col > 0)[0]
        if len(whites) > 0:
            boundary_points.append([x, whites[0]])
    return np.array(boundary_points, dtype=float)

def generate_lower_bound(img, top_boundary, thresh_vals):
    """
    Generate a list of lower boundaries, one for each gradient threshold.
    """
    boundaries = []
    for val in thresh_vals:
        boundaries.append(find_subtle_boundary(
            img, top_boundary, offset=25, search_depth=40, gradient_thresh=val
        ))
    return boundaries

def process_images_in_folder(input_folder, output_base_folder):
    """
    1) Iterates over all images in 'input_folder'.
    2) For each image, creates a subfolder in 'output_base_folder' with the same base name.
    3) Processes thresholds [60, 65, 70, 75, 80], saves 5 overlays & 1 Excel for that image.
    """

    #os.makedirs(output_base_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(len(image_files))

    # The thresholds we want to test
    thresh_vals = [60, 65, 70, 75, 80]

    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load image {image_file}, skipping.")
            continue
        
        height, width = img.shape[:2]

        # Create a subfolder for this image
        base_name = os.path.splitext(image_file)[0]
        image_output_folder = os.path.join(output_base_folder, base_name)
        os.makedirs(image_output_folder, exist_ok=True)

        # 1) Preprocess + find largest contour
        blurred = cv2.GaussianBlur(img, (3,3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"No contours found in {image_file}, skipping.")
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        # Draw filled largest contour on a mask
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Extract + smooth top boundary
        top_boundary = extract_top_boundary_from_mask(mask)
        if len(top_boundary) < 2:
            print(f"Not enough points in top boundary for {image_file}, skipping.")
            continue
        top_boundary_smooth = smooth_boundary(top_boundary, window_size=15, poly_order=2)

        # Generate lower boundaries for each threshold
        lower_boundaries = generate_lower_bound(img, top_boundary_smooth, thresh_vals)

        # We'll store the results for this image in a list,
        # then convert to Excel at the end
        results_for_image = []

        for i, tval in enumerate(thresh_vals):
            lower_boundary = lower_boundaries[i]
            # Remove spikes + final smooth
            lower_boundary_re = reduce_spikes(lower_boundary)
            lower_boundary_smooth = smooth_boundary(lower_boundary_re, window_size=27, poly_order=2)

            # Calculate average thickness (in pixels)
            thickness_in_pixels = calculate_average_thickness(top_boundary_smooth, lower_boundary_smooth)
            # Convert to micrometers based on your scale
            thickness_in_micrometers = pixels_to_micrometers(img, height, width, thickness_in_pixels)

            # Add row for Excel
            results_for_image.append([
                image_file,       # Image Name
                tval,             # Threshold
                thickness_in_micrometers
            ])

            # Create overlay for visualization
            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Draw top boundary (green)
            for (x,y) in top_boundary_smooth:
                cv2.circle(overlay, (int(x), int(y)), 1, (0,255,0), -1)
            # Draw lower boundary (red)
            for (x,y) in lower_boundary_smooth:
                cv2.circle(overlay, (int(x), int(y)), 1, (0,0,255), -1)
        
            # Save overlay
            overlay_filename = f"{base_name}_overlay_{tval}.png"
            overlay_path = os.path.join(image_output_folder, overlay_filename)
            cv2.imwrite(overlay_path, overlay)

        # Now save the Excel for just this image
        df = pd.DataFrame(results_for_image, columns=["Image Name", "Threshold", "Thickness_micrometers"])
        excel_filename = f"{base_name}_thickness_summary.xlsx"
        excel_path = os.path.join(image_output_folder, excel_filename)
        df.to_excel(excel_path, index=False)

        print(f"Finished processing {image_file}")
        print(f"Saved 5 overlays + Excel to: {image_output_folder}\n")

if __name__ == "__main__":
    # Example usage
    input_folder = r'D:\Graza Lab\OCT_Denoise\OCT_Denoise\cv2Test\Input_Folder'               # TODO: Set to your input folder
    output_base_folder = r'D:\Graza Lab\OCT_Denoise\OCT_Denoise\cv2Test\Output_Folder'      # TODO: Set to your output folder
    process_images_in_folder(input_folder, output_base_folder)


# import cv2
# import numpy as np

# # Load the original image
# img = cv2.imread('./Test_Images/denoising.jpg', cv2.IMREAD_GRAYSCALE)

# # Preprocessing - CLAHE to enhance contrast
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# enhanced = clahe.apply(img)

# # Denoise to reduce speckle
# denoised = cv2.GaussianBlur(enhanced, (3,3), 0)

# # Example 1: Canny Edge with multiple thresholds
# canny_params = [(20, 40), (30, 60), (40, 80), (50, 100)]
# canny_edges = []
# for (low, high) in canny_params:
#     edges = cv2.Canny(denoised, low, high)
#     canny_edges.append((f"Canny-{low}-{high}", edges))

# # Example 2: Sobel gradient based
# sobelx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
# gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
# gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

# # Adaptive threshold on gradient
# th = cv2.medianBlur(gradient_magnitude, 5)
# ret, grad_edges = cv2.threshold(th, 30, 255, cv2.THRESH_BINARY)

# canny_edges.append(("Sobel_Thresholded", grad_edges))

# # Display candidates
# for name, edge_map in canny_edges:
#     overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     overlay[edge_map > 0] = (0,0,255) # red edges
#     cv2.imshow(name, overlay)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # The user visually inspects these and chooses the best segmentation method.
# # Suppose the user chooses the Sobel_Thresholded result. Then you can proceed:
# selected_edges = grad_edges

# # Post-process selected edges (optional)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# closed = cv2.morphologyEx(selected_edges, cv2.MORPH_CLOSE, kernel)

# # Extract contours
# contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Here you would define logic to pick the contour that best matches the subtle lower boundary.
# For example, you might pick contours that run roughly parallel to the top boundary and are located 
# at a certain mean depth in the image.

# [Additional logic needed depending on domain knowledge and user input]
