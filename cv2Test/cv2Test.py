import cv2
import numpy as np
from scipy.signal import savgol_filter


def calculate_average_thickness(top_boundary, lower_boundary):
    """
    Calculates the average vertical thickness between top and bottom boundaries.
    Assumes both boundaries have the same length and aligned x-coordinates.

    top_boundary:   Nx2 numpy array of [x, y_top]
    lower_boundary: Nx2 numpy array of [x, y_bottom]

    Returns:
        A float representing the average vertical distance (thickness).
    """
    # Check if both boundaries have the same shape
    if top_boundary.shape != lower_boundary.shape:
        raise ValueError("Boundaries must have the same shape to calculate thickness.")

    # Calculate thickness at each column (difference in y-coordinates)
    thicknesses = lower_boundary[:,1] - top_boundary[:,1]

    # Compute average thickness
    avg_thickness = np.mean(thicknesses)
    return avg_thickness

# Function to reduce spiking by limiting the difference between consecutive points
def reduce_spikes(boundary, max_diff=5):
    """
    Reduces sudden vertical spikes in a top boundary array.
    
    boundary: Nx2 numpy array of [x, y] points.
    max_diff: Maximum allowed difference in vertical direction between consecutive points.
    """
    smoothed_boundary = [boundary[0]]
    for i in range(1, len(boundary)):
        prev_point = smoothed_boundary[-1]
        curr_point = boundary[i].copy()

        # If the vertical difference is too large, clamp it to max_diff
        if abs(curr_point[1] - prev_point[1]) > max_diff:
            direction = np.sign(curr_point[1] - prev_point[1])
            curr_point[1] = prev_point[1] + direction * max_diff

        smoothed_boundary.append(curr_point)

    return np.array(smoothed_boundary, dtype=boundary.dtype)

def smooth_boundary(boundary_points, window_size=11, poly_order=2):
    xs = boundary_points[:,0]
    ys = boundary_points[:,1]
    ys_smooth = savgol_filter(ys, window_length=window_size, polyorder=poly_order)
    return np.column_stack((xs, ys_smooth))

def find_subtle_boundary(img, top_boundary, offset=25, search_depth=50, gradient_thresh=10):
    """
    Find a subtle lower boundary starting at least 'offset' pixels below top_boundary.
    top_boundary: Nx2 array of [x, y_top]
    img: Grayscale image
    offset: minimum offset from top boundary
    search_depth: how far below top boundary to search
    gradient_thresh: threshold on vertical gradient to define a boundary
    """
    h, w = img.shape 
    lower_boundary = [] 

    # Pre-smooth image to reduce noise 
    img_blur = cv2.GaussianBlur(img, (3,3), 0) 
    
    # Compute vertical gradient using Sobel in y-direction
    grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_y_abs = cv2.convertScaleAbs(grad_y)

    for (x, y_top) in top_boundary:
        x_int = int(x)
        y_start = int(min(y_top + offset, h-1))
        y_end = min(y_start + search_depth, h-1)

        column_grad = grad_y_abs[y_start:y_end, x_int]

        # Find first location where gradient surpasses threshold
        candidates = np.where(column_grad > gradient_thresh)[0]
        if len(candidates) > 0:
            boundary_y = y_start + candidates[0]
        else:
            # Fallback: no boundary found, take the bottom of the search region
            boundary_y = y_end

        lower_boundary.append([x_int, boundary_y])

    lower_boundary = np.array(lower_boundary)
    # Smooth the lower boundary
    lower_boundary_smooth = smooth_boundary(lower_boundary, window_size=15, poly_order=2)
    return lower_boundary_smooth

def segment_layer(img, top_boundary, lower_boundary):
    """
    Create a mask for the layer between top_boundary and lower_boundary.
    top_boundary and lower_boundary should be arrays of shape Nx2 with (x, y).
    Assume both have the same set of x-coordinates.
    """
    mask = np.zeros_like(img, dtype=np.uint8)
    for (x_top, y_top), (x_bottom, y_bottom) in zip(top_boundary, lower_boundary):
        x_t = int(x_top)
        y_t = int(y_top)
        y_b = int(y_bottom)
        if y_b > y_t:
            mask[y_t:y_b+1, x_t] = 255
    return mask

def extract_top_boundary_from_mask(mask):
    """
    For each column in the mask, find the topmost white pixel.
    """
    h, w = mask.shape
    top_boundary = []
    for x in range(w):
        column = mask[:, x]
        # Find index of first white pixel
        whites = np.where(column > 0)[0]
        if len(whites) > 0:
            y_top = whites[0]
            top_boundary.append([x, y_top])
        else:
            # No white pixel in this column
            pass
    return np.array(top_boundary, dtype=np.float32)

def generate_lower_bound(img, top_boundary, thresh_vals):
    boundaries = []
    for val in thresh_vals:
        boundaries.append(find_subtle_boundary(img, top_boundary, offset=25, search_depth=40, gradient_thresh=val))

    return boundaries

def main():
    # Load image
    img = cv2.imread('cv2Test\Test_Images\original.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load image.")
        return

    # Preprocess
    blurred = cv2.GaussianBlur(img, (3,3), 0)

    # Threshold (Otsu)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find largest contour
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return
    
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw mask of largest contour
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Extract the top boundary from the mask
    top_boundary = extract_top_boundary_from_mask(mask)
    # Smooth the top boundary (x-values stay the same, only y smoothed)
    top_boundary_smooth = smooth_boundary(top_boundary, window_size=15, poly_order=2)

    # List of thresholds we want to test
    thresh_vals = [60, 65, 70, 75, 80]

    # Generate lower boundaries for each threshold
    lower_boundaries = generate_lower_bound(img, top_boundary, thresh_vals)

    # Loop over each threshold and the corresponding lower boundary
    for i, lower_boundary in enumerate(lower_boundaries):
        # 1) Remove spikes
        lower_boundary_re = reduce_spikes(lower_boundary)
        # 2) Smooth again
        lower_boundary_smooth = smooth_boundary(lower_boundary_re, window_size=27, poly_order=2)

        # 3) Calculate average thickness
        #    We use top_boundary_smooth so that both top/bottom are smoothed.
        avg_thickness = calculate_average_thickness(top_boundary_smooth, lower_boundary_smooth)
        print(f"Threshold = {thresh_vals[i]} => Average thickness: {avg_thickness:.2f} pixels")

        # Create the final segmented layer mask
        layer_mask = segment_layer(img, top_boundary_smooth, lower_boundary_smooth)

        # Visualization (optional display or save)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Mark top boundary (green)
        for (x,y) in top_boundary_smooth:
            cv2.circle(overlay, (int(x), int(y)), 1, (0,255,0), -1)
        # Mark lower boundary (red)
        for (x,y) in lower_boundary_smooth:
            cv2.circle(overlay, (int(x), int(y)), 1, (0,0,255), -1)

        # Save results
        # cv2.imwrite(f"original_{thresh_vals[i]}.png", img)  # Save the original image
        # cv2.imwrite(f"binary_{thresh_vals[i]}.png", binary)  # Save the binary image
        # cv2.imwrite(f"largest_component_mask_{thresh_vals[i]}.png", mask)  # Save the largest component mask
        # cv2.imwrite(f"segmented_layer_mask_{thresh_vals[i]}.png", layer_mask)  # Save the segmented layer mask
        cv2.imwrite(f"overlay_with_boundaries_{thresh_vals[i]}.png", overlay)
        i += 1

        # Display results
        #cv2.imshow('Original', img)
        #cv2.imshow('Binary', binary)
        #cv2.imshow('Largest Component Mask', mask)
        #cv2.imshow('Segmented Layer Mask', layer_mask)
        cv2.imshow('Overlay with Boundaries', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


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
