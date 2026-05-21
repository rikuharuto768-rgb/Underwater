import cv2
import numpy as np

def generate_multi_blur_depth(image_path):
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    # --- Calculation 1: Local Variance Blur (Statistical) ---
    mean = cv2.boxFilter(gray_f, -1, (15, 15))
    sq_mean = cv2.boxFilter(gray_f**2, -1, (15, 15))
    variance = np.clip(sq_mean - mean**2, 0, None)
    var_map = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Invert so distant/blurry areas are "hotter" or lighter
    depth_var = 255 - var_map

    # --- Calculation 2: Laplacian Blur (Edge Density) ---
    # High values = sharp edges; Low values = blur
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.uint8(np.absolute(laplacian))
    # Smooth the result to create a continuous map
    depth_lap = 255 - cv2.GaussianBlur(laplacian_abs, (25, 25), 0)
    depth_lap = cv2.normalize(depth_lap, None, 0, 255, cv2.NORM_MINMAX)

    # --- Calculation 3: Sobel Gradient Magnitude ---
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.magnitude(sobelx, sobely)
    depth_sobel = 255 - cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- Visualization: Apply Colormaps ---
    # Mapping the grayscale depth to 'COLORMAP_INFERNO' for a heat-map look
    color_var = cv2.applyColorMap(depth_var, cv2.COLORMAP_INFERNO)
    color_lap = cv2.applyColorMap(depth_lap.astype(np.uint8), cv2.COLORMAP_JET)
    color_sobel = cv2.applyColorMap(depth_sobel, cv2.COLORMAP_VIRIDIS)

    # Combine for a dashboard view
    top_row = np.hstack((img, color_var))
    bottom_row = np.hstack((color_lap, color_sobel))
    combined = np.vstack((top_row, bottom_row))

    # Save outputs
    cv2.imwrite('multi_depth_comparison.png', combined)
    print("Multi-calculation maps saved as 'multi_depth_comparison.png'")

# Verbatim file reference
generate_multi_blur_depth("3_img_.png")