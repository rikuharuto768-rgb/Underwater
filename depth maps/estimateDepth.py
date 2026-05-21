import cv2
import numpy as np

def estimate_underwater_depth(image_path):
    # Load the original image (Fig. 2a / Fig. 3a in image_38dd1a.png)
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return
    
    # Convert to grayscale for blurriness analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0

    # 1. Generate Blurriness Map (Representing Fig. 2b)
    # Using local variance as a measure of sharpness/blur
    mean = cv2.boxFilter(gray, -1, (15, 15))
    sq_mean = cv2.boxFilter(gray**2, -1, (15, 15))
    variance = sq_mean - mean**2
    blur_map = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
    blur_map = 255 - blur_map.astype(np.uint8) # Invert so blurry = distant/white

    # 2. Rough Depth Map (Representing Fig. 2c)
    # Apply a Guided Filter or a large Gaussian blur to smooth the noise
    rough_depth = cv2.GaussianBlur(blur_map, (21, 21), 0)

    # 3. Refined Depth Map (Representing Fig. 2d)
    # Use morphological operations to solidify structures
    kernel = np.ones((5,5), np.uint8)
    refined_depth = cv2.morphologyEx(rough_depth, cv2.MORPH_CLOSE, kernel)
    refined_depth = cv2.equalizeHist(refined_depth)

    # Save outputs for comparison
    cv2.imwrite('blur_map.png', blur_map)
    cv2.imwrite('rough_depth.png', rough_depth)
    cv2.imwrite('refined_depth.png', refined_depth)
    
    print("Depth maps generated successfully.")

# Reference the file verbatim as requested
estimate_underwater_depth("3_img_.png")