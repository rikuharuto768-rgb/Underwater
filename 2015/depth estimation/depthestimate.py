 
import cv2
import numpy as np
import os

def estimate_blurriness_map(I, n=4):
    """
    Compute blurriness map B as per Eq. (6):
    B(x) = (1/n) * sum_{i=1}^{n} |I(x) - G_{r_i, r_i}(x)|
    where r_i = 2*i + 1, i.e., r = [3,5,7,9] for n=4.
    """
    I_float = I.astype(np.float32)
    B = np.zeros_like(I_float)
    for i in range(1, n + 1):
        r = 2 * i + 1  # kernel size: 3,5,7,9
        sigma = r / 6.0  # standard heuristic: sigma ≈ r/6
        blurred = cv2.GaussianBlur(I_float, (r, r), sigma, borderType=cv2.BORDER_DEFAULT)
        B += np.abs(I_float - blurred)
    return B / n

def generate_rough_depth_map(B, z=7):
    """
    Generate rough depth map t̂ via max filtering over z×z local patch (Eq. 7).
    Larger local patch size => smoother but less precise depth edges.
    """
    return cv2.dilate(B, np.ones((z, z), np.float32), iterations=1)

def hole_fill_and_refine(rough_depth, guidance_image, radius=3, eps=1e-4):
    """
    Refine depth map:
    1. Identify and fill holes (flat/ambiguous regions) using inpainting.
    2. Apply guided filter for edge-preserving smoothing [14].
    """
    # Step 1: hole filling (simplified CMR via mask-based inpainting)
    # Flat regions often have low variance → mark as mask
    grad = cv2.Sobel(rough_depth, cv2.CV_32F, 0, 1, ksize=3) + cv2.Sobel(rough_depth, cv2.CV_32F, 1, 0, ksize=3)
    mask = (grad < 1e-4).astype(np.uint8)  # edges = 0, flat = 1 → invert
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8))  # grow mask for safety
    
    # Inpaint (fallback if no real holes, but robust)
    if mask.sum() > 0:
        filled = cv2.inpaint(rough_depth.astype(np.float32), mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        filled = rough_depth.copy()
    
    # Step 2: Guided filter smoothing [14]
    # Try OpenCV ximgproc (if available), else use Gaussian smoothing
    try:
        from cv2 import ximgproc
        refined = ximgproc.guidedFilter(guidance_image, filled.astype(np.float32), radius=radius, eps=eps)
    except (ImportError, AttributeError):
        print("[WARNING] cv2.ximgproc not available; falling back to Gaussian smoothing.")
        sigma = 3.0
        refined = cv2.GaussianBlur(filled, (0, 0), sigma, borderType=cv2.BORDER_DEFAULT)
    return refined

def depth_estimation_pipeline(image_path, output_dir="output"):
    """
    Full depth estimation pipeline for a single image.
    Returns: refined depth map (normalized to [0,255], uint8).
    """
    # Load image
    I = cv2.imread(image_path)
    if I is None:
        raise FileNotFoundError(f"Image at {image_path} not found or cannot be loaded.")
    
    # Ensure RGB for blurriness (grayscale or BGR → LAB for better performance?)
    # Paper works on single-channel; here we use luminance if RGB, else assume grayscale
    if len(I.shape) == 3:
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        gray = I.astype(np.float32) / 255.0

    os.makedirs(output_dir, exist_ok=True)

    # 1. Blurriness map B
    B = estimate_blurriness_map(gray)
    cv2.imwrite(f"{output_dir}/blurriness.png", np.clip(B * 255, 0, 255).astype(np.uint8))

    # 2. Rough depth map t̂
    t_hat = generate_rough_depth_map(B, z=7)
    cv2.imwrite(f"{output_dir}/rough_depth.png", np.clip(t_hat * 255, 0, 255).astype(np.uint8))

    # 3. Refined depth map
    guidance = (gray * 255).astype(np.uint8)  # guidance image (original intensity)
    t_refined = hole_fill_and_refine(t_hat, guidance)
    t_refined_norm = cv2.normalize(t_refined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/refined_depth.png", t_refined_norm)

    return t_refined_norm

# Example usage
if __name__ == "__main__":
    img_path = "3_img_.png"  # replace with your image path
    depth = depth_estimation_pipeline(img_path)
    print(f"Refined depth map saved (shape: {depth.shape})")