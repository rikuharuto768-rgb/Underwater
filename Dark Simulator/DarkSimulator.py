import cv2
import numpy as np
import random

def simulate_underwater(image_path, output_path="simulated_output.jpg", dark_mode=True):
    # Load the high-quality test image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not find image.")
        return

    rows, cols, _ = img.shape

    # --- 1. Color Cast & Selective Attenuation ---
    # Simulates the absorption of red light.
    # In dark mode, we drastically cut red and suppress green.
    b, g, r = cv2.split(img.astype(np.float32))
    if dark_mode:
        r *= 0.15  # Heavy red loss
        g *= 0.60  # Moderate green loss
        b *= 0.90  # Blue persists
    else:
        r *= 0.40
        g *= 0.90
    
    img_colored = cv2.merge([b, g, r])
    img_colored = np.clip(img_colored, 0, 255).astype(np.uint8)

    # --- 2. Underwater Haze (Dispersion) & Darkness ---
    # Adds turbidity and simulates the "veiling light" that breaks Dark Channel Priors.
    blur_k = 21 if dark_mode else 11
    haze_intensity = 0.5 if dark_mode else 0.2
    
    img_haze = cv2.GaussianBlur(img_colored, (blur_k, blur_k), 0)
    
    # Create a depth-based darkness overlay (darker at bottom)
    dark_overlay = np.zeros_like(img)
    for i in range(rows):
        # Darkness increases with 'depth' (row index)
        opacity = 0.4 + (0.4 * (i / rows)) if dark_mode else 0.1
        dark_overlay[i, :] = [80, 50, 20] # Murky navy/black base
    
    img_hazed = cv2.addWeighted(img_haze, 1 - haze_intensity, dark_overlay, haze_intensity, 0)

    # --- 3. Surface Refraction & Ripples ---
    # Spatial distortion using sine waves.
    map_x = np.zeros((rows, cols), np.float32)
    map_y = np.zeros((rows, cols), np.float32)
    
    # Parameters for the wavy effect
    amplitude = 6
    frequency = 30
    
    for i in range(rows):
        for j in range(cols):
            map_x[i, j] = j + amplitude * np.sin(i / frequency)
            map_y[i, j] = i + amplitude * np.cos(j / frequency)
            
    img_rippled = cv2.remap(img_hazed, map_x, map_y, cv2.INTER_LINEAR)

    # --- 4. Caustics & Artificial Artifacts ---
    # Simulates light rays and the noise amplification mentioned in your text.
    caustics = np.zeros((rows, cols), np.uint8)
    cv2.randu(caustics, 0, 255)
    caustics = cv2.threshold(caustics, 250, 255, cv2.THRESH_BINARY)[1]
    caustics = cv2.GaussianBlur(caustics, (15, 15), 0)
    
    # Blend caustics (dimmer in dark mode)
    caustic_layer = cv2.merge([caustics, caustics, caustics])
    alpha = 0.1 if dark_mode else 0.3
    img_final = cv2.addWeighted(img_rippled, 1.0, caustic_layer, alpha, 0)

    # --- 5. Add Sensor Noise (Restoration Failure Simulation) ---
    noise = np.random.normal(0, 8, img_final.shape).astype(np.int16)
    img_final = np.clip(img_final.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, img_final)
    print(f"Simulation complete. Saved to {output_path}")

# Run the simulation on your test image
simulate_underwater("3_img_.png", "dark_simulated_result.jpg", dark_mode=True)