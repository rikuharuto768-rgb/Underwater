import cv2
import numpy as np
import os
import random

def get_blind_parameters():
    """
    Instead of reading an image, we provide presets for typical 
    underwater scattering (eta) and backscatter (beta) coefficients.
    Ref: Jerlov Water Types and the paper's range [0.1, 1.0].
    """
    # Select a random water type to ensure dataset diversity
    water_types = [
        {'name': 'Clear Blue', 'eta': [0.1, 0.2, 0.5]},    # Low red (R), med green (G), high blue (B)
        {'name': 'Tropical Green', 'eta': [0.2, 0.5, 0.3]},# High green retention
        {'name': 'Murky/Coastal', 'eta': [0.6, 0.7, 0.8]} # High attenuation across all
    ]
    
    selected = random.choice(water_types)
    # Convert list to numpy array [B, G, R]
    # Note: OpenCV uses BGR, so we order them accordingly
    eta = np.array(selected['eta'], dtype=np.float32)
    
    # Beta (backscatter) usually relates to the brightness of the haze
    # We generate a random beta between 0.5 and 0.9 as per the paper's model
    beta = np.array([random.uniform(0.5, 0.8) for _ in range(3)], dtype=np.float32)
    
    return eta, beta

def simulate_underwater_degradation(image):
    """
    Applies the degradation model Eq. (1) to a clear image using blind parameters.
    """
    IR = image.astype(np.float32) / 255.0
    
    # Generate parameters blindly
    eta, beta = get_blind_parameters()
    
    # Per the paper (section 2.2), d is randomly generated from [0, 1]
    # We use a minimum of 0.2 to ensure the effect is visible
    d = random.uniform(0.2, 1.0) 
    
    # Eq. (1) / Eq. (6): IU = S1 + M2
    # S1 (Attenuation) = IR * exp(-eta * d)
    # M2 (Scattering) = beta * (1 - exp(-eta * d))
    
    attenuation = np.exp(-eta * d)
    
    # Reshape for broadcasting over the image [H, W, 3]
    attenuation = attenuation.reshape(1, 1, 3)
    beta = beta.reshape(1, 1, 3)
    
    S1 = IR * attenuation
    M2 = beta * (1 - attenuation)
    
    IU = S1 + M2
    
    # Clip and convert back to uint8
    IU = np.clip(IU * 255.0, 0, 255).astype(np.uint8)
    return IU

def process_dataset(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process all images in the input folder
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    
    print(f"Starting processing of {len(files)} images...")

    for filename in files:
        img_path = os.path.join(input_folder, filename)
        clear_img = cv2.imread(img_path)
        
        if clear_img is not None:
            # Generate "bad" image blindly
            degraded_img = simulate_underwater_degradation(clear_img)
            
            # Save to output
            save_path = os.path.join(output_folder, f"degraded_{filename}")
            cv2.imwrite(save_path, degraded_img)
            
    print(f"Success: All images saved to {output_folder}")

# Updated paths from your terminal output
input_dir = "/media/mesho/Storage/Underwater/Underwater GAN/UIEB Dataset-raw/raw-890/"
output_dir = "/media/mesho/Storage/Underwater/Underwater GAN/UIEB Dataset-raw/processed/"

process_dataset(input_dir, output_dir)