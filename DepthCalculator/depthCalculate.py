import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class DepthEstimatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Underwater Depth Estimator Pro")
        self.root.geometry("800x600")

        # UI Elements
        self.btn_load = tk.Button(root, text="Load Image", command=self.load_image, font=("Arial", 12))
        self.btn_load.pack(pady=10)

        self.canvas = tk.Canvas(root, width=500, height=350, bg="gray")
        self.canvas.pack()

        self.result_label = tk.Label(root, text="Estimated Depth: -- meters", font=("Arial", 16, "bold"), fg="blue")
        self.result_label.pack(pady=20)

        self.info_label = tk.Label(root, text="Calculation based on Red-Channel Attenuation (Beer-Lambert Approximation)", font=("Arial", 10))
        self.info_label.pack()

    def load_image(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if not file_path:
            return

        # Process the specific file mentioned
        # In a real scenario, it uses file_path. For this task, we reference image_38dd1a.png
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Could not open image.")
            return

        # 1. Calculate Estimated Depth using the Beer-Lambert Logic
        # Red light attenuates fastest. We compare Red vs Blue/Green.
        b, g, r = cv2.split(image)
        avg_r = np.mean(r)
        avg_gb = (np.mean(g) + np.mean(b)) / 2

        # Physics Logic: If Red is 0, depth is > 20m. If Red is high, depth is < 2m.
        # Ratio-based approximation for recreational depths (0-30m)
        ratio = avg_r / (avg_gb + 1e-6)
        
        # Heuristic mapping: 
        # ratio ~1.0 -> ~0m
        # ratio ~0.1 -> ~20m
        depth_m = max(0, min(35, (1 - ratio) * 25)) 

        # 2. Update GUI
        self.result_label.config(text=f"Estimated Depth: {depth_m:.2f} meters")

        # 3. Display Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_pil.thumbnail((500, 350))
        self.img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(250, 175, image=self.img_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = DepthEstimatorGUI(root)
    root.mainloop()