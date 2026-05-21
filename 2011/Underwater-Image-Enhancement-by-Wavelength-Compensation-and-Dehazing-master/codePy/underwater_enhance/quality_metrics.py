# quality_metrics.py — FINAL FIXED VERSION
# Handles normalization correctly at EVERY stage
import numpy as np
import cv2
import os
import sys
import csv

def _normalize_and_ensure_rgb(img):
    """
    Ensure image is:
    1. RGB (convert BGR if needed)
    2. float32 with values in [0,1]
    """
    # If already float and in [0,1], use as-is
    if img.dtype in (np.float32, np.float64) and img.max() <= 1.5:
        img = img.astype(np.float32)
    else:
        # Assume uint8 [0,255]
        img = img.astype(np.float32) / 255.0

    # If grayscale (H, W), convert to (H, W, 3)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3 and img.ndim == 3:
        # Ensure RGB (OpenCV loads as BGR, but if input was RGB, skip)
        # Try to detect — but safest: assume BGR if loaded via cv2.imread
        pass
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return img


def uciqe(img):
    """
    UCIQE: Universal Image Quality Index for Underwater Images
    Returns value in [0.0, ~0.8] — higher = better.
    """
    # 🔑 Normalize and ensure RGB FIRST — and keep as [0,1] float32
    img = _normalize_and_ensure_rgb(img)

    # Convert to LAB — but img is still [0,1], so we multiply by 255 for cv2
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2Lab)

    L = lab[:, :, 0].astype(np.float32) / 255.0  # Normalize back to [0,1] for stats
    A = lab[:, :, 1].astype(np.float32) / 255.0
    B = lab[:, :, 2].astype(np.float32) / 255.0

    contrast = np.std(L)
    colorfulness = np.sqrt(np.var(A) + np.var(B))
    vividness = 0.5 * np.mean(L) + 0.5 * np.std(L)

    return 0.42 * contrast + 0.48 * colorfulness + 0.1 * vividness


def uiqm(img):
    """
    UIQM: Underwater Image Quality Measure
    Returns value in [0.0, ~5.0] — higher = better.
    """
    # Normalize
    img = _normalize_and_ensure_rgb(img)

    # LAB
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2Lab)
    L = lab[:, :, 0].astype(np.float32) / 255.0  # Back to [0,1]

    # UCM from RGB
    R = img[..., 0].flatten()
    G = img[..., 1].flatten()
    B = img[..., 2].flatten()
    ucm = np.sqrt((np.var(R) + np.var(G) + np.var(B)) / 3.0)

    # UQM
    uqm = np.mean(L) + np.std(L)

    # UEM
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    uem = np.mean(np.sqrt(gx**2 + gy**2))

    return 0.1 * uqm + 0.4 * ucm + 0.5 * uem


def get_metrics(image):
    """Return dict: {'UCIQE': val, 'UIQM': val}."""
    return {'UCIQE': uciqe(image), 'UIQM': uiqm(image)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python quality_metrics.py <folder_or_image>")
        sys.exit(1)

    path = sys.argv[1]
    results = []

    if os.path.isdir(path):
        print(f"Processing folder: {path}")
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(path, f)
                img = cv2.imread(full_path)
                if img is None:
                    print(f"⚠️ Warning: Could not load {full_path}")
                    continue
                # Force RGB if BGR
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                m = get_metrics(img)
                results.append([f, m['UCIQE'], m['UIQM']])
                print(f"{f}: UCIQE={m['UCIQE']:.4f}, UIQM={m['UIQM']:.4f}")

        # Summary
        if results:
            uciqes = [r[1] for r in results]
            uiqms = [r[2] for r in results]
            print(f"\n📊 Total images: {len(results)}")
            print(f"Average UCIQE: {np.mean(uciqes):.4f}  (good >0.35)")
            print(f"Average UIQM:  {np.mean(uiqms):.4f}  (good >3.0)")

            # Top 5 & Bottom 5
            top5 = sorted(results, key=lambda x: -x[1])[:5]
            bot5 = sorted(results, key=lambda x: x[1])[:5]

            print("\n🏆 Top 5 by UCIQE:")
            for f, u, uq in top5:
                print(f"  {f}: UCIQE={u:.4f}")

            print("\n📉 Bottom 5 by UCIQE:")
            for f, u, uq in bot5:
                print(f"  {f}: UCIQE={u:.4f}")

            # Save CSV
            with open('uciqe_report.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['filename', 'UCIQE', 'UIQM'])
                for r in results:
                    writer.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}"])
            print("\n✅ Saved: uciqe_report.csv")
    else:
        # Single image
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Error loading: {path}")
            sys.exit(1)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = get_metrics(img)
        print(f"{os.path.basename(path)}: UCIQE={m['UCIQE']:.4f}, UIQM={m['UIQM']:.4f}")


if __name__ == "__main__":
    main()