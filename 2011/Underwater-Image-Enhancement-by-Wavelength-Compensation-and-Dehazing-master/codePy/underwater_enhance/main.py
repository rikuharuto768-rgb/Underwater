# main.py
import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from utils.dehaze import underwater_enhance

def process_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    if not image_files:
        print(f"⚠️ No images found in {input_dir}")
        return

    for img_path in tqdm(image_files, desc="Processing images"):
        out_path = output_dir / img_path.name
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️ Could not load image: {img_path.name}")
                continue
            enhanced = underwater_enhance(image)
            cv2.imwrite(str(out_path), enhanced)
            print(f"✅ Saved: {out_path}")
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Underwater Image Enhancement via Wavelength Compensation & Dehazing"
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input image OR directory')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to output image OR directory (must match input type)')
    parser.add_argument('--no-wb', action='store_true', help='Skip white balancing')
    parser.add_argument('--no-contrast', action='store_true', help='Skip contrast enhancement')
    parser.add_argument('--pyramid', action='store_true', help='Use full multi-salient pyramid fusion')
    parser.add_argument('--saliency-boost', type=float, default=0.3,
                        help='Boost factor for saliency cue (default: 0.3)')
    parser.add_argument('--chromatic-boost', type=float, default=0.2,
                        help='Boost factor for chromatic cue (default: 0.2)')
    parser.add_argument('--luminance-boost', type=float, default=0.3,
                        help='Boost factor for luminance cue (default: 0.3)')
    
    args = parser.parse_args()

    # Resolve absolute paths — fixes ./media vs /media confusion
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    print(f"🔍 Input path:  {input_path}")
    print(f"🔍 Output path: {output_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input}\n→ Searched absolute path: {input_path}")

    if input_path.is_file():
        print(f"➡️ Enhancing single image: {input_path.name}")
        image = cv2.imread(str(input_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {input_path}")
        enhanced = underwater_enhance(
            image,
            do_white_balance=not args.no_wb,
            use_contrast=not args.no_contrast,
            use_pyramid_fusion=args.pyramid,
            alpha_saliency=args.saliency_boost,
            alpha_chromatic=args.chromatic_boost,
            alpha_luminance=args.luminance_boost,
        )
        cv2.imwrite(str(output_path), enhanced)
        print(f"✅ Saved: {output_path}")
    elif input_path.is_dir():
        print(f"➡️ Enhancing all images in: {input_path.name}")
        output_path.mkdir(parents=True, exist_ok=True)
        process_folder(str(input_path), str(output_path))
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

if __name__ == "__main__":
    main()