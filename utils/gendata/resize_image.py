#!/usr/bin/env python3
"""
Resize an image to the given width and height.
"""

import cv2
import argparse
import os

def resize_image(input_path, output_path, width, height):
    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    # Resize
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save result
    cv2.imwrite(output_path, resized)
    print(f"Saved resized image to: {output_path} ({width}x{height})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize an image to new dimensions")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save resized image (with new name)")
    parser.add_argument("--width", type=int, required=True, help="New width in pixels")
    parser.add_argument("--height", type=int, required=True, help="New height in pixels")

    args = parser.parse_args()
    resize_image(args.input, args.output, args.width, args.height)
