#!/usr/bin/env python3
"""
Pre-compress all validation images by pixel dimensions.
Run this BEFORE evaluation to avoid kernel crashes.
"""
import json
import os
from PIL import Image
import tempfile

# Params
MAX_PIXELS = 1_500_000  # ~1200x1200
MAX_DIMENSION = 1500
QUALITY = 85

def compress_by_pixels(image_path, max_pixels=MAX_PIXELS, max_dimension=MAX_DIMENSION, quality=QUALITY):
    """Compress image if it has too many pixels."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        total_pixels = width * height

        needs_compression = (total_pixels > max_pixels) or (max(width, height) > max_dimension)

        if not needs_compression:
            img.close()
            return image_path, False  # No compression needed

        # Generate compressed path
        dir_path = os.path.dirname(image_path)
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)
        compressed_path = os.path.join(dir_path, f"eval_compressed_{name}.jpg")

        # Skip if already exists
        if os.path.exists(compressed_path):
            img.close()
            return compressed_path, True

        print(f"Compressing {basename}: {width}x{height} ({total_pixels:,} pixels, {os.path.getsize(image_path)/1_000_000:.2f} MB)", end=" ")

        # Resize if needed
        if max(width, height) > max_dimension:
            ratio = max_dimension / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"→ {new_size[0]}x{new_size[1]}", end=" ")

        # Convert to RGB
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img.close()
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Save
        img.save(compressed_path, 'JPEG', quality=quality, optimize=True)
        img.close()

        compressed_size = os.path.getsize(compressed_path) / 1_000_000
        print(f"→ {compressed_size:.2f} MB")

        return compressed_path, True

    except Exception as e:
        print(f"ERROR: {e}")
        return image_path, False

def main():
    print("=" * 80)
    print("PRE-COMPRESSING VALIDATION IMAGES")
    print("=" * 80)

    # Load val.jsonl
    with open('val.jsonl', 'r') as f:
        val_data = [json.loads(line) for line in f if line.strip()]

    print(f"Found {len(val_data)} validation samples\n")

    # Compress each image
    compressed_count = 0
    total_saved = 0

    for i, sample in enumerate(val_data):
        img_path = sample['image']

        if not os.path.exists(img_path):
            continue

        original_size = os.path.getsize(img_path)
        compressed_path, was_compressed = compress_by_pixels(img_path)

        if was_compressed:
            compressed_count += 1
            # Update path in sample
            sample['image'] = compressed_path
            compressed_size = os.path.getsize(compressed_path)
            total_saved += (original_size - compressed_size)

    # Save updated val.jsonl
    print(f"\n" + "=" * 80)
    print(f"Compressed {compressed_count} images")
    print(f"Total saved: {total_saved / 1_000_000:.2f} MB")

    # Save updated val.jsonl
    with open('val.jsonl', 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')

    print("✅ Updated val.jsonl with compressed paths")
    print("=" * 80)

if __name__ == '__main__':
    main()
