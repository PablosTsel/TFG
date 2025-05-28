#!/usr/bin/env python3
# Visualize data augmentation techniques used in training

import os
import sys
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

def visualize_augmentations(image_path):
    """
    Create a 2×2 grid showing different augmentations:
    1. Original image
    2. Horizontally flipped image
    3. Rotated image (90 degrees)
    4. Color jittered image
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load the image
    try:
        img = Image.open(image_path).convert("RGB")
        print(f"Successfully loaded image: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Create individual augmentations
    img_hflip = T.functional.hflip(img)
    img_rotated = T.functional.rotate(img, angle=90)
    
    # Color jitter augmentation
    brightness = 1.3  # Increase brightness (range: 0.7-1.3)
    contrast = 1.2    # Increase contrast (range: 0.8-1.2)
    saturation = 0.8  # Decrease saturation (range: 0.8-1.2)
    
    img_jittered = T.functional.adjust_brightness(img, brightness)
    img_jittered = T.functional.adjust_contrast(img_jittered, contrast)
    img_jittered = T.functional.adjust_saturation(img_jittered, saturation)
    
    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Plot images
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(img_hflip)
    axes[0, 1].set_title("Horizontal Flip")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(img_rotated)
    axes[1, 0].set_title("90° Rotation")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(img_jittered)
    axes[1, 1].set_title("Color Jittering\n(Brightness, Contrast, Saturation)")
    axes[1, 1].axis("off")
    
    # Add title with image name
    img_name = os.path.basename(image_path)
    plt.suptitle(f"Data Augmentation Techniques: {img_name}", fontsize=14)
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = "augmentation_visualization.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Visualization saved as {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    # Use the specific image requested
    image_path = "data/xBD/guatemala-volcano/images/guatemala-volcano_00000015_pre_disaster.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        sys.exit(1)
    
    visualize_augmentations(image_path) 