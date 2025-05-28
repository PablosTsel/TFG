#!/usr/bin/env python3
# Visualize Dense Building Samples
# This script generates visualizations for samples with high building density

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as T
import random
from pathlib import Path
from shapely import wkt
import json
import argparse

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

# Import from our other scripts
from scripts.training.utils import UNet
from scripts.training.train_dam_classifier import (
    ImprovedDamageClassifier, DAMAGE_CLASS_MAP
)
from scripts.inference.evaluate_two_stage import IntegratedDamageAssessment, ImprovedEvaluationDataset

def count_building_pixels(dataset, num_samples=100, min_building_pixels=1000):
    """
    Count building pixels in samples and return indices of samples with many buildings.
    
    Args:
        dataset: Dataset with building masks
        num_samples: Number of samples to check (randomly)
        min_building_pixels: Minimum number of building pixels to consider a sample as dense
        
    Returns:
        List of indices with high building pixel count
    """
    dataset_size = len(dataset)
    indices_to_check = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    dense_indices = []
    building_counts = []
    
    print("Finding samples with many buildings...")
    for idx in tqdm(indices_to_check):
        try:
            # Get sample
            _, _, building_mask, _ = dataset[idx]
            
            # Count building pixels
            building_count = (building_mask > 0.5).sum().item()
            building_counts.append((idx, building_count))
            
            # Add to dense list if above threshold
            if building_count >= min_building_pixels:
                dense_indices.append(idx)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
    
    # Sort by building count (descending)
    building_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(dense_indices)} samples with at least {min_building_pixels} building pixels")
    print("Top 10 densest samples:")
    for i, (idx, count) in enumerate(building_counts[:10]):
        print(f"  Sample {idx}: {count} building pixels")
    
    return [idx for idx, _ in building_counts[:num_samples]]

def visualize_sample(model, dataset, sample_idx, output_dir):
    """
    Create a visualization for a single sample in the same format as evaluate_two_stage.py
    
    Args:
        model: IntegratedDamageAssessment model
        dataset: Dataset with samples
        sample_idx: Index of the sample to visualize
        output_dir: Directory to save visualization
    """
    # Define colors for damage classes
    damage_colors = [
        [0, 0, 0],       # Background (black)
        [0, 255, 0],     # No damage (green)
        [255, 255, 0],   # Minor damage (yellow)
        [255, 165, 0],   # Major damage (orange)
        [255, 0, 0]      # Destroyed (red)
    ]
    
    # Get sample
    pre_img, post_img, gt_building_mask, gt_damage_mask = dataset[sample_idx]
    
    # Get predictions
    with torch.no_grad():
        # Run the integrated model
        building_mask, damage_pred, final_pred = model.predict(pre_img, post_img)
        
        # Get class predictions
        _, pred_classes = torch.max(final_pred, dim=1)
        
        # Move to CPU and numpy for evaluation
        pred_classes = pred_classes.squeeze().cpu().numpy()
        building_mask = building_mask.squeeze().cpu().numpy()
        gt_building_mask = gt_building_mask.squeeze().cpu().numpy()
        gt_damage_mask = gt_damage_mask.cpu().numpy()
        
        # Calculate building IoU
        building_tp = np.logical_and(building_mask > 0.5, gt_building_mask > 0.5).sum()
        building_union = np.logical_or(building_mask > 0.5, gt_building_mask > 0.5).sum()
        building_iou = building_tp / building_union if building_union > 0 else 0
        
        # ---------- VISUALISATION ---------- #
        # 1. Prepare colour maps
        colored_pred = np.zeros((*pred_classes.shape, 3), dtype=np.uint8)
        colored_gt = np.zeros((*gt_damage_mask.shape, 3), dtype=np.uint8)
        for cls_colour in range(len(damage_colors)):
            colored_pred[pred_classes == cls_colour] = damage_colors[cls_colour]
            colored_gt[gt_damage_mask == cls_colour] = damage_colors[cls_colour]

        # Binary building masks (white)
        pred_build_rgb = np.zeros((*building_mask.shape, 3), dtype=np.uint8)
        gt_build_rgb = np.zeros((*gt_building_mask.shape, 3), dtype=np.uint8)
        pred_build_rgb[building_mask > 0.5] = [255, 255, 255]
        gt_build_rgb[gt_building_mask > 0.5] = [255, 255, 255]

        # Building detection error map: red = FN, blue = FP
        build_error = np.zeros((*building_mask.shape, 3), dtype=np.uint8)
        false_neg = np.logical_and(gt_building_mask > 0.5, building_mask <= 0.5)
        false_pos = np.logical_and(gt_building_mask <= 0.5, building_mask > 0.5)
        build_error[false_neg] = [255, 0, 0]   # red
        build_error[false_pos] = [0, 0, 255]   # blue

        # Damage prediction error map: green background for correct, red for wrong
        damage_error = np.zeros((*pred_classes.shape, 3), dtype=np.uint8)
        correct_pix = (pred_classes == gt_damage_mask)
        damage_error[correct_pix] = [0, 255, 0]      # green
        damage_error[~correct_pix] = [255, 0, 0]     # red

        # Prepare pre & post images (denormalise)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        pre_np = pre_img.squeeze().cpu().numpy()
        pre_np = np.transpose(pre_np, (1, 2, 0)) * std + mean
        pre_np = np.clip(pre_np, 0, 1)

        post_np = post_img.squeeze().cpu().numpy()
        post_np = np.transpose(post_np, (1, 2, 0)) * std + mean
        post_np = np.clip(post_np, 0, 1)

        # Optional overlay that highlights minor-damage pixels on the post image
        minor_overlay = post_np.copy()
        minor_mask = (pred_classes == 2)
        minor_overlay[minor_mask] = [0, 1, 0]  # vivid green highlight

        # -------- Plot -------- #
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))

        # Row 0 — raw imagery
        axes[0, 0].imshow(pre_np)
        axes[0, 0].set_title("Pre-disaster Image")
        axes[0, 0].axis('off')
        axes[0, 1].imshow(post_np)
        axes[0, 1].set_title("Post-disaster Image")
        axes[0, 1].axis('off')
        axes[0, 2].imshow(minor_overlay)
        axes[0, 2].set_title("Minor Damage Visualisation")
        axes[0, 2].axis('off')

        # Row 1 — building masks
        axes[1, 0].imshow(gt_build_rgb)
        axes[1, 0].set_title("Ground Truth Buildings")
        axes[1, 0].axis('off')
        axes[1, 1].imshow(pred_build_rgb)
        axes[1, 1].set_title(f"Predicted Buildings (IoU: {building_iou:.4f})")
        axes[1, 1].axis('off')
        axes[1, 2].imshow(build_error)
        axes[1, 2].set_title("Building Detection Errors")
        axes[1, 2].axis('off')

        # Row 2 — damage masks
        axes[2, 0].imshow(colored_gt)
        axes[2, 0].set_title("Ground Truth Damage")
        axes[2, 0].axis('off')
        axes[2, 1].imshow(colored_pred)
        axes[2, 1].set_title("Predicted Damage")
        axes[2, 1].axis('off')
        axes[2, 2].imshow(damage_error)
        axes[2, 2].set_title("Prediction Errors")
        axes[2, 2].axis('off')

        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"dense_buildings_{sample_idx}.png")
        plt.savefig(fig_path)
        plt.close()
        
        return building_tp  # Return building pixel count as a measure of density

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for samples with many buildings')
    parser.add_argument('--xbd_dir', default='data/xBD', help='Path to the xBD data directory')
    parser.add_argument('--output_dir', default='error_analysis/dense_buildings', help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of dense building samples to visualize')
    parser.add_argument('--min_building_pixels', type=int, default=2000, help='Minimum number of building pixels to consider a sample as dense')
    parser.add_argument('--check_samples', type=int, default=200, help='Number of random samples to check for building density')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Model paths
    building_detector_path = os.path.join(project_root, "output", "building_detector", "binary_building_best_new.pt")
    damage_classifier_path = os.path.join(project_root, "output", "dam_classifier", "improved_damage_best_new.pt")
    
    # Check if model files exist
    if not os.path.exists(building_detector_path):
        print(f"ERROR: Building detector model not found at: {building_detector_path}")
        return
    
    if not os.path.exists(damage_classifier_path):
        print(f"ERROR: Damage classifier model not found at: {damage_classifier_path}")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the integrated model
    integrated_model = IntegratedDamageAssessment(
        building_detector_path=building_detector_path,
        damage_classifier_path=damage_classifier_path,
        device=device,
        building_threshold=0.5
    )
    
    # Create dataset
    print("Initializing dataset...")
    dataset = ImprovedEvaluationDataset(
        root_dir=os.path.join(project_root, args.xbd_dir),
        building_detector_path=building_detector_path,
        device=device,
        image_size=256,
        use_xy=True,
        augment=False
    )
    
    # Find samples with many buildings
    dense_indices = count_building_pixels(
        dataset, 
        num_samples=args.check_samples, 
        min_building_pixels=args.min_building_pixels
    )
    
    # Limit to the requested number of samples
    dense_indices = dense_indices[:args.num_samples]
    
    # Generate visualizations for dense building samples
    print(f"Generating visualizations for {len(dense_indices)} dense building samples...")
    for i, idx in enumerate(tqdm(dense_indices)):
        building_count = visualize_sample(integrated_model, dataset, idx, output_dir)
        print(f"Sample {idx}: {building_count} building pixels visualized")
    
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main() 