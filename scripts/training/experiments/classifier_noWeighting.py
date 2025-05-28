#!/usr/bin/env python3
# No-Weighting Damage Classification Model
# This version removes class weighting and sample weighting mechanisms
# to demonstrate the importance of handling class imbalance in damage assessment

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_score
import random
from datetime import datetime
import time
import torch.nn.functional as F
import json
from PIL import Image, ImageDraw
import torchvision.transforms as T
from shapely import wkt
import shutil

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

# Import from project modules
from scripts.training.utils import (
    UNet, seed_everything, create_versioned_directory, 
    calculate_iou, plot_learning_curves, FocalLoss
)

# Import damage class mapping and dice loss from the main damage classifier
from scripts.training.train_dam_classifier import (
    DAMAGE_CLASS_MAP, ImprovedDamageDataset, dice_loss_multiclass, 
    AttentionFusion, ImprovedDamageClassifier
)

def visualize_noweighting_predictions(model, dataset, device, num_samples=4, save_path=None):
    """
    Visualize damage classification predictions
    """
    model.eval()
    # Select random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Define colors for each damage class
    damage_colors = [
        [0, 0, 0],       # Background (black)
        [0, 255, 0],     # No damage (green)
        [255, 255, 0],   # Minor damage (yellow)
        [255, 165, 0],   # Major damage (orange)
        [255, 0, 0]      # Destroyed (red)
    ]
    
    # Class names for display
    class_names = ['Background', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    
    # Configure the plot
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]  # Make sure axes is a list for single sample case
    
    # Track per-class IoU
    class_iou = {cls: [] for cls in range(5)}
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                # Get sample
                pre_img, post_img, damage_mask = dataset[idx]
                pre_img = pre_img.unsqueeze(0).to(device)
                post_img = post_img.unsqueeze(0).to(device)
                
                # Get prediction
                outputs = model(pre_img, post_img)
                
                # Get predicted class for each pixel
                _, pred_classes = torch.max(outputs, dim=1)
                
                # Move tensors to CPU for visualization
                pred_classes = pred_classes.squeeze().cpu().numpy()
                damage_mask = damage_mask.cpu().numpy()
                
                # Create colored visualizations
                colored_pred = np.zeros((pred_classes.shape[0], pred_classes.shape[1], 3), dtype=np.uint8)
                colored_mask = np.zeros((damage_mask.shape[0], damage_mask.shape[1], 3), dtype=np.uint8)
                
                # Color each class in the prediction and ground truth
                for class_idx in range(len(damage_colors)):
                    colored_pred[pred_classes == class_idx] = damage_colors[class_idx]
                    colored_mask[damage_mask == class_idx] = damage_colors[class_idx]
                
                # Calculate per-class IoU for this sample
                for cls in range(5):
                    true_mask = (damage_mask == cls)
                    pred_mask = (pred_classes == cls)
                    
                    intersection = np.logical_and(true_mask, pred_mask).sum()
                    union = np.logical_or(true_mask, pred_mask).sum()
                    iou = intersection / union if union > 0 else float('nan')
                    
                    if not np.isnan(iou):
                        class_iou[cls].append(iou)
                
                # Denormalize images for display
                pre_img_np = pre_img.squeeze().cpu().numpy()
                post_img_np = post_img.squeeze().cpu().numpy()
                
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                pre_img_np = np.transpose(pre_img_np, (1, 2, 0))
                pre_img_np = pre_img_np * std + mean
                pre_img_np = np.clip(pre_img_np, 0, 1)
                
                post_img_np = np.transpose(post_img_np, (1, 2, 0))
                post_img_np = post_img_np * std + mean
                post_img_np = np.clip(post_img_np, 0, 1)
                
                # Plot in the current row
                axes[i][0].imshow(pre_img_np)
                axes[i][0].set_title("Pre-disaster Image")
                axes[i][0].axis('off')
                
                axes[i][1].imshow(post_img_np)
                axes[i][1].set_title("Post-disaster Image")
                axes[i][1].axis('off')
                
                axes[i][2].imshow(colored_mask)
                axes[i][2].set_title("Ground Truth Damage")
                axes[i][2].axis('off')
                
                axes[i][3].imshow(colored_pred)
                axes[i][3].set_title("Predicted Damage")
                axes[i][3].axis('off')
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                # In case of error, fill the row with blank plots
                for j in range(4):
                    axes[i][j].imshow(np.zeros((10, 10, 3)))
                    axes[i][j].set_title("Error")
                    axes[i][j].axis('off')
    
    # Calculate average IoU for each class
    avg_class_iou = {}
    for cls in range(5):
        if class_iou[cls]:
            avg_class_iou[cls] = sum(class_iou[cls]) / len(class_iou[cls])
        else:
            avg_class_iou[cls] = float('nan')
    
    # Calculate mean IoU (excluding background)
    valid_ious = [iou for cls, iou in avg_class_iou.items() if cls > 0 and not np.isnan(iou)]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0
    
    # Add a text box with IoU results
    plt.figtext(0.5, 0.01, f"Mean IoU (excluding background): {mean_iou:.4f}", ha="center", 
                fontsize=14, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    text = ""
    for cls in range(5):
        iou_value = avg_class_iou[cls]
        iou_str = f"{iou_value:.4f}" if not np.isnan(iou_value) else "N/A"
        text += f"{class_names[cls]}: {iou_str}   "
    
    plt.figtext(0.5, 0.005, text, ha="center", fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    # Print IoU values
    print("\nPer-Class IoU:")
    for cls in range(5):
        iou_value = avg_class_iou[cls]
        iou_str = f"{iou_value:.4f}" if not np.isnan(iou_value) else "N/A"
        print(f"  {class_names[cls]}: {iou_str}")
    print(f"Mean IoU (excluding background): {mean_iou:.4f}")
    
    plt.close()
    
    return avg_class_iou, mean_iou

def main():
    """
    Main function to run the damage classification training pipeline WITHOUT class weighting.
    This version uses equal weighting for all classes to demonstrate the importance
    of addressing class imbalance in damage assessment.
    """
    # Set multiprocessing start method to 'spawn' to avoid CUDA initialization errors
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        print("Multiprocessing start method already set to 'spawn' or could not be set")
        pass
        
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_everything(42)
    
    # Get project root directory
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    
    # Hyperparameters & settings
    root_dir = os.path.join(project_root, "data", "xBD")
    batch_size = 4
    lr = 0.0002
    num_epochs = 10
    val_ratio = 0.2
    image_size = 256
    num_classes = 5  # Background + 4 damage classes

    # Create output directory
    output_dir = os.path.join(project_root, "output", "experiments", "damage_classifier")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create run directory for this experiment
    experiment_name = "no_weighting"
    run_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    
    model_dir = os.path.join(run_dir, "models")
    viz_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "experiment": "No Class Weighting",
        "timestamp": timestamp,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "val_ratio": val_ratio,
        "image_size": image_size,
        "num_classes": num_classes
    }
    
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dataset with both pre and post disaster images
    print("Initializing ImprovedDamageDataset for training...")
    full_dataset = ImprovedDamageDataset(
        root_dir=root_dir,
        image_size=image_size,
        use_xy=True,
        augment=True
    )
    
    # Create train-val split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(np.floor(val_ratio * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create dataloaders WITHOUT weighted sampling - this is a key difference
    # from the original model, which uses weighted sampling to address class imbalance
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Regular shuffling instead of weighted sampling
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model with the same architecture as the original model
    model = ImprovedDamageClassifier(in_channels=3, out_channels=num_classes).to(device)
    
    # Loss components WITHOUT class weighting - this is a key difference
    # from the original model which uses class weighting to focus on minor damage
    gamma = 3.0  # focus parameter for Focal
    
    # Use uniform class weights (equal weights for all classes)
    class_weights = torch.ones(num_classes).to(device)
    
    focal_loss_fn = FocalLoss(gamma=gamma, alpha=class_weights)

    def combined_loss_fn(logits, targets):
        focal = focal_loss_fn(logits, targets)
        dice = dice_loss_multiclass(logits, targets, num_classes=num_classes, ignore_index=None)
        return 0.6 * focal + 0.4 * dice
    
    print(f"Using combined loss WITHOUT class weighting: 0.6*Focal(gamma={gamma}) + 0.4*Dice")
    print(f"Using uniform class weights: {class_weights.tolist()}")
    
    # Optimizer with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler - using OneCycleLR 
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training metrics tracking
    epochs_list = []
    train_losses = []
    val_losses = []
    val_ious = []
    best_iou = 0.0
    best_model_path = None
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        # Training phase
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]", leave=True)
        for pre_imgs, post_imgs, damage_masks in train_pbar:
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            damage_masks = damage_masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - outputs shape: [batch_size, num_classes, H, W]
            outputs = model(pre_imgs, post_imgs)
            
            # Compute combined loss
            loss = combined_loss_fn(outputs, damage_masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]", leave=True)
            for pre_imgs, post_imgs, damage_masks in val_pbar:
                pre_imgs = pre_imgs.to(device)
                post_imgs = post_imgs.to(device)
                damage_masks = damage_masks.to(device)
                
                # Forward pass
                outputs = model(pre_imgs, post_imgs)
                
                # Compute combined loss
                loss = combined_loss_fn(outputs, damage_masks)
                val_loss += loss.item()
                
                # Get predicted class for each pixel
                _, preds = torch.max(outputs, dim=1)
                
                # Calculate IoU for multi-class segmentation
                batch_iou = calculate_iou(preds, damage_masks, threshold=0.5, num_classes=num_classes)
                val_iou += batch_iou
                
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{batch_iou:.4f}"})
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        epochs_list.append(epoch + 1)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val IoU: {avg_val_iou:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save the model if it's the best so far
        if avg_val_iou > best_iou:
            # Delete previous best model file if it exists
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"Removed previous best model: {best_model_path}")
            
            best_iou = avg_val_iou
            best_model_path = os.path.join(model_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with IoU: {best_iou:.4f}")
            
            # Visualize predictions with best model
            vis_save_path = os.path.join(viz_dir, f"predictions_epoch_{epoch+1}.png")
            visualize_noweighting_predictions(model, full_dataset, device, num_samples=4, save_path=vis_save_path)
    
    # Create learning curves plot
    curves_save_path = os.path.join(viz_dir, "learning_curves.png")
    plot_learning_curves(
        epochs_list,
        train_losses,
        val_losses,
        val_ious,
        curves_save_path
    )
    
    # Additional analysis: per-class performance at final model
    # This will be particularly useful to show the impact of missing class weighting
    if best_model_path:
        print("\nAnalyzing per-class performance of best model...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        class_names = ['Background', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        
        with torch.no_grad():
            for pre_imgs, post_imgs, damage_masks in tqdm(val_loader, desc="Evaluating class performance"):
                pre_imgs = pre_imgs.to(device)
                post_imgs = post_imgs.to(device)
                damage_masks = damage_masks.to(device)
                
                outputs = model(pre_imgs, post_imgs)
                _, preds = torch.max(outputs, dim=1)
                
                # Count per-class accuracy
                for c in range(num_classes):
                    class_mask = (damage_masks == c)
                    if class_mask.sum() > 0:  # Only calculate if class exists in ground truth
                        class_correct[c] += ((preds == c) & class_mask).sum().item()
                        class_total[c] += class_mask.sum().item()
        
        # Calculate per-class accuracy
        class_accuracy = {}
        print("\nPer-Class Accuracy:")
        for c in range(num_classes):
            if class_total[c] > 0:
                accuracy = class_correct[c] / class_total[c]
                class_accuracy[c] = accuracy
                print(f"  {class_names[c]}: {accuracy:.4f} ({class_correct[c]}/{class_total[c]})")
            else:
                class_accuracy[c] = float('nan')
                print(f"  {class_names[c]}: N/A (No samples)")
        
        # Save class performance
        class_perf_path = os.path.join(run_dir, "class_performance.txt")
        with open(class_perf_path, "w") as f:
            f.write("Class Performance (Best Model):\n")
            for c in range(num_classes):
                if class_total[c] > 0:
                    f.write(f"{class_names[c]}: Accuracy={class_accuracy[c]:.4f} ({class_correct[c]}/{class_total[c]})\n")
                else:
                    f.write(f"{class_names[c]}: N/A (No samples)\n")
                    
        # Create a bar chart to visualize class performance
        plt.figure(figsize=(10, 6))
        valid_classes = [c for c in range(num_classes) if not np.isnan(class_accuracy[c])]
        valid_names = [class_names[c] for c in valid_classes]
        valid_accuracies = [class_accuracy[c] for c in valid_classes]
        
        colors = ['gray', 'green', 'yellow', 'orange', 'red']
        bar_colors = [colors[c] for c in valid_classes]
        
        plt.bar(valid_names, valid_accuracies, color=bar_colors)
        plt.xlabel('Damage Class')
        plt.ylabel('Pixel Accuracy')
        plt.title('Per-Class Accuracy (No Weighting)')
        plt.ylim([0, 1.0])
        
        for i, v in enumerate(valid_accuracies):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
            
        plt.tight_layout()
        
        class_chart_path = os.path.join(viz_dir, "class_accuracy.png")
        plt.savefig(class_chart_path)
        print(f"Class accuracy chart saved to {class_chart_path}")
        plt.close()
    
    # Save final metrics
    metrics = {
        "epochs": epochs_list,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_ious": val_ious,
        "best_iou": best_iou
    }
    
    metrics_path = os.path.join(run_dir, "training_metrics.txt")
    with open(metrics_path, "w") as f:
        for key, values in metrics.items():
            if isinstance(values, list):
                f.write(f"{key}: {values}\n")
            else:
                f.write(f"{key}: {values}\n")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"All artifacts saved to {run_dir}")


if __name__ == "__main__":
    main() 