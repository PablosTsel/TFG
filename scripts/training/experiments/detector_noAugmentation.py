#!/usr/bin/env python3
# Building Detection Model without Data Augmentation
# This version uses the combined BCE+Dice loss but disables data augmentation
# to show the impact of augmentation on generalization

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

# Import the UNet architecture and helper functions
from scripts.training.utils import (
    UNet, seed_everything, create_versioned_directory, 
    calculate_iou, plot_learning_curves
)

# Import the dataset class and visualization function
from scripts.training.train_building_detector import XBDBinaryBuildingDataset, visualize_binary_predictions

# Import combined loss function 
from scripts.training.train_building_detector import dice_loss

def main():
    """
    Main function to run the binary building segmentation training pipeline
    with combined BCE+Dice loss but without data augmentation.
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
    
    # Hyperparameters & settings
    root_dir = os.path.join(project_root, "data", "xBD")
    batch_size = 16
    lr = 0.0002
    num_epochs = 10
    val_ratio = 0.2
    image_size = 256
    
    # Create output directory for experimental results
    output_dir = os.path.join(project_root, "output", "experiments", "building_detector")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create run directory for this experiment
    experiment_name = "no_augmentation"
    run_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    
    model_dir = os.path.join(run_dir, "models")
    viz_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "experiment": "No Data Augmentation",
        "timestamp": timestamp,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "val_ratio": val_ratio,
        "image_size": image_size
    }
    
    with open(os.path.join(run_dir, f"config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dataset without augmentation (setting augment=False)
    print("Initializing XBDBinaryBuildingDataset without augmentation...")
    full_dataset = XBDBinaryBuildingDataset(
        root_dir=root_dir,
        image_size=image_size,
        use_xy=True,
        augment=False  # No augmentation in this experiment
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    
    # Initialize model - binary segmentation has 1 output channel
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Use combined BCE + Dice loss (like the full model)
    bce_loss = nn.BCEWithLogitsLoss()

    def combined_loss(logits, targets):
        return 0.5 * bce_loss(logits, targets) + 0.5 * dice_loss(logits, targets)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
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
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute combined loss
            loss = combined_loss(outputs, masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
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

            # Prepare IoU containers for this epoch
            thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
            iou_per_thr = {t: 0.0 for t in thresholds}

            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute combined loss
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()
                
                # Probabilities once per batch
                probs = torch.sigmoid(outputs)

                # Compute IoU for each threshold in this batch
                for t in thresholds:
                    preds_t = (probs > t).float()
                    iou_t_batch = calculate_iou(preds_t, masks, threshold=t) 
                    iou_per_thr[t] += iou_t_batch 

                # Use 0.5 threshold for progress bar
                preds_default = (probs > 0.5).float()
                batch_iou = calculate_iou(preds_default, masks, threshold=0.5)
                val_iou += batch_iou

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou@0.5": f"{batch_iou:.4f}"})
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_ious.append(val_iou / len(val_loader)) 
        val_losses.append(avg_val_loss)
        epochs_list.append(epoch + 1)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Pick best threshold for this epoch
        avg_iou_per_thr = {t: total_iou / len(val_loader) for t, total_iou in iou_per_thr.items()}
        best_thr_epoch = max(avg_iou_per_thr, key=avg_iou_per_thr.get)
        best_avg_val_iou_epoch = avg_iou_per_thr[best_thr_epoch]

        print(f"Best threshold this epoch: {best_thr_epoch:.2f}, IoU: {best_avg_val_iou_epoch:.4f}")
        
        # Save the model if it's the best so far
        if best_avg_val_iou_epoch > best_iou:
            # Delete previous best model file if it exists
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"Removed previous best model: {best_model_path}")
            
            best_iou = best_avg_val_iou_epoch
            best_model_path = os.path.join(model_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with IoU: {best_iou:.4f}")
            
            # Persist the best threshold
            thr_path = os.path.join(model_dir, "best_threshold.txt")
            with open(thr_path, "w") as f_thr:
                f_thr.write(str(best_thr_epoch))
            print(f"Saved best threshold {best_thr_epoch:.2f} to {thr_path}")
            
            # Visualize predictions with best model
            vis_save_path = os.path.join(viz_dir, f"predictions_epoch_{epoch+1}.png")
            visualize_binary_predictions(model, full_dataset, device, num_samples=4, save_path=vis_save_path)
    
    # Create learning curves plot
    curves_save_path = os.path.join(viz_dir, "learning_curves.png")
    plot_learning_curves(
        epochs_list,
        train_losses,
        val_losses,
        val_ious,
        curves_save_path
    )
    
    # Save final metrics
    metrics = {
        "epochs": epochs_list,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_ious": val_ious,
        "best_iou": best_iou
    }
    
    metrics_path = os.path.join(run_dir, f"training_metrics.txt")
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