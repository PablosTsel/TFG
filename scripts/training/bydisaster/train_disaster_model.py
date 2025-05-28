#!/usr/bin/env python3
# Training script for the Disaster-Conditioned Damage Classifier
# This model learns to predict disaster type and conditions damage assessment on it

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime
import time
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

# Import project components
from scripts.training.utils import (
    seed_everything, create_versioned_directory, 
    calculate_iou, plot_learning_curves, FocalLoss, 
    dice_loss_multiclass
)
from scripts.training.bydisaster.disaster_model import (
    DisasterConditionedDamageClassifier, DISASTER_TYPES
)
from scripts.training.bydisaster.disaster_dataset import DisasterAwareDataset

def visualize_predictions(model, dataset, device, num_samples=4, save_path=None):
    """
    Visualize model predictions including damage classification and disaster type
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
    damage_class_names = ['Background', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    
    # Configure the plot
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]  # Make sure axes is a list for single sample case
    
    # Track metrics
    disaster_correct = 0
    damage_ious = []
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                # Get sample
                pre_img, post_img, damage_mask, disaster_label = dataset[idx]
                pre_img = pre_img.unsqueeze(0).to(device)
                post_img = post_img.unsqueeze(0).to(device)
                
                # Get predictions
                damage_logits, disaster_logits = model(pre_img, post_img)
                
                # Get predicted damage classes
                _, pred_damage = torch.max(damage_logits, dim=1)
                
                # Get predicted disaster type
                _, pred_disaster = torch.max(disaster_logits, dim=1)
                
                # Move tensors to CPU for visualization
                pred_damage = pred_damage.squeeze().cpu().numpy()
                damage_mask = damage_mask.cpu().numpy()
                true_disaster = disaster_label.item()
                pred_disaster = pred_disaster.item()
                
                # Check if disaster prediction is correct
                if true_disaster == pred_disaster:
                    disaster_correct += 1
                
                # Create colored visualizations for damage
                colored_pred = np.zeros((pred_damage.shape[0], pred_damage.shape[1], 3), dtype=np.uint8)
                colored_mask = np.zeros((damage_mask.shape[0], damage_mask.shape[1], 3), dtype=np.uint8)
                
                # Color each class in the prediction and ground truth
                for class_idx in range(len(damage_colors)):
                    colored_pred[pred_damage == class_idx] = damage_colors[class_idx]
                    colored_mask[damage_mask == class_idx] = damage_colors[class_idx]
                
                # Calculate IoU for damage prediction
                # Skip background (class 0) in calculation
                damage_iou = 0
                valid_classes = 0
                for cls in range(1, len(damage_colors)):
                    true_mask = (damage_mask == cls)
                    pred_mask = (pred_damage == cls)
                    
                    intersection = np.logical_and(true_mask, pred_mask).sum()
                    union = np.logical_or(true_mask, pred_mask).sum()
                    iou = intersection / union if union > 0 else float('nan')
                    
                    if not np.isnan(iou):
                        damage_iou += iou
                        valid_classes += 1
                
                if valid_classes > 0:
                    damage_iou /= valid_classes
                    damage_ious.append(damage_iou)
                
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
                
                # Get disaster type names
                true_disaster_name = DISASTER_TYPES[true_disaster]
                pred_disaster_name = DISASTER_TYPES[pred_disaster]
                
                # Plot in the current row
                axes[i][0].imshow(pre_img_np)
                axes[i][0].set_title("Pre-disaster Image")
                axes[i][0].axis('off')
                
                axes[i][1].imshow(post_img_np)
                disaster_title = f"Post-disaster Image\nTrue: {true_disaster_name}, Pred: {pred_disaster_name}"
                axes[i][1].set_title(disaster_title, color='green' if true_disaster == pred_disaster else 'red')
                axes[i][1].axis('off')
                
                axes[i][2].imshow(colored_mask)
                axes[i][2].set_title("Ground Truth Damage")
                axes[i][2].axis('off')
                
                axes[i][3].imshow(colored_pred)
                axes[i][3].set_title(f"Predicted Damage (IoU: {damage_iou:.4f})")
                axes[i][3].axis('off')
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                # In case of error, fill the row with blank plots
                for j in range(4):
                    axes[i][j].imshow(np.zeros((10, 10, 3)))
                    axes[i][j].set_title("Error")
                    axes[i][j].axis('off')
    
    # Calculate metrics
    disaster_accuracy = disaster_correct / num_samples if num_samples > 0 else 0
    mean_damage_iou = sum(damage_ious) / len(damage_ious) if damage_ious else 0
    
    # Add a text box with metrics
    plt.figtext(0.5, 0.01, 
                f"Disaster Type Accuracy: {disaster_accuracy:.4f}, Mean Damage IoU: {mean_damage_iou:.4f}", 
                ha="center", fontsize=14, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.close()
    
    return disaster_accuracy, mean_damage_iou

def train_disaster_conditioned_model(args):
    """Main training function for the disaster-conditioned model"""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_everything(args.seed)
    
    # Create output directory
    output_dir = os.path.join(project_root, "output", "disaster_model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create versioned run directory
    run_dir, run_num = create_versioned_directory(output_dir, prefix="disaster_run")
    model_dir = os.path.join(run_dir, "models")
    viz_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config["timestamp"] = timestamp
    
    with open(os.path.join(run_dir, f"config_run{run_num}.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Initializing DisasterAwareDataset...")
    full_dataset = DisasterAwareDataset(
        root_dir=os.path.join(project_root, args.data_dir),
        image_size=args.image_size,
        use_xy=True,
        augment=True,
        disaster_type_balance=args.balance_disasters
    )
    
    # Create train-val split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(np.floor(args.val_ratio * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = DisasterConditionedDamageClassifier(
        in_channels=3, 
        out_channels=5,
        num_disaster_types=len(DISASTER_TYPES)
    ).to(device)
    
    # Define loss functions
    # Damage segmentation loss
    class_weights = torch.tensor([0.05, 2.0, 5.0, 3.0, 3.0]).to(device)
    focal_loss_fn = FocalLoss(gamma=3.0, alpha=class_weights)
    
    def damage_loss_fn(logits, targets):
        focal = focal_loss_fn(logits, targets)
        dice = dice_loss_multiclass(logits, targets, num_classes=5)
        return 0.6 * focal + 0.4 * dice
    
    # Disaster classification loss
    disaster_loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training metrics tracking
    epochs_list = []
    train_losses = []
    val_losses = []
    val_damage_ious = []
    val_disaster_accs = []
    best_combined_metric = 0.0
    
    # Train the model
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_damage_loss = 0.0
        running_disaster_loss = 0.0
        epoch_start_time = time.time()
        
        # Training phase
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]", leave=True)
        for batch in train_pbar:
            pre_imgs, post_imgs, damage_masks, disaster_labels = batch
            
            # Move to device
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            damage_masks = damage_masks.to(device)
            disaster_labels = disaster_labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            damage_logits, disaster_logits = model(pre_imgs, post_imgs)
            
            # Calculate losses
            damage_loss = damage_loss_fn(damage_logits, damage_masks)
            disaster_loss = disaster_loss_fn(disaster_logits, disaster_labels)
            
            # Combined loss
            loss = args.damage_weight * damage_loss + args.disaster_weight * disaster_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            running_damage_loss += damage_loss.item()
            running_disaster_loss += disaster_loss.item()
            
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "damage": f"{damage_loss.item():.4f}",
                "disaster": f"{disaster_loss.item():.4f}"
            })
        
        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        avg_damage_loss = running_damage_loss / len(train_loader)
        avg_disaster_loss = running_disaster_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_damage_loss = 0.0
        val_disaster_loss = 0.0
        val_damage_iou = 0.0
        val_disaster_correct = 0
        val_total = 0
        
        # Store true and predicted disaster labels for confusion matrix
        all_true_disasters = []
        all_pred_disasters = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]", leave=True)
            for batch in val_pbar:
                pre_imgs, post_imgs, damage_masks, disaster_labels = batch
                
                # Move to device
                pre_imgs = pre_imgs.to(device)
                post_imgs = post_imgs.to(device)
                damage_masks = damage_masks.to(device)
                disaster_labels = disaster_labels.to(device)
                
                # Forward pass
                damage_logits, disaster_logits = model(pre_imgs, post_imgs)
                
                # Calculate losses
                damage_loss = damage_loss_fn(damage_logits, damage_masks)
                disaster_loss = disaster_loss_fn(disaster_logits, disaster_labels)
                
                # Combined loss
                loss = args.damage_weight * damage_loss + args.disaster_weight * disaster_loss
                
                # Update metrics
                val_loss += loss.item()
                val_damage_loss += damage_loss.item()
                val_disaster_loss += disaster_loss.item()
                
                # Calculate damage IoU
                _, preds = torch.max(damage_logits, dim=1)
                batch_iou = calculate_iou(preds, damage_masks, threshold=0.5, num_classes=5)
                val_damage_iou += batch_iou
                
                # Calculate disaster classification accuracy
                _, pred_disasters = torch.max(disaster_logits, dim=1)
                val_disaster_correct += (pred_disasters == disaster_labels).sum().item()
                val_total += disaster_labels.size(0)
                
                # Store for confusion matrix
                all_true_disasters.extend(disaster_labels.cpu().numpy())
                all_pred_disasters.extend(pred_disasters.cpu().numpy())
                
                val_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "damage_iou": f"{batch_iou:.4f}",
                    "disaster_acc": f"{(pred_disasters == disaster_labels).sum().item() / disaster_labels.size(0):.4f}"
                })
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_damage_loss = val_damage_loss / len(val_loader)
        avg_val_disaster_loss = val_disaster_loss / len(val_loader)
        avg_val_damage_iou = val_damage_iou / len(val_loader)
        val_disaster_accuracy = val_disaster_correct / val_total if val_total > 0 else 0
        
        val_losses.append(avg_val_loss)
        val_damage_ious.append(avg_val_damage_iou)
        val_disaster_accs.append(val_disaster_accuracy)
        epochs_list.append(epoch + 1)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f} (Damage: {avg_damage_loss:.4f}, Disaster: {avg_disaster_loss:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} (Damage: {avg_val_damage_loss:.4f}, Disaster: {avg_val_disaster_loss:.4f}) | "
              f"Damage IoU: {avg_val_damage_iou:.4f} | "
              f"Disaster Acc: {val_disaster_accuracy:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Calculate combined metric (weighted average of IoU and accuracy)
        combined_metric = args.damage_weight * avg_val_damage_iou + args.disaster_weight * val_disaster_accuracy
        
        # Save the model if it's the best so far
        if combined_metric > best_combined_metric:
            best_combined_metric = combined_metric
            best_model_path = os.path.join(model_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with combined metric: {best_combined_metric:.4f}")
            
            # Generate confusion matrix for disaster type classification
            conf_matrix = confusion_matrix(all_true_disasters, all_pred_disasters)
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=DISASTER_TYPES, yticklabels=DISASTER_TYPES)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Disaster Type Confusion Matrix (Epoch {epoch+1})')
            conf_matrix_path = os.path.join(viz_dir, f"confusion_matrix_epoch_{epoch+1}.png")
            plt.savefig(conf_matrix_path)
            plt.close()
            
            # Generate classification report
            report = classification_report(all_true_disasters, all_pred_disasters, 
                                          target_names=DISASTER_TYPES, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_path = os.path.join(run_dir, f"classification_report_epoch_{epoch+1}.csv")
            report_df.to_csv(report_path)
            
            # Visualize predictions
            vis_save_path = os.path.join(viz_dir, f"predictions_epoch_{epoch+1}.png")
            visualize_predictions(model, full_dataset, device, num_samples=4, save_path=vis_save_path)
    
    # Create learning curves plot
    curves_save_path = os.path.join(viz_dir, "learning_curves.png")
    
    plt.figure(figsize=(16, 12))
    
    # Plot loss curves
    plt.subplot(2, 1, 1)
    plt.plot(epochs_list, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_list, val_losses, 'r-', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot metrics
    plt.subplot(2, 2, 3)
    plt.plot(epochs_list, val_damage_ious, 'g-', label='Damage IoU')
    plt.title('Damage Segmentation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs_list, val_disaster_accs, 'm-', label='Disaster Accuracy')
    plt.title('Disaster Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(curves_save_path)
    plt.close()
    
    # Copy the best model to the main directory
    if os.path.exists(best_model_path):
        final_best_path = os.path.join(output_dir, "disaster_model_best.pt")
        import shutil
        shutil.copy2(best_model_path, final_best_path)
        print(f"Best model copied to {final_best_path}")
    
    # Save final metrics
    metrics = {
        "epochs": epochs_list,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_damage_ious": val_damage_ious,
        "val_disaster_accs": val_disaster_accs,
    }
    
    metrics_path = os.path.join(run_dir, f"training_metrics_run{run_num}.txt")
    with open(metrics_path, "w") as f:
        for key, values in metrics.items():
            f.write(f"{key}: {values}\n")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best combined metric: {best_combined_metric:.4f}")
    print(f"All artifacts saved to {run_dir}")

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description="Train disaster-conditioned damage classifier")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/xBD", 
                        help="Path to the xBD dataset directory")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size of input images (square)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation set ratio")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Loss weighting
    parser.add_argument("--damage_weight", type=float, default=0.8,
                        help="Weight for damage segmentation loss")
    parser.add_argument("--disaster_weight", type=float, default=0.2,
                        help="Weight for disaster classification loss")
    
    # Other options
    parser.add_argument("--balance_disasters", action="store_true",
                        help="Balance dataset by disaster type")
    
    args = parser.parse_args()
    
    # Start training
    train_disaster_conditioned_model(args)

if __name__ == "__main__":
    main() 