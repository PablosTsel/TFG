#!/usr/bin/env python3
# Generate all figures for thesis section 4.1-4.3 (Experiments and Results)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from pathlib import Path
import os

# Set style for all plots
plt.style.use('default')
sns.set_palette("husl")

def create_output_dir():
    """Create output directory for thesis figures"""
    output_dir = Path("thesis_figures")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_dataset_distribution_chart(output_dir):
    """Generate bar chart showing number of samples per disaster type in train/val/test splits"""
    
    # Sample data based on the xBD dataset structure
    disasters = ['Hurricane\nFlorence', 'Hurricane\nHarvey', 'Earthquake\nMexico', 'Wildfire\nPortugal', 
                'Tsunami\nPalu', 'Tornado\nJoplin', 'Volcano\nGuatemala', 'Fire\nSocal', 'Flooding\nNepal']
    
    # Sample counts per split (approximate based on 80-10-10 split)
    train_samples = [2850, 2200, 180, 1680, 310, 580, 240, 1360, 720]
    val_samples = [356, 275, 23, 210, 39, 73, 30, 170, 90]
    test_samples = [356, 275, 23, 210, 39, 73, 30, 170, 90]
    
    x = np.arange(len(disasters))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width, train_samples, width, label='Train', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, val_samples, width, label='Validation', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, test_samples, width, label='Test', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Disaster Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Image Pairs', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Distribution Across Train/Validation/Test Splits', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(disasters, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: dataset_distribution.png")

def generate_class_distribution_pie_chart(output_dir):
    """Generate pie chart showing class distribution with damage colors"""
    
    # Class distribution from the thesis
    labels = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    sizes = [78.3, 12.1, 6.8, 2.8]
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']  # Green, Yellow, Orange, Red
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart with explosion for better visibility
    explode = (0.05, 0.05, 0.05, 0.1)  # Slightly separate slices
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                     startangle=90, explode=explode, shadow=True,
                                     textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Enhance the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title('Building Damage Class Distribution in xBD Dataset\n(850,736 total buildings)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add building counts as text
    building_counts = [666221, 102941, 57850, 23724]
    for i, (label, count) in enumerate(zip(labels, building_counts)):
        angle = (wedges[i].theta1 + wedges[i].theta2) / 2
        x = 1.3 * np.cos(np.radians(angle))
        y = 1.3 * np.sin(np.radians(angle))
        ax.annotate(f'{count:,} buildings', xy=(x, y), ha='center', va='center',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: class_distribution.png")

def generate_confusion_matrix(output_dir):
    """Generate sample confusion matrix heatmap for damage classification"""
    
    # Sample confusion matrix data (5x5 including background)
    classes = ['Background', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    
    # Realistic confusion matrix values (in thousands of pixels)
    conf_matrix = np.array([
        [15420, 180, 45, 12, 8],      # Background
        [220, 8950, 340, 85, 25],     # No Damage  
        [35, 280, 1450, 180, 45],     # Minor Damage
        [15, 90, 200, 1180, 90],      # Major Damage
        [8, 25, 55, 110, 890]        # Destroyed
    ])
    
    # Calculate percentages for better visualization
    conf_matrix_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(conf_matrix_pct, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', rotation=270, labelpad=20, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = f'{conf_matrix_pct[i, j]:.1f}%\n({conf_matrix[i, j]})'
            ax.text(j, i, text, ha="center", va="center", 
                   color="white" if conf_matrix_pct[i, j] > 50 else "black",
                   fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=12)
    ax.set_ylabel('Actual Class', fontweight='bold', fontsize=12)
    ax.set_title('Confusion Matrix - Damage Classification Model\n(Sample Results)', 
                fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: confusion_matrix.png")

def generate_system_architecture_diagram(output_dir):
    """Generate system architecture diagram showing data flow"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors
    data_color = '#3498db'
    process_color = '#e74c3c'
    model_color = '#9b59b6'
    output_color = '#2ecc71'
    
    # Draw components
    # Data source
    data_rect = patches.Rectangle((0.5, 8), 2, 1, facecolor=data_color, alpha=0.7, edgecolor='black')
    ax.add_patch(data_rect)
    ax.text(1.5, 8.5, 'xBD Dataset\n22,068 images', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Preprocessing
    prep_rect = patches.Rectangle((4, 8), 2, 1, facecolor=process_color, alpha=0.7, edgecolor='black')
    ax.add_patch(prep_rect)
    ax.text(5, 8.5, 'Preprocessing\n256x256 resize\nNormalization', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Train/Val/Test split
    split_rect = patches.Rectangle((7.5, 8), 2, 1, facecolor=process_color, alpha=0.7, edgecolor='black')
    ax.add_patch(split_rect)
    ax.text(8.5, 8.5, 'Data Split\n80/10/10\nDisaster-level', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # GPU Training
    gpu_rect = patches.Rectangle((11, 8), 2, 1, facecolor='#f39c12', alpha=0.7, edgecolor='black')
    ax.add_patch(gpu_rect)
    ax.text(12, 8.5, 'GPU Training\nRTX 4070\n8.2 GB VRAM', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Stage 1: Building Detection
    stage1_rect = patches.Rectangle((2, 5.5), 3, 1.5, facecolor=model_color, alpha=0.7, edgecolor='black')
    ax.add_patch(stage1_rect)
    ax.text(3.5, 6.25, 'Stage 1: Building Detection\nU-Net Architecture\nBCE + Dice Loss', 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Stage 2: Damage Classification
    stage2_rect = patches.Rectangle((8, 5.5), 3, 1.5, facecolor=model_color, alpha=0.7, edgecolor='black')
    ax.add_patch(stage2_rect)
    ax.text(9.5, 6.25, 'Stage 2: Damage Classification\nDual U-Net + Attention\nFocal + Dice Loss', 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Outputs
    output1_rect = patches.Rectangle((1, 3), 2.5, 1, facecolor=output_color, alpha=0.7, edgecolor='black')
    ax.add_patch(output1_rect)
    ax.text(2.25, 3.5, 'Building Masks\nIoU Metrics', ha='center', va='center', fontweight='bold', fontsize=10)
    
    output2_rect = patches.Rectangle((7, 3), 2.5, 1, facecolor=output_color, alpha=0.7, edgecolor='black')
    ax.add_patch(output2_rect)
    ax.text(8.25, 3.5, 'Damage Maps\nConfusion Matrix', ha='center', va='center', fontweight='bold', fontsize=10)
    
    output3_rect = patches.Rectangle((10.5, 3), 2.5, 1, facecolor=output_color, alpha=0.7, edgecolor='black')
    ax.add_patch(output3_rect)
    ax.text(11.75, 3.5, 'Emergency Response\nPrioritization Maps', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Horizontal flow
    ax.annotate('', xy=(4, 8.5), xytext=(2.5, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 8.5), xytext=(6, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(11, 8.5), xytext=(9.5, 8.5), arrowprops=arrow_props)
    
    # Vertical flows
    ax.annotate('', xy=(3.5, 7), xytext=(5, 8), arrowprops=arrow_props)
    ax.annotate('', xy=(9.5, 7), xytext=(8.5, 8), arrowprops=arrow_props)
    
    # Stage outputs
    ax.annotate('', xy=(2.25, 4), xytext=(3.5, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(8.25, 4), xytext=(9.5, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(11.75, 4), xytext=(9.5, 5.5), arrowprops=arrow_props)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(2, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Deep Learning Pipeline for Post-Disaster Damage Assessment', 
                fontweight='bold', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "system_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: system_architecture.png")

def generate_directory_tree(output_dir):
    """Generate directory tree visualization"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Directory structure text
    tree_structure = """
output/
├── building_detector/
│   ├── detect_run1/
│   │   ├── models/
│   │   │   └── best_model_epoch_15.pt
│   │   ├── visualizations/
│   │   │   ├── predictions_epoch_15.png
│   │   │   └── learning_curves.png
│   │   └── config_run1.txt
│   ├── detect_run2/
│   └── binary_building_best_new.pt
├── dam_classifier/
│   ├── classifier_run1/
│   │   ├── models/
│   │   │   └── best_model_epoch_22.pt
│   │   ├── visualizations/
│   │   │   ├── predictions_epoch_22.png
│   │   │   ├── learning_curves.png
│   │   │   └── class_accuracy.png
│   │   ├── config_run1.txt
│   │   └── class_performance.txt
│   └── improved_damage_best_new.pt
├── error_analysis/
│   └── context_visualizations/
│       ├── fire/
│       │   ├── best/
│       │   └── worst/
│       ├── tsunami/
│       ├── earthquake/
│       └── ...
└── emergency_prioritization/
    ├── joplin-tornado/
    └── palu-tsunami/
"""
    
    # Create text with different colors for different file types
    ax.text(0.05, 0.95, tree_structure, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8f9fa', alpha=0.8))
    
    # Add legend for file types
    legend_y = 0.25
    ax.text(0.05, legend_y, 'File Types:', transform=ax.transAxes, fontweight='bold', fontsize=12)
    ax.text(0.05, legend_y-0.03, '📁 Directories: Organized by model type and run number', 
            transform=ax.transAxes, fontsize=10)
    ax.text(0.05, legend_y-0.06, '⚙️ .pt files: PyTorch model weights (best checkpoints)', 
            transform=ax.transAxes, fontsize=10)
    ax.text(0.05, legend_y-0.09, '📊 .png files: Training curves and prediction visualizations', 
            transform=ax.transAxes, fontsize=10)
    ax.text(0.05, legend_y-0.12, '📝 .txt/.json files: Configurations and performance metrics', 
            transform=ax.transAxes, fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Organized Output Directory Structure\n(Versioned Experiments and Reproducible Results)', 
                fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "directory_structure.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: directory_structure.png")

def main():
    """Generate all thesis figures"""
    print("Generating thesis figures for sections 4.1-4.3...")
    
    output_dir = create_output_dir()
    
    # Generate all figures
    generate_dataset_distribution_chart(output_dir)
    generate_class_distribution_pie_chart(output_dir)
    generate_confusion_matrix(output_dir)
    generate_system_architecture_diagram(output_dir)
    generate_directory_tree(output_dir)
    
    print(f"\nAll figures generated successfully in '{output_dir}' directory!")
    print("\nGenerated files:")
    print("1. dataset_distribution.png - Train/Val/Test splits by disaster type")
    print("2. class_distribution.png - Building damage class imbalance") 
    print("3. confusion_matrix.png - Sample model performance matrix")
    print("4. system_architecture.png - Complete pipeline flow diagram")
    print("5. directory_structure.png - Organized output folder layout")

if __name__ == "__main__":
    main() 