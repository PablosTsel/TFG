#!/usr/bin/env python3
# Enhanced Damage Assessment Visualization with Full Context
# Shows pre/post disaster images alongside ground truth, predictions and errors

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import torchvision.transforms as T
from tqdm import tqdm
from shapely import wkt
import glob
import random

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

# Import necessary model definitions
try:
    from scripts.training.utils import UNet
    from scripts.training.train_dam_classifier import ImprovedDamageClassifier, DAMAGE_CLASS_MAP
    from scripts.inference.evaluate_two_stage import IntegratedDamageAssessment
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure you're running this script from the project root directory")
    sys.exit(1)

# Define disaster mapping (same as in group_by_disaster.py)
DISASTER_MAPPING = {
    "fire": ["socal-fire", "woolsey-fire"],
    "tsunami": ["palu-tsunami", "sunda-tsunami"],
    "tornado": ["joplin-tornado", "moore-tornado", "tuscaloosa-tornado"],
    "wildfire": ["portugal-wildfire", "santa-rosa-wildfire"],
    "bushfire": ["pinery-bushfire"],
    "flooding": ["midwest-flooding", "nepal-flooding"],
    "earthquake": ["mexico-earthquake"],
    "volcano": ["guatemala-volcano", "lower-puna-volcano"],
    "hurricane": ["hurricane-florence", "hurricane-harvey", "hurricane-matthew", "hurricane-michael"]
}

# Create a reverse mapping for easy lookup
REVERSE_MAPPING = {}
for category, disasters in DISASTER_MAPPING.items():
    for disaster in disasters:
        REVERSE_MAPPING[disaster] = category

# Define class names and colors for visualization
CLASS_NAMES = ["Background", "No Damage", "Minor Damage", "Major Damage", "Destroyed"]
DAMAGE_COLORS = [
    [0, 0, 0],       # Background (black)
    [0, 255, 0],     # No damage (green)
    [255, 255, 0],   # Minor damage (yellow)
    [255, 165, 0],   # Major damage (orange)
    [255, 0, 0]      # Destroyed (red)
]

def create_color_mask(mask):
    """Convert a class prediction mask to an RGB color mask"""
    # Create a blank RGB image
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Set colors for each class
    for class_idx, color in enumerate(DAMAGE_COLORS):
        rgb_mask[mask == class_idx] = color
        
    return rgb_mask

def create_error_visualization(gt_mask, pred_mask):
    """Create visualization of prediction errors
    Green: Correct prediction
    Red: Wrong prediction
    """
    # Create error mask (1 where prediction matches ground truth, 0 where it doesn't)
    correct_mask = (gt_mask == pred_mask).astype(np.uint8)
    
    # Create RGB visualization
    h, w = gt_mask.shape
    error_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Green where correct
    error_vis[correct_mask == 1] = [0, 255, 0]
    
    # Red where incorrect
    error_vis[correct_mask == 0] = [255, 0, 0]
    
    return error_vis

def load_and_prepare_disaster_data(data_dir):
    """Load disaster data and organize by disaster type"""
    disaster_data = {category: [] for category in DISASTER_MAPPING.keys()}
    
    # Find all disaster folders in the data directory
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item in REVERSE_MAPPING:
            # This is a valid disaster folder
            disaster_type = REVERSE_MAPPING[item]
            
            # Look for image/label pairs
            images_dir = os.path.join(item_path, "images")
            labels_dir = os.path.join(item_path, "labels")
            
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                # Find pre/post disaster image pairs
                pre_images = glob.glob(os.path.join(images_dir, "*_pre_disaster.png"))
                
                for pre_img_path in pre_images:
                    base_name = os.path.basename(pre_img_path).replace("_pre_disaster.png", "")
                    post_img_path = os.path.join(images_dir, f"{base_name}_post_disaster.png")
                    pre_json_path = os.path.join(labels_dir, f"{base_name}_pre_disaster.json")
                    post_json_path = os.path.join(labels_dir, f"{base_name}_post_disaster.json")
                    
                    if all(os.path.exists(p) for p in [post_img_path, pre_json_path, post_json_path]):
                        disaster_data[disaster_type].append({
                            "pre_img_path": pre_img_path,
                            "post_img_path": post_img_path,
                            "pre_json_path": pre_json_path,
                            "post_json_path": post_json_path,
                            "disaster": item,
                            "disaster_type": disaster_type,
                            "img_id": base_name
                        })
    
    # Print summary of loaded data
    total_samples = sum(len(samples) for samples in disaster_data.values())
    print(f"Loaded {total_samples} total samples across {len(disaster_data)} disaster types")
    for category, samples in disaster_data.items():
        print(f"  - {category}: {len(samples)} samples")
    
    return disaster_data

def create_masks_from_json(json_path, original_size, target_size=256):
    """Create a damage mask from a JSON file"""
    # Create empty mask
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Scale factors
    scale_x = target_size / original_size[0]
    scale_y = target_size / original_size[1]
    
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Check different possible structures
        features = None
        
        # Try to find features in the JSON
        if "features" in json_data:
            if isinstance(json_data["features"], dict):
                # Try common coordinate keys
                for key in ["xy", "lng_lat"]:
                    if key in json_data["features"]:
                        features = json_data["features"][key]
                        break
            else:
                features = json_data["features"]
        else:
            # Try direct coordinate keys
            for key in ["xy", "lng_lat"]:
                if key in json_data:
                    features = json_data[key]
                    break
        
        # If still no features found and JSON is a list, use it directly
        if features is None and isinstance(json_data, list):
            features = json_data
        
        # Create PIL image for drawing
        pil_mask = Image.new("L", (target_size, target_size), 0)
        draw = ImageDraw.Draw(pil_mask)
        
        # Process features if we found any
        if features:
            for feat in features:
                if not isinstance(feat, dict):
                    continue
                    
                wkt_str = feat.get("wkt", None)
                
                # Try to get damage class from properties
                damage_class = 1  # Default to no-damage
                if "properties" in feat:
                    props = feat["properties"]
                    if "subtype" in props:
                        subtype = props["subtype"].lower()
                        damage_class = DAMAGE_CLASS_MAP.get(subtype, 1)
                
                if wkt_str:
                    try:
                        # Parse WKT and get polygon points
                        polygon = wkt.loads(wkt_str)
                        
                        # Convert coordinates to mask space
                        coords = []
                        for x, y in polygon.exterior.coords:
                            coords.append((x * scale_x, y * scale_y))
                        
                        # Draw polygon on the mask
                        if len(coords) > 2:  # Need at least 3 points for a polygon
                            draw.polygon(coords, fill=damage_class)
                    except Exception as e:
                        pass
        
        # Convert PIL image to numpy array
        mask = np.array(pil_mask)
        
    except Exception as e:
        print(f"Error creating mask from {json_path}: {e}")
    
    return mask

def denormalize_image(tensor):
    """Convert a normalized image tensor back to a numpy array for visualization"""
    # Move to CPU and convert to numpy
    img = tensor.cpu().numpy()
    # Transpose from CxHxW to HxWxC
    img = np.transpose(img, (1, 2, 0))
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    # Clip values to valid range
    img = np.clip(img, 0, 1)
    
    return img

def calculate_error_rate(gt_mask, pred_mask):
    """Calculate the error rate (percentage of pixels that are misclassified)"""
    total_pixels = gt_mask.size
    errors = np.sum(gt_mask != pred_mask)
    error_rate = errors / total_pixels
    
    return error_rate

def save_full_context_visualization(pre_img, post_img, gt_mask, pred_mask, output_path, img_id, error_rate):
    """
    Save a comprehensive visualization with full context:
    - Pre-disaster image
    - Post-disaster image
    - Ground truth damage mask
    - Predicted damage mask
    - Error visualization
    """
    # Convert masks to RGB
    gt_rgb = create_color_mask(gt_mask)
    pred_rgb = create_color_mask(pred_mask)
    
    # Create error visualization
    error_vis = create_error_visualization(gt_mask, pred_mask)
    
    # Create figure with grid layout (3×2)
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 2, hspace=0.1, wspace=0.05)
    
    # Top row: Pre-disaster and Post-disaster images
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.imshow(pre_img)
    ax1.set_title("Pre-disaster Image")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.imshow(post_img)
    ax2.set_title("Post-disaster Image")
    ax2.axis("off")
    
    # Middle row: Ground truth and Prediction
    ax3 = fig.add_subplot(grid[1, 0])
    ax3.imshow(gt_rgb)
    ax3.set_title("Ground Truth Damage")
    ax3.axis("off")
    
    ax4 = fig.add_subplot(grid[1, 1])
    ax4.imshow(pred_rgb)
    ax4.set_title("Predicted Damage")
    ax4.axis("off")
    
    # Bottom row: Error visualization (span both columns)
    ax5 = fig.add_subplot(grid[2, :])
    ax5.imshow(error_vis)
    ax5.set_title("Prediction Errors (Red: Error, Green: Correct)")
    ax5.axis("off")
    
    # Add legend for damage classes
    legend_elements = []
    for i, (color, name) in enumerate(zip(DAMAGE_COLORS, CLASS_NAMES)):
        # Convert RGB color to matplotlib format (0-1 range)
        mpl_color = [c/255 for c in color]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=mpl_color, label=name))
    
    # Place legend on right side of the bottom plot
    ax5.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5))
    
    # Add a title with the image ID and error rate
    plt.suptitle(f"Image: {img_id} (Error Rate: {error_rate:.2%})", fontsize=16)
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def create_damage_visualizations(args):
    """Create visualizations showing full context for damage assessment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load all disaster data
    print("Loading disaster data...")
    disaster_data = load_and_prepare_disaster_data(args.data_dir)
    
    # Load the model
    try:
        model = IntegratedDamageAssessment(
            args.building_detector_path,
            args.damage_classifier_path,
            device=device
        )
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Pre-processing transform for model input
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process each disaster type
    for disaster_type, samples in disaster_data.items():
        if not samples:
            print(f"No samples for disaster type: {disaster_type}, skipping")
            continue
            
        print(f"\nProcessing disaster type: {disaster_type} with {len(samples)} samples")
        
        # Skip if there are too few samples to produce meaningful best/worst examples
        # We need at least twice the num_examples to have different best and worst sets
        if len(samples) < args.num_examples * 2:
            print(f"Warning: Not enough samples for {disaster_type} to generate distinct best/worst sets "
                  f"(found {len(samples)}, need at least {args.num_examples * 2})")
            if args.skip_small_sets:
                print(f"Skipping {disaster_type} due to insufficient samples")
                continue
            else:
                print(f"Processing with reduced examples to ensure distinct sets")
        
        # Create output directories
        disaster_output_dir = os.path.join(output_dir, disaster_type)
        best_dir = os.path.join(disaster_output_dir, "best")
        worst_dir = os.path.join(disaster_output_dir, "worst")
        
        Path(best_dir).mkdir(parents=True, exist_ok=True)
        Path(worst_dir).mkdir(parents=True, exist_ok=True)
        
        # For each disaster type, process a limited number of random samples
        selected_samples = samples
        if len(samples) > args.max_samples_per_type:
            selected_samples = random.sample(samples, args.max_samples_per_type)
            
        results = []
        
        print(f"Running predictions on {len(selected_samples)} samples...")
        for sample in tqdm(selected_samples):
            try:
                # Load pre and post disaster images
                pre_img = Image.open(sample["pre_img_path"]).convert("RGB")
                post_img = Image.open(sample["post_img_path"]).convert("RGB")
                original_size = pre_img.size
                
                # Create ground truth mask from JSON
                gt_mask = create_masks_from_json(sample["post_json_path"], original_size)
                
                # Check if this image has buildings (not just background)
                building_pixels = np.sum(gt_mask > 0)
                if building_pixels < 100:  # Skip if very few or no buildings
                    continue
                
                # Apply transforms for model input
                pre_tensor = transform(pre_img).unsqueeze(0).to(device)
                post_tensor = transform(post_img).unsqueeze(0).to(device)
                
                # Run model prediction
                with torch.no_grad():
                    building_pred, _, damage_pred = model.predict(pre_tensor, post_tensor)
                
                # Get class predictions for damage
                pred_mask = damage_pred.argmax(dim=1).cpu().numpy()[0]
                
                # Calculate error rate
                error_rate = calculate_error_rate(gt_mask, pred_mask)
                
                # Convert tensors back to images for visualization
                pre_img_np = denormalize_image(pre_tensor[0])
                post_img_np = denormalize_image(post_tensor[0])
                
                # Store results
                results.append({
                    "img_id": sample["img_id"],
                    "error_rate": error_rate,
                    "gt_mask": gt_mask,
                    "pred_mask": pred_mask,
                    "pre_img": pre_img_np,
                    "post_img": post_img_np
                })
            except Exception as e:
                print(f"Error processing sample {sample['img_id']}: {e}")
        
        if not results:
            print(f"No valid predictions for disaster type: {disaster_type}")
            continue
        
        # Sort by error rate
        results.sort(key=lambda x: x["error_rate"])
        
        # Log error rate distribution
        error_rates = [r["error_rate"] for r in results]
        print(f"Error rate statistics for {disaster_type}:")
        print(f"  Min: {min(error_rates):.4f}, Max: {max(error_rates):.4f}")
        print(f"  Mean: {np.mean(error_rates):.4f}, Median: {np.median(error_rates):.4f}")
        
        # Calculate adjusted number of examples based on available samples
        # Ensure we have distinct best and worst sets by taking at most half of available samples
        adjusted_num = min(args.num_examples, len(results) // 2)
        if adjusted_num < args.num_examples:
            print(f"Warning: Reducing examples from {args.num_examples} to {adjusted_num} "
                  f"to ensure distinct best/worst sets")
        
        # Get best and worst predictions (ensuring no overlap)
        best_examples = results[:adjusted_num]  # Lowest error rate
        worst_examples = results[-adjusted_num:]  # Highest error rate
        
        # Sanity check - ensure best and worst have different error rates
        best_ids = set(r["img_id"] for r in best_examples)
        worst_ids = set(r["img_id"] for r in worst_examples)
        overlap = best_ids.intersection(worst_ids)
        
        if overlap:
            print(f"Warning: Found {len(overlap)} overlapping samples between best and worst.")
            print("This should not happen - error rates might not have enough variance.")
            print(f"Overlapping IDs: {overlap}")
            
            # Force separation by adjusting the worst set to exclude any best IDs
            worst_examples = [r for r in results[-len(results)//2:] if r["img_id"] not in best_ids][:adjusted_num]
            
            # Double check
            if any(r["img_id"] in best_ids for r in worst_examples):
                print("Critical error: Still have overlapping samples after adjustment")
                continue
        
        # Print statistics on selected examples
        best_error_rates = [r["error_rate"] for r in best_examples]
        worst_error_rates = [r["error_rate"] for r in worst_examples]
        print(f"Selected {len(best_examples)} best examples: "
              f"error rate range {min(best_error_rates):.4f} - {max(best_error_rates):.4f}")
        print(f"Selected {len(worst_examples)} worst examples: "
              f"error rate range {min(worst_error_rates):.4f} - {max(worst_error_rates):.4f}")
        
        # Save visualizations
        print(f"Saving visualizations for {disaster_type}...")
        
        # Save best predictions
        for i, result in enumerate(best_examples):
            output_path = os.path.join(best_dir, f"best_{i+1}.png")
            save_full_context_visualization(
                result["pre_img"],
                result["post_img"],
                result["gt_mask"],
                result["pred_mask"],
                output_path,
                result["img_id"],
                result["error_rate"]
            )
        
        # Save worst predictions
        for i, result in enumerate(worst_examples):
            output_path = os.path.join(worst_dir, f"worst_{i+1}.png")
            save_full_context_visualization(
                result["pre_img"],
                result["post_img"],
                result["gt_mask"],
                result["pred_mask"],
                output_path,
                result["img_id"],
                result["error_rate"]
            )
        
        print(f"Saved {len(best_examples)} best and {len(worst_examples)} worst predictions for {disaster_type}")
    
    print(f"\nCompleted context-rich visualizations for all disaster types!")
    print(f"Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create detailed damage assessment visualizations with full context')
    parser.add_argument('--data_dir', default='data/xBD',
                       help='Path to the xBD data directory')
    parser.add_argument('--output_dir', default='error_analysis/context_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--building_detector_path', 
                       default='output/building_detector/binary_building_best_new.pt',
                       help='Path to the building detector model weights')
    parser.add_argument('--damage_classifier_path',
                       default='output/dam_classifier/improved_damage_best_new.pt',
                       help='Path to the damage classifier model weights')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of best/worst examples to save per disaster type')
    parser.add_argument('--max_samples_per_type', type=int, default=50,
                       help='Maximum number of samples to process per disaster type')
    parser.add_argument('--skip_small_sets', action='store_true',
                       help='Skip disaster types with too few samples for distinct best/worst sets')
    
    args = parser.parse_args()
    
    # Create visualizations
    create_damage_visualizations(args)

if __name__ == "__main__":
    main() 