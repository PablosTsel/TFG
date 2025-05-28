#!/usr/bin/env python3
# Find best and worst predictions for each disaster type
# Shows ground truth, prediction, and error visualization

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import torchvision.transforms as T
from tqdm import tqdm
from shapely import wkt
import shutil
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

class DisasterDataset(Dataset):
    """Dataset that loads images from a specific disaster category"""
    def __init__(self, disaster_dir, image_size=256):
        super().__init__()
        self.disaster_dir = disaster_dir
        self.image_size = image_size
        
        # Transforms for input images
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        # Find all pre/post disaster image pairs and their corresponding labels
        self.samples = self._gather_samples()
        print(f"Loaded {len(self.samples)} samples from {disaster_dir}")
    
    def _gather_samples(self):
        """Gather all valid image pairs and labels"""
        samples = []
        
        # If this is a direct disaster folder (not a category folder)
        if os.path.exists(os.path.join(self.disaster_dir, "images")) and os.path.exists(os.path.join(self.disaster_dir, "labels")):
            samples.extend(self._process_disaster_folder(self.disaster_dir))
        else:
            # Look for disaster subfolders
            for item in os.listdir(self.disaster_dir):
                full_path = os.path.join(self.disaster_dir, item)
                if os.path.isdir(full_path) or os.path.islink(full_path):
                    # Check if this is a disaster folder with images and labels
                    if os.path.exists(os.path.join(full_path, "images")) and os.path.exists(os.path.join(full_path, "labels")):
                        samples.extend(self._process_disaster_folder(full_path))
                    
        return samples
    
    def _process_disaster_folder(self, disaster_folder):
        """Process a single disaster folder to find valid image/label pairs"""
        folder_samples = []
        
        images_dir = os.path.join(disaster_folder, "images")
        labels_dir = os.path.join(disaster_folder, "labels")
        
        # Find all pre-disaster image files
        pre_img_files = [f for f in os.listdir(images_dir) if f.endswith("_pre_disaster.png")]
        
        for pre_img_file in pre_img_files:
            base_id = pre_img_file.replace("_pre_disaster.png", "")
            post_img_file = base_id + "_post_disaster.png"
            pre_json_file = base_id + "_pre_disaster.json"
            post_json_file = base_id + "_post_disaster.json"
            
            # Check if all required files exist
            pre_img_path = os.path.join(images_dir, pre_img_file)
            post_img_path = os.path.join(images_dir, post_img_file)
            pre_json_path = os.path.join(labels_dir, pre_json_file)
            post_json_path = os.path.join(labels_dir, post_json_file)
            
            if all(os.path.isfile(p) for p in [pre_img_path, post_img_path, pre_json_path, post_json_path]):
                folder_samples.append({
                    "pre_img_path": pre_img_path,
                    "post_img_path": post_img_path,
                    "pre_json_path": pre_json_path,
                    "post_json_path": post_json_path,
                    "disaster_folder": os.path.basename(disaster_folder),
                    "img_id": base_id
                })
        
        return folder_samples
        
    def _create_masks_from_json(self, pre_json_path, post_json_path, original_size):
        """Create binary building mask and damage mask from JSON files"""
        
        # Create empty masks
        building_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        damage_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Scale factor for coordinate conversion
        scale_x = self.image_size / original_size[0]
        scale_y = self.image_size / original_size[1]
        
        # Try different possible coordinate keys
        coord_keys = ["xy", "lng_lat", "features"]
        
        try:
            # Load pre-disaster JSON to get building polygons
            with open(pre_json_path, 'r') as f:
                pre_json_data = json.load(f)
            
            # Find the correct coordinate key
            pre_features = None
            for key in coord_keys:
                if key in pre_json_data:
                    if key == "features":
                        # Handle special case for 'features' key which might contain nested structure
                        pre_features = pre_json_data.get(key, [])
                        break
                    else:
                        pre_features = pre_json_data.get(key, [])
                        break
            
            # If we didn't find features with the standard keys, check nested structure
            if pre_features is None:
                if "features" in pre_json_data and isinstance(pre_json_data["features"], dict):
                    for key in coord_keys:
                        if key in pre_json_data["features"]:
                            pre_features = pre_json_data["features"].get(key, [])
                            break
            
            # If still no features found, try to use the whole JSON as features
            if pre_features is None and isinstance(pre_json_data, list):
                pre_features = pre_json_data
            
            # Create PIL Image for drawing
            pil_building_mask = Image.new("L", (self.image_size, self.image_size), 0)
            draw = ImageDraw.Draw(pil_building_mask)
            
            # Process features if we found any
            if pre_features:
                for feat in pre_features:
                    wkt_str = None
                    # Try to get WKT from different possible locations
                    if isinstance(feat, dict):
                        wkt_str = feat.get("wkt", None)
                        
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
                                draw.polygon(coords, fill=1)
                        except Exception as e:
                            pass
            
            # Convert PIL image to numpy array
            building_mask = np.array(pil_building_mask)
            
            # Now load post-disaster JSON for damage classification
            with open(post_json_path, 'r') as f:
                post_json_data = json.load(f)
                
            # Draw damage polygons
            pil_damage_mask = Image.new("L", (self.image_size, self.image_size), 0)
            draw = ImageDraw.Draw(pil_damage_mask)
            
            # Find the correct coordinate key for post data
            post_features = None
            for key in coord_keys:
                if key in post_json_data:
                    if key == "features":
                        # Handle special case for 'features' key
                        post_features = post_json_data.get(key, [])
                        break
                    else:
                        post_features = post_json_data.get(key, [])
                        break
            
            # If we didn't find features with the standard keys, check nested structure
            if post_features is None:
                if "features" in post_json_data and isinstance(post_json_data["features"], dict):
                    for key in coord_keys:
                        if key in post_json_data["features"]:
                            post_features = post_json_data["features"].get(key, [])
                            break
            
            # If still no features found, try to use the whole JSON as features
            if post_features is None and isinstance(post_json_data, list):
                post_features = post_json_data
            
            # Process features if we found any
            if post_features:
                for feat in post_features:
                    wkt_str = None
                    damage_class = None
                    
                    # Try to get WKT and damage class from different possible locations
                    if isinstance(feat, dict):
                        wkt_str = feat.get("wkt", None)
                        properties = feat.get("properties", {})
                        if properties:
                            damage_class = properties.get("subtype", "no-damage")
                    
                    # Map damage class string to integer
                    damage_int = 1  # Default to no-damage
                    if damage_class and isinstance(damage_class, str):
                        damage_class = damage_class.lower()
                        damage_int = DAMAGE_CLASS_MAP.get(damage_class, 1)
                    
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
                                draw.polygon(coords, fill=damage_int)
                        except Exception as e:
                            pass
            
            # Convert PIL image to numpy array
            damage_mask = np.array(pil_damage_mask)
            
        except Exception as e:
            print(f"Error creating masks: {e}")
            
        return building_mask, damage_mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Load pre and post disaster images
        pre_img = Image.open(item["pre_img_path"]).convert("RGB")
        post_img = Image.open(item["post_img_path"]).convert("RGB")
        
        original_size = pre_img.size  # (width, height)
        
        # Apply transformations
        pre_tensor = self.image_transform(pre_img)
        post_tensor = self.image_transform(post_img)
        
        # Create masks from JSON files
        building_mask, damage_mask = self._create_masks_from_json(
            item["pre_json_path"], 
            item["post_json_path"],
            original_size
        )
        
        # Check if this image has buildings (not just background)
        has_buildings = np.sum(building_mask) > 0
        
        # Convert numpy arrays to tensors
        building_mask_tensor = torch.from_numpy(building_mask).unsqueeze(0).float()
        damage_mask_tensor = torch.from_numpy(damage_mask).long()
        
        return {
            "pre_img": pre_tensor,
            "post_img": post_tensor,
            "building_mask": building_mask_tensor,
            "damage_mask": damage_mask_tensor,
            "img_id": item["img_id"],
            "disaster": item["disaster_folder"],
            "has_buildings": has_buildings
        }

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

def create_color_mask(mask):
    """Convert a class prediction mask to an RGB color mask"""
    # Create a blank RGB image
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Set colors for each class
    for class_idx, color in enumerate(DAMAGE_COLORS):
        rgb_mask[mask == class_idx] = color
        
    return rgb_mask

def save_visualization(gt_mask, pred_mask, output_path, img_id, pre_img, post_img):
    """Save a visualization of ground truth, prediction, errors and original images
    
    Creates a 5-panel visualization:
    - Top row: GT damage, predicted damage, error visualization
    - Bottom row: Pre-disaster image, Post-disaster image
    """
    # Convert masks to RGB
    gt_rgb = create_color_mask(gt_mask)
    pred_rgb = create_color_mask(pred_mask)
    
    # Create error visualization
    error_vis = create_error_visualization(gt_mask, pred_mask)
    
    # Create figure with two rows - top row has 3 images, bottom row has 2 images
    fig = plt.figure(figsize=(18, 10))
    
    # Define grid spec - 2 rows, 6 columns (for better control of spacing)
    gs = plt.GridSpec(2, 6, figure=fig, height_ratios=[1, 1])
    
    # Top row - damage visualizations (3 images across 6 grid cells)
    ax1 = fig.add_subplot(gs[0, 0:2])  # Spans columns 0 and 1
    ax2 = fig.add_subplot(gs[0, 2:4])  # Spans columns 2 and 3
    ax3 = fig.add_subplot(gs[0, 4:6])  # Spans columns 4 and 5
    
    # Bottom row - original images (2 images across 6 grid cells)
    ax4 = fig.add_subplot(gs[1, 1:3])  # Spans columns 1 and 2
    ax5 = fig.add_subplot(gs[1, 3:5])  # Spans columns 3 and 4
    
    # Plot damage visualizations (top row)
    ax1.imshow(gt_rgb)
    ax1.set_title("Ground Truth Damage", fontsize=12)
    ax1.axis("off")
    
    ax2.imshow(pred_rgb)
    ax2.set_title("Predicted Damage", fontsize=12)
    ax2.axis("off")
    
    ax3.imshow(error_vis)
    ax3.set_title("Prediction Errors", fontsize=12)
    ax3.axis("off")
    
    # Plot original images (bottom row)
    ax4.imshow(pre_img)
    ax4.set_title("Pre-disaster Image", fontsize=12)
    ax4.axis("off")
    
    ax5.imshow(post_img)
    ax5.set_title("Post-disaster Image", fontsize=12)
    ax5.axis("off")
    
    # Add a title with the image ID
    plt.suptitle(f"Image: {img_id}", fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def calculate_error_rate(gt_mask, pred_mask):
    """Calculate the error rate (percentage of pixels that are misclassified)"""
    total_pixels = gt_mask.size
    errors = np.sum(gt_mask != pred_mask)
    error_rate = errors / total_pixels
    
    return error_rate

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
    for category, samples in disaster_data.items():
        print(f"Loaded {len(samples)} samples for disaster type: {category}")
    
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

def find_best_worst_predictions(args):
    """Find the best and worst predictions for each disaster type"""
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
    
    # Process each disaster type
    for disaster_type, samples in disaster_data.items():
        if not samples:
            print(f"No samples for disaster type: {disaster_type}, skipping")
            continue
            
        print(f"\nProcessing disaster type: {disaster_type} with {len(samples)} samples")
        
        # Create output directories
        disaster_output_dir = os.path.join(output_dir, disaster_type)
        best_dir = os.path.join(disaster_output_dir, "best")
        worst_dir = os.path.join(disaster_output_dir, "worst")
        
        Path(best_dir).mkdir(parents=True, exist_ok=True)
        Path(worst_dir).mkdir(parents=True, exist_ok=True)
        
        # Process more samples to ensure we have enough candidates after filtering
        max_samples_to_process = min(len(samples), 100)  # Process up to 100 samples per disaster type
        selected_samples = samples
        if len(samples) > max_samples_to_process:
            selected_samples = random.sample(samples, max_samples_to_process)
            
        results = []
        skipped_small = 0
        skipped_large = 0
        
        # Preprocess and run model on samples
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Running predictions on {len(selected_samples)} samples...")
        for sample in tqdm(selected_samples):
            try:
                # Load pre and post disaster images
                pre_img = Image.open(sample["pre_img_path"]).convert("RGB")
                post_img = Image.open(sample["post_img_path"]).convert("RGB")
                original_size = pre_img.size
                
                # Create ground truth mask
                gt_mask = create_masks_from_json(sample["post_json_path"], original_size)
                
                # Check if this image has buildings within the specified pixel range
                building_pixels = np.sum(gt_mask > 0)
                
                # Skip images outside the desired building pixel range
                if building_pixels < args.min_building_pixels:
                    skipped_small += 1
                    continue
                if building_pixels > args.max_building_pixels:
                    skipped_large += 1
                    continue
                
                # Apply transforms
                pre_tensor = transform(pre_img).unsqueeze(0).to(device)
                post_tensor = transform(post_img).unsqueeze(0).to(device)
                
                # Run model prediction
                with torch.no_grad():
                    _, _, final_pred = model.predict(pre_tensor, post_tensor)
                
                # Get class predictions
                pred_mask = final_pred.argmax(dim=1).cpu().numpy()[0]
                
                # Calculate error rate
                error_rate = calculate_error_rate(gt_mask, pred_mask)
                
                # Store results
                results.append({
                    "img_id": sample["img_id"],
                    "error_rate": error_rate,
                    "gt_mask": gt_mask,
                    "pred_mask": pred_mask,
                    "building_pixels": building_pixels,
                    "pre_img_path": sample["pre_img_path"],
                    "post_img_path": sample["post_img_path"]
                })
            except Exception as e:
                print(f"Error processing sample {sample['img_id']}: {e}")
        
        print(f"Filtered samples: {len(results)} kept, {skipped_small} skipped (too few buildings), {skipped_large} skipped (too many buildings)")
        
        if not results:
            print(f"No valid predictions for disaster type: {disaster_type} after filtering")
            continue
        
        # Sort by error rate
        results.sort(key=lambda x: x["error_rate"])
        
        # Get best and worst predictions
        # Make sure we have at least 5 examples if possible
        desired_examples = max(args.num_examples, 5)
        num_vis = min(desired_examples, len(results))
        
        # If we have fewer than desired examples but more than 0, use what we have
        if num_vis < desired_examples and num_vis > 0:
            print(f"Warning: Only found {num_vis} valid samples for {disaster_type} after filtering")
        
        best_examples = results[:num_vis]  # Lowest error rate
        worst_examples = results[-num_vis:]  # Highest error rate
        
        # Save visualizations
        print(f"Saving visualizations for {disaster_type}...")
        
        # Save best predictions with simpler filenames
        for i, result in enumerate(best_examples):
            output_path = os.path.join(best_dir, f"best_{i+1}.png")
            save_visualization(
                result["gt_mask"],
                result["pred_mask"],
                output_path,
                result["img_id"],
                Image.open(result["pre_img_path"]).convert("RGB"),
                Image.open(result["post_img_path"]).convert("RGB")
            )
        
        # Save worst predictions with simpler filenames
        for i, result in enumerate(worst_examples):
            output_path = os.path.join(worst_dir, f"worst_{i+1}.png")
            save_visualization(
                result["gt_mask"],
                result["pred_mask"],
                output_path,
                result["img_id"],
                Image.open(result["pre_img_path"]).convert("RGB"),
                Image.open(result["post_img_path"]).convert("RGB")
            )
        
        print(f"Saved {len(best_examples)} best and {len(worst_examples)} worst predictions for {disaster_type}")
        
        # Save pixel count statistics
        stats_path = os.path.join(disaster_output_dir, "pixel_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Disaster type: {disaster_type}\n")
            f.write(f"Building pixel filter: {args.min_building_pixels} to {args.max_building_pixels}\n")
            f.write(f"Total samples processed: {len(selected_samples)}\n")
            f.write(f"Samples kept: {len(results)}\n")
            f.write(f"Samples skipped (too few buildings): {skipped_small}\n")
            f.write(f"Samples skipped (too many buildings): {skipped_large}\n\n")
            
            f.write("Best Examples (lowest error rate):\n")
            for i, result in enumerate(best_examples):
                f.write(f"  {i+1}. {result['img_id']} - Building pixels: {result['building_pixels']}, Error rate: {result['error_rate']:.4f}\n")
            
            f.write("\nWorst Examples (highest error rate):\n")
            for i, result in enumerate(worst_examples):
                f.write(f"  {i+1}. {result['img_id']} - Building pixels: {result['building_pixels']}, Error rate: {result['error_rate']:.4f}\n")
    
    print("\nCompleted best and worst prediction analysis for all disaster types!")

def main():
    parser = argparse.ArgumentParser(description='Find best and worst predictions for each disaster type')
    parser.add_argument('--data_dir', default='data/xBD',
                       help='Path to the xBD data directory')
    parser.add_argument('--output_dir', default='error_analysis/best_worst',
                       help='Output directory for visualizations')
    parser.add_argument('--building_detector_path', 
                       default='output/building_detector/binary_building_best_new.pt',
                       help='Path to the building detector model weights')
    parser.add_argument('--damage_classifier_path',
                       default='output/dam_classifier/improved_damage_best_new.pt',
                       help='Path to the damage classifier model weights')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of best/worst examples to save per disaster type')
    parser.add_argument('--min_building_pixels', type=int, default=100,
                       help='Minimum number of building pixels for inclusion')
    parser.add_argument('--max_building_pixels', type=int, default=500,
                       help='Maximum number of building pixels for inclusion')
    
    args = parser.parse_args()
    
    # Find best and worst predictions
    find_best_worst_predictions(args)

if __name__ == "__main__":
    main() 