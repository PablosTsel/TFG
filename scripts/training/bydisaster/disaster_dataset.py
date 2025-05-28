#!/usr/bin/env python3
# Disaster-aware dataset class
# This dataset returns both damage masks and disaster type labels

import os
import sys
import torch
import numpy as np
import random
from PIL import Image, ImageDraw
import torchvision.transforms as T
import json
from shapely import wkt
from torch.utils.data import Dataset

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

# Import necessary components
from scripts.training.train_dam_classifier import DAMAGE_CLASS_MAP
from scripts.training.bydisaster.disaster_model import DISASTER_TYPES

# Define a mapping of specific disasters to broader categories
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

class DisasterAwareDataset(Dataset):
    """
    Dataset for damage classification that:
    - Loads pre-disaster and post-disaster satellite images
    - Creates multi-class masks for damage levels
    - Returns disaster type labels
    """
    def __init__(self, 
                 root_dir,
                 image_size=256,
                 use_xy=True,
                 max_samples=None,
                 augment=False,
                 disaster_type_balance=False):
        """
        Initialize the damage classification dataset with disaster awareness.
        
        Args:
            root_dir: Directory where xBD data is stored
            image_size: Size of the input/output images (square)
            use_xy: Use 'xy' coordinates if True; else use 'lng_lat'
            max_samples: Optional limit on the number of samples
            augment: If True, applies data augmentation
            disaster_type_balance: If True, tries to balance disaster type distribution
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.coord_key = "xy" if use_xy else "lng_lat"
        self.max_samples = max_samples
        self.augment = augment
        self.disaster_type_balance = disaster_type_balance
        
        # Transforms for the input images
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        # Gather all samples from the dataset directory
        self.samples = self._gather_samples()
        
        # Get disaster type distribution
        self.disaster_counts = self._count_disaster_types()
        
        # Apply disaster type balancing if requested
        if self.disaster_type_balance:
            self.samples = self._balance_disaster_types()
        
        # Limit the number of samples if specified
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples = random.sample(self.samples, self.max_samples)
            
        print(f"Loaded {len(self.samples)} samples for disaster-aware damage classification")
        
    def _count_disaster_types(self):
        """Count the number of samples for each disaster type"""
        counts = {disaster_type: 0 for disaster_type in DISASTER_TYPES}
        
        for sample in self.samples:
            disaster_type = sample.get("disaster_type")
            if disaster_type in counts:
                counts[disaster_type] += 1
        
        print("Disaster type distribution:")
        for disaster_type, count in counts.items():
            print(f"  {disaster_type}: {count} samples")
            
        return counts
    
    def _balance_disaster_types(self):
        """Balance the dataset by disaster type"""
        # Group samples by disaster type
        disaster_groups = {disaster_type: [] for disaster_type in DISASTER_TYPES}
        
        for sample in self.samples:
            disaster_type = sample.get("disaster_type")
            if disaster_type in disaster_groups:
                disaster_groups[disaster_type].append(sample)
        
        # Find the minimum non-zero count
        min_count = min([len(group) for group in disaster_groups.values() if len(group) > 0])
        
        # Oversample small classes, undersample large classes
        balanced_samples = []
        for disaster_type, group in disaster_groups.items():
            if len(group) == 0:
                continue
                
            if len(group) <= min_count:
                # Oversample small classes
                balanced_samples.extend(group)
                # Add duplicates if needed
                if len(group) < min_count:
                    additional = random.choices(group, k=min_count - len(group))
                    balanced_samples.extend(additional)
            else:
                # Undersample large classes
                balanced_samples.extend(random.sample(group, min_count))
        
        print(f"Balanced dataset: {len(balanced_samples)} samples")
        return balanced_samples
    
    def _gather_samples(self):
        """
        Parse the dataset directory to find all valid pre/post-disaster image pairs
        with corresponding label files, and identify their disaster types
        """
        samples = []
        
        # Process hierarchical directory structure
        try:
            # List all disaster directories
            disasters = [d for d in os.listdir(self.root_dir)
                       if os.path.isdir(os.path.join(self.root_dir, d))
                       and d.lower() != "spacenet_gt"]
            
            print(f"Found {len(disasters)} disaster folders")
            
            # Process each disaster folder
            for disaster in disasters:
                disaster_dir = os.path.join(self.root_dir, disaster)
                images_dir = os.path.join(disaster_dir, "images")
                labels_dir = os.path.join(disaster_dir, "labels")
                
                # Skip if directories don't exist
                if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
                    print(f"Warning: Missing images or labels directory for disaster: {disaster}")
                    continue
                
                # Determine disaster type
                disaster_type = REVERSE_MAPPING.get(disaster, None)
                if disaster_type is None:
                    print(f"Warning: Unknown disaster type for: {disaster}")
                    continue
                
                # Find all post-disaster JSON files (which contain damage labels)
                post_label_files = [f for f in os.listdir(labels_dir) if f.endswith("_post_disaster.json")]
                print(f"Disaster {disaster} ({disaster_type}): Found {len(post_label_files)} label files")
                
                # Process each label file
                for post_label_file in post_label_files:
                    base_id = post_label_file.replace("_post_disaster.json", "")
                    post_img_name = base_id + "_post_disaster.png"
                    pre_img_name = base_id + "_pre_disaster.png"
                    pre_json_path = os.path.join(labels_dir, base_id + "_pre_disaster.json")
                    
                    post_json_path = os.path.join(labels_dir, post_label_file)
                    post_img_path = os.path.join(images_dir, post_img_name)
                    pre_img_path = os.path.join(images_dir, pre_img_name)
                    
                    # Skip if files don't exist
                    if not (os.path.isfile(post_json_path) and os.path.isfile(post_img_path) and 
                            os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                        continue
                    
                    # Convert disaster type to integer label (index in DISASTER_TYPES)
                    disaster_label = DISASTER_TYPES.index(disaster_type)
                    
                    samples.append({
                        "pre_img_path": pre_img_path,
                        "post_img_path": post_img_path,
                        "pre_json_path": pre_json_path,
                        "post_json_path": post_json_path,
                        "disaster": disaster,
                        "disaster_type": disaster_type,
                        "disaster_label": disaster_label
                    })
        except Exception as e:
            print(f"Error gathering samples: {e}")
        
        return samples
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        Returns:
            pre_image_tensor: Pre-disaster image tensor
            post_image_tensor: Post-disaster image tensor
            damage_mask: Multi-class damage mask with 5 classes
            disaster_label: Integer label for disaster type
        """
        item = self.samples[idx]
        pre_img_path = item["pre_img_path"]
        post_img_path = item["post_img_path"]
        pre_json_path = item["pre_json_path"]
        post_json_path = item["post_json_path"]
        disaster_label = item["disaster_label"]
        
        try:
            # Load the pre-disaster and post-disaster images
            pre_img = Image.open(pre_img_path).convert("RGB")
            post_img = Image.open(post_img_path).convert("RGB")
            original_size = pre_img.size  # (width, height)
            
            # Create an empty damage mask (0 = background/non-building)
            damage_mask = Image.new("L", original_size, 0)
            draw = ImageDraw.Draw(damage_mask)
            
            # Load building polygons and damage classes
            with open(pre_json_path, 'r') as f:
                pre_json_data = json.load(f)
                
            with open(post_json_path, 'r') as f:
                post_json_data = json.load(f)
            
            # Map building IDs to damage classes
            building_damage = {}
            post_feats = post_json_data.get("features", {}).get(self.coord_key, [])
            for feat in post_feats:
                uid = feat.get("properties", {}).get("uid", None)
                damage_type = feat.get("properties", {}).get("subtype", "").lower()
                if uid and damage_type in DAMAGE_CLASS_MAP:
                    building_damage[uid] = DAMAGE_CLASS_MAP[damage_type]
            
            # Extract building polygons from the pre-disaster JSON
            pre_feats = pre_json_data.get("features", {}).get(self.coord_key, [])
            for feat in pre_feats:
                uid = feat.get("properties", {}).get("uid", None)
                wkt_str = feat.get("wkt", None)
                
                if not wkt_str:
                    continue
                
                # Determine damage class for this building
                # Default to no-damage (class 1) if not found in post-disaster data
                damage_class = building_damage.get(uid, 1)
                
                # Parse the WKT string to get the polygon
                polygon = wkt.loads(wkt_str)
                
                # Convert polygon to a list of (x, y) tuples for PIL's polygon drawing
                if hasattr(polygon, 'exterior'):
                    # For simple polygons with exterior coordinates
                    coords = list(polygon.exterior.coords)
                else:
                    # For multipolygons or other geometries, try to extract coordinates
                    try:
                        coords = list(polygon.coords)
                    except:
                        # Skip polygons that can't be processed
                        continue
                
                # Special handling for minor damage class to make it more prominent
                if damage_class == 2:  # Minor damage class
                    # Draw the polygon outline with width=2 first to slightly dilate it
                    # then fill with the damage class. This makes minor damage regions
                    # slightly more prominent and helps the model learn them better.
                    draw.line(coords + [coords[0]], fill=damage_class, width=2)
                    draw.polygon(coords, fill=damage_class)
                else:
                    # Standard drawing for other damage classes
                    draw.polygon(coords, fill=damage_class)
            
            # Apply augmentation if enabled (with the same transformations to both images)
            if self.augment:
                # Same random seed for both transforms to ensure consistency
                seed = np.random.randint(2147483647)
                random.seed(seed)
                torch.manual_seed(seed)
                
                # Simple rotation/flip augmentation
                if random.random() > 0.5:
                    pre_img = T.functional.hflip(pre_img)
                    post_img = T.functional.hflip(post_img)
                    damage_mask = T.functional.hflip(damage_mask)
                if random.random() > 0.5:
                    pre_img = T.functional.vflip(pre_img)
                    post_img = T.functional.vflip(post_img)
                    damage_mask = T.functional.vflip(damage_mask)
                if random.random() > 0.5:
                    angle = random.choice([90, 180, 270])
                    pre_img = T.functional.rotate(pre_img, angle)
                    post_img = T.functional.rotate(post_img, angle)
                    damage_mask = T.functional.rotate(damage_mask, angle)
                
                # Convert damage_mask to numpy array for the .any() check, if it's still a PIL Image
                damage_mask_numpy_for_check = np.array(damage_mask)

                # Enhanced augmentation for minor damage cases
                if random.random() > 0.5 and (damage_mask_numpy_for_check == 2).any():
                    # Add more diverse color augmentation for images with minor damage
                    brightness = random.uniform(0.7, 1.3)
                    contrast = random.uniform(0.7, 1.3)
                    saturation = random.uniform(0.7, 1.3)
                    hue = random.uniform(-0.1, 0.1)
                    
                    post_img = T.functional.adjust_brightness(post_img, brightness)
                    post_img = T.functional.adjust_contrast(post_img, contrast)
                    post_img = T.functional.adjust_saturation(post_img, saturation)
                    post_img = T.functional.adjust_hue(post_img, hue)
                else:
                    # Standard color augmentation
                    if random.random() > 0.5:
                        brightness = random.uniform(0.8, 1.2)
                        contrast = random.uniform(0.8, 1.2)
                        saturation = random.uniform(0.8, 1.2)
                        pre_img = T.functional.adjust_brightness(pre_img, brightness)
                        pre_img = T.functional.adjust_contrast(pre_img, contrast)
                        pre_img = T.functional.adjust_saturation(pre_img, saturation)
            
            # Apply transforms to convert images to tensors
            pre_tensor = self.image_transform(pre_img)
            post_tensor = self.image_transform(post_img)
            
            # Resize the damage mask and convert to tensor
            damage_mask = damage_mask.resize((self.image_size, self.image_size), Image.NEAREST)
            damage_tensor = torch.from_numpy(np.array(damage_mask)).long()
            
            # Create disaster type tensor
            disaster_tensor = torch.tensor(disaster_label, dtype=torch.long)
            
            return pre_tensor, post_tensor, damage_tensor, disaster_tensor
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return placeholder tensors in case of error
            pre_tensor = torch.zeros(3, self.image_size, self.image_size)
            post_tensor = torch.zeros(3, self.image_size, self.image_size)
            damage_tensor = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            disaster_tensor = torch.zeros(1, dtype=torch.long)
            return pre_tensor, post_tensor, damage_tensor, disaster_tensor 