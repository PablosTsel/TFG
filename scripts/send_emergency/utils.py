#!/usr/bin/env python3
# Utilities for emergency prioritization

import os
import json
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from shapely import wkt
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_json_file(json_path):
    """
    Load and parse a JSON file containing building data.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        dict: Parsed JSON data
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return None


def extract_coordinates(json_data):
    """
    Extract geographic coordinates from the JSON data.
    
    Args:
        json_data: Parsed JSON data
        
    Returns:
        tuple: (latitude, longitude) of the center of the image
    """
    try:
        # Try to extract from metadata
        metadata = json_data.get('metadata', {})
        
        # If there are features with lng_lat, use the average of their coordinates
        if 'features' in json_data and 'lng_lat' in json_data['features']:
            features = json_data['features']['lng_lat']
            if features:
                lats = []
                lngs = []
                
                for feature in features:
                    if 'wkt' in feature:
                        # Parse the WKT string to get the polygon
                        polygon = wkt.loads(feature['wkt'])
                        
                        # Extract coordinates
                        if hasattr(polygon, 'exterior'):
                            coords = list(polygon.exterior.coords)
                            # Average coordinates for this polygon
                            feature_lngs = [coord[0] for coord in coords]
                            feature_lats = [coord[1] for coord in coords]
                            lngs.append(np.mean(feature_lngs))
                            lats.append(np.mean(feature_lats))
                
                if lats and lngs:
                    # Return the average of all feature centers
                    return np.mean(lats), np.mean(lngs)
        
        # If no coordinates found, return None
        return None, None
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None, None


def calculate_distance(coord1, coord2):
    """
    Calculate distance between two geographic coordinates in kilometers.
    
    Args:
        coord1: (latitude, longitude) of first point
        coord2: (latitude, longitude) of second point
        
    Returns:
        float: Distance in kilometers
    """
    if None in coord1 or None in coord2:
        return float('inf')
    
    return geodesic(coord1, coord2).kilometers


def find_image_pairs(disaster_dir):
    """
    Find all pre/post disaster image pairs in a disaster directory.
    
    Args:
        disaster_dir: Path to the disaster directory
        
    Returns:
        list: List of dictionaries containing file paths for pre/post pairs
    """
    image_pairs = []
    
    images_dir = os.path.join(disaster_dir, "images")
    labels_dir = os.path.join(disaster_dir, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Images or labels directory not found in {disaster_dir}")
        return image_pairs
    
    # Get all post-disaster image files
    post_images = [f for f in os.listdir(images_dir) if f.endswith("_post_disaster.png")]
    
    for post_image in post_images:
        # Get corresponding pre-disaster image
        pre_image = post_image.replace("_post_disaster.png", "_pre_disaster.png")
        
        # Get label files
        post_json = post_image.replace(".png", ".json")
        pre_json = pre_image.replace(".png", ".json")
        
        # Check if all files exist
        if all(os.path.exists(os.path.join(dir_path, file)) for dir_path, file in [
            (images_dir, pre_image), (images_dir, post_image), 
            (labels_dir, pre_json), (labels_dir, post_json)
        ]):
            pair = {
                "pre_img_path": os.path.join(images_dir, pre_image),
                "post_img_path": os.path.join(images_dir, post_image),
                "pre_json_path": os.path.join(labels_dir, pre_json),
                "post_json_path": os.path.join(labels_dir, post_json),
            }
            
            # Load the post-disaster JSON to extract coordinates
            post_json_data = load_json_file(os.path.join(labels_dir, post_json))
            if post_json_data:
                lat, lng = extract_coordinates(post_json_data)
                if lat is not None and lng is not None:
                    pair["latitude"] = lat
                    pair["longitude"] = lng
                    pair["filename"] = post_image
                    image_pairs.append(pair)
    
    return image_pairs


def group_nearby_images(image_pairs, max_distance_km=2.0):
    """
    Group images that are within a certain distance of each other.
    
    Args:
        image_pairs: List of image pair dictionaries with lat/lng
        max_distance_km: Maximum distance (in km) for grouping
        
    Returns:
        list: List of image groups
    """
    image_groups = []
    processed = set()
    
    for i, pair in enumerate(image_pairs):
        if i in processed:
            continue
            
        # Start a new group with this image
        group = [pair]
        processed.add(i)
        
        # Check all other unprocessed images
        coord1 = (pair.get("latitude"), pair.get("longitude"))
        if None in coord1:
            continue
            
        # Find all nearby images
        for j, other_pair in enumerate(image_pairs):
            if j in processed:
                continue
                
            coord2 = (other_pair.get("latitude"), other_pair.get("longitude"))
            if None in coord2:
                continue
                
            distance = calculate_distance(coord1, coord2)
            if distance <= max_distance_km:
                group.append(other_pair)
                processed.add(j)
        
        image_groups.append(group)
    
    return image_groups


def preprocess_image(image_path, image_size=256):
    """
    Preprocess an image for input to the models.
    
    Args:
        image_path: Path to the image file
        image_size: Size to resize the image to
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return tensor


def create_custom_colormap():
    """
    Create a custom colormap for priority visualization (red-yellow-green).
    
    Returns:
        matplotlib.colors.Colormap: Custom colormap
    """
    # Define the colors: high priority (red) -> medium priority (yellow) -> low priority (green)
    colors = [(0.8, 0, 0), (1, 0.8, 0), (0, 0.8, 0)]
    
    # Create the colormap
    cmap_name = 'priority_cmap'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=100) 