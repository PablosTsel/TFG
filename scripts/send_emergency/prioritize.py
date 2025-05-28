#!/usr/bin/env python3
# Emergency Prioritization System for Disaster Response

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import datetime  # Add this import for datetime operations

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

# Import our utility functions
from scripts.send_emergency.utils import (
    find_image_pairs, group_nearby_images, preprocess_image,
    load_json_file, extract_coordinates
)

# Import model definitions
from scripts.training.utils import UNet
from scripts.training.train_dam_classifier import ImprovedDamageClassifier


class EmergencyPrioritizer:
    """
    Class for analyzing disaster images and prioritizing areas for emergency response.
    """
    # Map damage class to severity score (higher = more severe)
    DAMAGE_SEVERITY = {
        0: 0,    # Background
        1: 0,    # No damage
        2: 0.35, # Minor damage
        3: 0.7,  # Major damage
        4: 1.0   # Destroyed
    }
    
    # Priority levels mapping (score range -> priority level)
    PRIORITY_LEVELS = {
        (0, 0.25): "Low",
        (0.25, 0.55): "Medium",
        (0.55, 1.0): "High"
    }
    
    def __init__(
        self,
        building_detector_path,
        damage_classifier_path,
        output_dir=None,
        device=None,
        building_threshold=0.5
    ):
        """
        Initialize the prioritizer with pre-trained models.
        
        Args:
            building_detector_path: Path to the saved building detector model
            damage_classifier_path: Path to the saved damage classifier model
            output_dir: Directory to save results (default: output/emergency_prioritization)
            device: Device to run inference on (default: GPU if available)
            building_threshold: Threshold for building detection
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.building_threshold = building_threshold
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(project_root, "output", "emergency_prioritization")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Load building detector
        self.building_detector = UNet(in_channels=3, out_channels=1).to(self.device)
        self.building_detector.load_state_dict(
            torch.load(building_detector_path, map_location=self.device)
        )
        self.building_detector.eval()
        print("Building detector loaded successfully")
        
        # Load damage classifier
        self.damage_classifier = ImprovedDamageClassifier(in_channels=3, out_channels=5).to(self.device)
        self.damage_classifier.load_state_dict(
            torch.load(damage_classifier_path, map_location=self.device)
        )
        self.damage_classifier.eval()
        print("Damage classifier loaded successfully")
    
    def analyze_image_pair(self, pre_img_path, post_img_path):
        """
        Analyze a pre/post disaster image pair to detect buildings and assess damage.
        
        Args:
            pre_img_path: Path to pre-disaster image
            post_img_path: Path to post-disaster image
            
        Returns:
            dict: Analysis results with building count, damage statistics, and priority score
        """
        try:
            # Preprocess images
            pre_img = preprocess_image(pre_img_path).to(self.device)
            post_img = preprocess_image(post_img_path).to(self.device)
            
            # Get building mask
            with torch.no_grad():
                building_logits = self.building_detector(pre_img)
                building_mask = (torch.sigmoid(building_logits) > self.building_threshold).float()
                
                # Get damage classification
                damage_logits = self.damage_classifier(pre_img, post_img)
                damage_probs = torch.softmax(damage_logits, dim=1)
                _, damage_pred = torch.max(damage_probs, dim=1)
            
            # Move tensors to CPU for processing
            building_mask = building_mask.squeeze().cpu().numpy()
            damage_pred = damage_pred.squeeze().cpu().numpy()
            
            # Calculate statistics
            building_pixels = np.sum(building_mask > 0.5)
            total_pixels = building_mask.size
            building_coverage = building_pixels / total_pixels
            
            # Count pixels of each damage class
            damage_counts = defaultdict(int)
            damage_pixels = 0
            
            # Only count damage for pixels that are buildings
            for cls in range(5):
                cls_pixels = np.sum((damage_pred == cls) & (building_mask > 0.5))
                damage_counts[cls] = cls_pixels
                if cls > 0:  # Count non-background pixels
                    damage_pixels += cls_pixels
            
            # Avoid division by zero
            if building_pixels == 0:
                damage_distribution = {cls: 0 for cls in range(5)}
            else:
                damage_distribution = {cls: count / building_pixels for cls, count in damage_counts.items()}
            
            # Calculate priority score based on damage distribution
            priority_score = 0
            for cls, ratio in damage_distribution.items():
                priority_score += ratio * self.DAMAGE_SEVERITY[cls]
            
            # Adjust priority score by building coverage (more buildings = higher priority)
            adjusted_score = priority_score * min(1.0, building_coverage * 10)
            
            # Determine priority level
            priority_level = "Unknown"
            for (min_score, max_score), level in self.PRIORITY_LEVELS.items():
                if min_score <= adjusted_score < max_score:
                    priority_level = level
            
            # Estimated building count (assuming average building is ~1000 pixels)
            est_building_count = max(1, int(building_pixels / 1000))
            
            return {
                "building_coverage": building_coverage,
                "building_pixels": int(building_pixels),
                "estimated_buildings": est_building_count,
                "damage_distribution": damage_distribution,
                "priority_score": float(adjusted_score),
                "priority_level": priority_level,
                "damage_counts": {cls: int(count) for cls, count in damage_counts.items()}
            }
            
        except Exception as e:
            print(f"Error analyzing image pair: {e}")
            return {
                "error": str(e),
                "building_coverage": 0,
                "building_pixels": 0,
                "estimated_buildings": 0,
                "damage_distribution": {cls: 0 for cls in range(5)},
                "priority_score": 0,
                "priority_level": "Low",
                "damage_counts": {cls: 0 for cls in range(5)}
            }
    
    def prioritize_disaster_area(self, disaster_dir, max_distance_km=1.0):
        """
        Analyze an entire disaster directory and prioritize areas for emergency response.
        
        Args:
            disaster_dir: Path to disaster directory
            max_distance_km: Maximum distance for grouping nearby images
            
        Returns:
            dict: Results including priorities and statistics
        """
        print(f"Analyzing disaster directory: {disaster_dir}")
        
        # Find all image pairs
        image_pairs = find_image_pairs(disaster_dir)
        print(f"Found {len(image_pairs)} valid image pairs")
        
        if not image_pairs:
            print("No valid image pairs found. Exiting.")
            return None
        
        # Analyze each image pair
        results = {}
        for pair in tqdm(image_pairs, desc="Analyzing images"):
            pair_id = os.path.basename(pair["post_img_path"])
            analysis = self.analyze_image_pair(pair["pre_img_path"], pair["post_img_path"])
            
            # Add coordinates
            analysis["latitude"] = pair.get("latitude")
            analysis["longitude"] = pair.get("longitude")
            
            # Add to results
            results[pair_id] = analysis
        
        # Group nearby images
        print("Grouping nearby images...")
        image_groups = group_nearby_images(
            [{"filename": k, **v} for k, v in results.items() if "latitude" in v and "longitude" in v],
            max_distance_km=max_distance_km
        )
        
        print(f"Created {len(image_groups)} image groups")
        
        # Calculate group priorities
        group_results = []
        for i, group in enumerate(image_groups):
            # Skip empty groups
            if not group:
                continue
                
            # Calculate average coordinates
            lats = [img.get("latitude") for img in group if img.get("latitude") is not None]
            lngs = [img.get("longitude") for img in group if img.get("longitude") is not None]
            
            avg_lat = np.mean(lats) if lats else None
            avg_lng = np.mean(lngs) if lngs else None
            
            # Calculate group priority as the maximum of individual priorities
            group_priority_score = max([img.get("priority_score", 0) for img in group])
            
            # Determine priority level
            group_priority_level = "Unknown"
            for (min_score, max_score), level in self.PRIORITY_LEVELS.items():
                if min_score <= group_priority_score < max_score:
                    group_priority_level = level
            
            # Count buildings and calculate damage distributions
            total_buildings = sum([img.get("estimated_buildings", 0) for img in group])
            
            # Aggregate damage counts across all images in the group
            group_damage_counts = defaultdict(int)
            for img in group:
                for cls, count in img.get("damage_counts", {}).items():
                    group_damage_counts[cls] += count
            
            # Calculate group damage distribution
            total_damage_pixels = sum(group_damage_counts.values())
            if total_damage_pixels > 0:
                group_damage_distribution = {
                    cls: count / total_damage_pixels 
                    for cls, count in group_damage_counts.items()
                }
            else:
                group_damage_distribution = {cls: 0 for cls in range(5)}
                
            # Image filenames in this group
            image_filenames = [img.get("filename") for img in group]
            
            group_results.append({
                "group_id": i + 1,
                "center_latitude": avg_lat,
                "center_longitude": avg_lng,
                "priority_score": group_priority_score,
                "priority_level": group_priority_level,
                "num_images": len(group),
                "estimated_buildings": total_buildings,
                "damage_distribution": group_damage_distribution,
                "damage_counts": {str(cls): count for cls, count in group_damage_counts.items()},
                "image_filenames": image_filenames
            })
        
        # Sort groups by priority score (descending)
        group_results.sort(key=lambda x: x["priority_score"], reverse=True)
        
        # Calculate overall statistics
        total_images = len(image_pairs)
        total_groups = len(group_results)
        high_priority_groups = sum(1 for g in group_results if g["priority_level"] == "High")
        medium_priority_groups = sum(1 for g in group_results if g["priority_level"] == "Medium")
        low_priority_groups = sum(1 for g in group_results if g["priority_level"] == "Low")
        
        # Create a summary of the analysis
        summary = {
            "disaster_dir": disaster_dir,
            "total_images": total_images,
            "total_groups": total_groups,
            "high_priority_groups": high_priority_groups,
            "medium_priority_groups": medium_priority_groups,
            "low_priority_groups": low_priority_groups,
            "max_distance_km": max_distance_km,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save results
        disaster_name = os.path.basename(disaster_dir)
        results_path = os.path.join(self.output_dir, f"{disaster_name}_prioritization.json")
        
        with open(results_path, 'w') as f:
            json.dump({
                "summary": summary,
                "groups": group_results,
                "image_results": results
            }, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        return {
            "summary": summary,
            "groups": group_results,
            "image_results": results
        }


def main():
    """Main function to run the emergency prioritization system"""
    parser = argparse.ArgumentParser(description="Emergency Prioritization System for Disaster Response")
    parser.add_argument('disaster_dir', type=str, help='Path to disaster directory')
    parser.add_argument('--building-model', type=str, 
                       default=os.path.join(project_root, "output", "building_detector", "binary_building_best_new.pt"),
                       help='Path to building detector model')
    parser.add_argument('--damage-model', type=str,
                       default=os.path.join(project_root, "output", "dam_classifier", "improved_damage_best_new.pt"),
                       help='Path to damage classifier model')
    parser.add_argument('--output-dir', type=str, 
                       default=os.path.join(project_root, "output", "emergency_prioritization"),
                       help='Output directory for results')
    parser.add_argument('--max-distance', type=float, default=1.0,
                       help='Maximum distance (km) for grouping nearby images')
    parser.add_argument('--building-threshold', type=float, default=0.5,
                       help='Threshold for building detection')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.disaster_dir):
        print(f"Error: Disaster directory {args.disaster_dir} does not exist")
        return
        
    if not os.path.exists(args.building_model):
        print(f"Error: Building detector model {args.building_model} does not exist")
        return
        
    if not os.path.exists(args.damage_model):
        print(f"Error: Damage classifier model {args.damage_model} does not exist")
        return
    
    # Initialize prioritizer
    prioritizer = EmergencyPrioritizer(
        building_detector_path=args.building_model,
        damage_classifier_path=args.damage_model,
        output_dir=args.output_dir,
        building_threshold=args.building_threshold
    )
    
    # Run prioritization
    prioritizer.prioritize_disaster_area(
        disaster_dir=args.disaster_dir,
        max_distance_km=args.max_distance
    )


if __name__ == "__main__":
    main() 