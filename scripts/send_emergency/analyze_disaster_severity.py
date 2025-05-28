#!/usr/bin/env python3
# Analyze disaster severity based on ground truth data

import os
import sys
import json
import glob
from tqdm import tqdm
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

# Damage class mapping
DAMAGE_CLASS_MAP = {
    'no-damage': 1,
    'minor-damage': 2,
    'major-damage': 3,
    'destroyed': 4,
    'un-classified': 1  # Map unclassified to no-damage
}

def analyze_disaster_severity(xbd_dir):
    """
    Analyze the severity of different disaster areas based on ground truth damage labels.
    
    Args:
        xbd_dir: Path to the xBD dataset directory
    
    Returns:
        A dictionary containing damage statistics for each disaster
    """
    results = {}
    
    # Get all disaster directories
    disaster_dirs = [d for d in os.listdir(xbd_dir) 
                    if os.path.isdir(os.path.join(xbd_dir, d))
                    and not d.startswith('.')]
    
    print(f"Found {len(disaster_dirs)} disaster directories")
    
    for disaster in tqdm(disaster_dirs, desc="Analyzing disasters"):
        # Skip files, only process directories
        if not os.path.isdir(os.path.join(xbd_dir, disaster)):
            continue
            
        # Initialize counters for this disaster
        damage_counts = Counter()
        building_count = 0
        severe_ratio = 0.0
        
        # Look for post-disaster JSON files which contain damage labels
        labels_dir = os.path.join(xbd_dir, disaster, "labels")
        if not os.path.isdir(labels_dir):
            print(f"Warning: No labels directory found for {disaster}")
            continue
        
        post_json_files = glob.glob(os.path.join(labels_dir, "*_post_disaster.json"))
        
        # Process each JSON file
        for json_file in post_json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Process features from the JSON file
                features = []
                if "features" in data:
                    # Try both coordinate systems
                    if "xy" in data["features"]:
                        features = data["features"]["xy"]
                    elif "lng_lat" in data["features"]:
                        features = data["features"]["lng_lat"]
                
                # Count damage types
                for feature in features:
                    if "properties" in feature:
                        props = feature["properties"]
                        if "feature_type" in props and props["feature_type"] == "building":
                            building_count += 1
                            
                            # Get damage subtype
                            damage_type = props.get("subtype", "no-damage").lower()
                            damage_counts[damage_type] += 1
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # Calculate severity metrics
        total_buildings = sum(damage_counts.values())
        if total_buildings > 0:
            # Calculate proportion of severe damage (major + destroyed)
            major_count = damage_counts.get("major-damage", 0)
            destroyed_count = damage_counts.get("destroyed", 0)
            severe_damage = major_count + destroyed_count
            severe_ratio = severe_damage / total_buildings
            
            # Calculate weighted damage score
            # Weight: no-damage=0, minor=1, major=3, destroyed=5
            damage_score = (
                0 * damage_counts.get("no-damage", 0) +
                1 * damage_counts.get("minor-damage", 0) +
                3 * damage_counts.get("major-damage", 0) +
                5 * damage_counts.get("destroyed", 0)
            ) / total_buildings
        
        # Store results
        results[disaster] = {
            'total_buildings': total_buildings,
            'no_damage': damage_counts.get("no-damage", 0),
            'minor_damage': damage_counts.get("minor-damage", 0),
            'major_damage': damage_counts.get("major-damage", 0),
            'destroyed': damage_counts.get("destroyed", 0),
            'severe_ratio': severe_ratio,
            'damage_score': damage_score if total_buildings > 0 else 0,
        }
    
    return results


def print_disaster_rankings(results):
    """
    Print a ranking of disasters by severity.
    
    Args:
        results: Dictionary containing damage statistics for each disaster
    """
    # Convert results to DataFrame for easier manipulation
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # Sort by damage score (descending)
    df_sorted = df.sort_values('damage_score', ascending=False)
    
    # Print header
    print("\n" + "="*80)
    print("DISASTER SEVERITY RANKINGS")
    print("="*80)
    
    print("\nRanked by weighted damage score (higher = more severe damage):")
    print("-" * 100)
    print(f"{'Rank':<5} {'Disaster':<25} {'Score':<8} {'Total':<8} {'No Dmg':<8} {'Minor':<8} {'Major':<8} {'Destroyed':<10} {'Severe %':<10}")
    print("-" * 100)
    
    # Print each disaster's statistics
    for i, (disaster, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i:<5} {disaster:<25} {row['damage_score']:.2f}   {row['total_buildings']:<8} {row['no_damage']:<8} {row['minor_damage']:<8} {row['major_damage']:<8} {row['destroyed']:<10} {row['severe_ratio']*100:.1f}%")
    
    print("\n")
    print("NOTE: Damage Score is a weighted average of damage levels:")
    print("      No Damage = 0, Minor = 1, Major = 3, Destroyed = 5")
    print("      Severe % = Percentage of buildings with Major damage or Destroyed")
    
    return df_sorted


def visualize_disaster_severity(results):
    """
    Create visualizations of disaster severity.
    
    Args:
        results: Dictionary containing damage statistics for each disaster
    """
    # Convert results to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # Sort by damage score
    df_sorted = df.sort_values('damage_score', ascending=False)
    
    # Create a stacked bar chart of damage types
    plt.figure(figsize=(14, 10))
    
    # Create the stacked bars
    df_plot = df_sorted[['no_damage', 'minor_damage', 'major_damage', 'destroyed']]
    df_plot.plot(kind='bar', stacked=True, 
                color=['green', 'yellow', 'orange', 'red'],
                figsize=(14, 10))
    
    plt.title('Damage Distribution by Disaster', fontsize=16)
    plt.xlabel('Disaster', fontsize=14)
    plt.ylabel('Number of Buildings', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed'], 
               loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    out_dir = os.path.join(project_root, "output", "disaster_analysis")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "disaster_damage_distribution.png"))
    
    # Also create a bar chart of damage scores
    plt.figure(figsize=(14, 8))
    df_sorted['damage_score'].plot(kind='bar', 
                                  color=plt.cm.YlOrRd(df_sorted['damage_score'] / df_sorted['damage_score'].max()),
                                  figsize=(14, 8))
    plt.title('Disaster Severity Ranking', fontsize=16)
    plt.xlabel('Disaster', fontsize=14)
    plt.ylabel('Damage Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(out_dir, "disaster_severity_ranking.png"))
    
    print(f"Visualizations saved to {out_dir}")


def main():
    """Main function"""
    # Get the path to the xBD dataset
    xbd_dir = os.path.join(project_root, "data", "xBD")
    if not os.path.exists(xbd_dir):
        print(f"Error: xBD dataset directory not found at {xbd_dir}")
        return 1
    
    # Analyze disaster severity
    print("Analyzing disaster severity based on ground truth data...")
    results = analyze_disaster_severity(xbd_dir)
    
    # Print the rankings
    df_sorted = print_disaster_rankings(results)
    
    # Create visualizations
    visualize_disaster_severity(results)
    
    # Save results to JSON
    out_dir = os.path.join(project_root, "output", "disaster_analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "disaster_severity_analysis.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {os.path.join(out_dir, 'disaster_severity_analysis.json')}")
    
    # Return the name of the most severe disaster
    if not df_sorted.empty:
        worst_disaster = df_sorted.index[0]
        print(f"\nMost severe disaster: {worst_disaster}")
        return worst_disaster
    
    return None


if __name__ == "__main__":
    main() 