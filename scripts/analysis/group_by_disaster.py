#!/usr/bin/env python3
# Group xBD data by disaster type
# This script creates a mapping between specific disasters and broader categories

import os
import json
import argparse
from pathlib import Path
import shutil

# Define disaster type mapping
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

def create_directory_structure(output_base_dir):
    """Create the base directory structure for error analysis"""
    # Create main directories
    Path(output_base_dir).mkdir(exist_ok=True, parents=True)
    
    # Create quantitative analysis directories
    quant_dir = os.path.join(output_base_dir, "quantitative")
    Path(quant_dir).mkdir(exist_ok=True)
    
    disaster_type_dir = os.path.join(quant_dir, "by_disaster_type")
    Path(disaster_type_dir).mkdir(exist_ok=True)
    
    # Create qualitative analysis directories
    qual_dir = os.path.join(output_base_dir, "qualitative")
    Path(qual_dir).mkdir(exist_ok=True)
    
    good_pred_dir = os.path.join(qual_dir, "good_predictions")
    Path(good_pred_dir).mkdir(exist_ok=True)
    
    poor_pred_dir = os.path.join(qual_dir, "poor_predictions")
    Path(poor_pred_dir).mkdir(exist_ok=True)
    
    # Create directories for each disaster type
    for disaster_type in DISASTER_MAPPING.keys():
        Path(os.path.join(disaster_type_dir, disaster_type)).mkdir(exist_ok=True)
        Path(os.path.join(good_pred_dir, disaster_type)).mkdir(exist_ok=True)
        Path(os.path.join(poor_pred_dir, disaster_type)).mkdir(exist_ok=True)
    
    print(f"Created directory structure in {output_base_dir}")
    return disaster_type_dir

def create_disaster_mapping_file(output_base_dir):
    """Create a JSON file that maps specific disasters to broader categories"""
    mapping_file = os.path.join(output_base_dir, "disaster_mapping.json")
    
    # Create a reverse mapping (from specific disaster to category)
    reverse_mapping = {}
    for category, disasters in DISASTER_MAPPING.items():
        for disaster in disasters:
            reverse_mapping[disaster] = category
    
    with open(mapping_file, 'w') as f:
        json.dump(reverse_mapping, f, indent=2)
    
    print(f"Created disaster mapping file: {mapping_file}")
    return mapping_file

def create_disaster_symlinks(xbd_data_dir, disaster_type_dir, mapping_file):
    """Create symbolic links from original disaster folders to category folders"""
    
    # Load the mapping
    with open(mapping_file, 'r') as f:
        disaster_to_category = json.load(f)
    
    # Check if the data directory exists
    if not os.path.exists(xbd_data_dir):
        print(f"Error: xBD data directory {xbd_data_dir} not found")
        return
    
    # Find all disaster directories in the xBD folder
    found_disasters = []
    for item in os.listdir(xbd_data_dir):
        if os.path.isdir(os.path.join(xbd_data_dir, item)) and item in disaster_to_category:
            found_disasters.append(item)
    
    print(f"Found {len(found_disasters)} disaster directories in {xbd_data_dir}")
    
    # Create symbolic links
    symlinks_created = 0
    for disaster in found_disasters:
        category = disaster_to_category[disaster]
        category_dir = os.path.join(disaster_type_dir, category)
        
        # Create a symbolic link to the original disaster folder
        src_path = os.path.abspath(os.path.join(xbd_data_dir, disaster))
        link_path = os.path.join(category_dir, disaster)
        
        # Remove any existing symlink before creating a new one
        if os.path.exists(link_path):
            os.remove(link_path)
            
        os.symlink(src_path, link_path, target_is_directory=True)
        symlinks_created += 1
    
    print(f"Created {symlinks_created} symbolic links for disaster folders")

def main():
    parser = argparse.ArgumentParser(description='Group xBD data by disaster type')
    parser.add_argument('--xbd_dir', default='data/xBD', help='Path to the xBD data directory')
    parser.add_argument('--output_dir', default='error_analysis', help='Output directory for analysis')
    
    args = parser.parse_args()
    
    print(f"Organizing disasters from {args.xbd_dir} into categories...")
    
    # Create directory structure
    disaster_type_dir = create_directory_structure(args.output_dir)
    
    # Create mapping file
    mapping_file = create_disaster_mapping_file(args.output_dir)
    
    # Create symbolic links
    create_disaster_symlinks(args.xbd_dir, disaster_type_dir, mapping_file)
    
    print("Disaster grouping complete!")
    print(f"Results stored in: {args.output_dir}")

if __name__ == "__main__":
    main() 