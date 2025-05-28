#!/usr/bin/env python3
# Run the entire emergency prioritization pipeline

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)


def run_command(cmd, description=None):
    """Run a command and print output"""
    if description:
        print(f"\n[{description}]")
    
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Command failed with error code {result.returncode}")
        print(f"Error: {result.stderr}")
        return False
    
    print(f"Command completed in {elapsed_time:.2f} seconds")
    print(result.stdout)
    return True


def run_pipeline(args):
    """Run the complete emergency prioritization pipeline"""
    # Extract disaster name from the path
    disaster_name = os.path.basename(args.disaster_dir)
    
    print("\n" + "="*80)
    print(f"EMERGENCY PRIORITIZATION PIPELINE: {disaster_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Create main output directory
    base_output_dir = args.output_dir
    if base_output_dir is None:
        base_output_dir = os.path.join(project_root, "output", "emergency_prioritization")
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create disaster-specific output directory
    disaster_output_dir = os.path.join(base_output_dir, disaster_name)
    os.makedirs(disaster_output_dir, exist_ok=True)
    print(f"Using output directory: {disaster_output_dir}")
    
    # Step 1: Run prioritization
    prioritize_cmd = [
        sys.executable,
        os.path.join(script_dir, "prioritize.py"),
        args.disaster_dir,
        "--building-model", args.building_model,
        "--damage-model", args.damage_model,
        "--output-dir", disaster_output_dir,
        "--max-distance", str(args.max_distance),
        "--building-threshold", str(args.building_threshold)
    ]
    
    if not run_command(prioritize_cmd, "Step 1: Prioritize disaster areas"):
        print("Prioritization failed. Exiting pipeline.")
        return False
    
    # Find the generated results file
    results_file = os.path.join(disaster_output_dir, f"{disaster_name}_prioritization.json")
    
    if not os.path.exists(results_file):
        print(f"Results file not found at {results_file}. Exiting pipeline.")
        return False
    
    # Step 2: Generate visualizations
    visualize_cmd = [
        sys.executable,
        os.path.join(script_dir, "visualization.py"),
        results_file,
        "--output-dir", disaster_output_dir
    ]
    
    if not run_command(visualize_cmd, "Step 2: Generate visualizations"):
        print("Visualization generation failed, but continuing with pipeline...")
    
    # Step 3: Generate reports
    report_type = "--text-only" if args.text_only else ""
    report_cmd = [
        sys.executable,
        os.path.join(script_dir, "report_generator.py"),
        results_file,
        "--output-dir", disaster_output_dir
    ]
    
    if report_type:
        report_cmd.append(report_type)
    
    if not run_command(report_cmd, "Step 3: Generate reports"):
        print("Report generation failed, but continuing...")
    
    print("\n" + "="*80)
    print("EMERGENCY PRIORITIZATION PIPELINE COMPLETED")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {disaster_output_dir}")
    print("="*80 + "\n")
    
    return True


def main():
    """Main function to run the emergency prioritization pipeline"""
    parser = argparse.ArgumentParser(description="Emergency Prioritization Pipeline")
    parser.add_argument('disaster_dir', type=str, help='Path to disaster directory')
    parser.add_argument('--building-model', type=str, 
                       default=os.path.join(project_root, "output", "building_detector", "binary_building_best_new.pt"),
                       help='Path to building detector model')
    parser.add_argument('--damage-model', type=str,
                       default=os.path.join(project_root, "output", "dam_classifier", "improved_damage_best_new.pt"),
                       help='Path to damage classifier model')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for all results')
    parser.add_argument('--max-distance', type=float, default=1.0,
                       help='Maximum distance (km) for grouping nearby images')
    parser.add_argument('--building-threshold', type=float, default=0.5,
                       help='Threshold for building detection')
    parser.add_argument('--text-only', action='store_true',
                       help='Generate only text report (no PDF)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.disaster_dir):
        print(f"Error: Disaster directory {args.disaster_dir} does not exist")
        return 1
        
    if not os.path.exists(args.building_model):
        print(f"Error: Building detector model {args.building_model} does not exist")
        return 1
        
    if not os.path.exists(args.damage_model):
        print(f"Error: Damage classifier model {args.damage_model} does not exist")
        return 1
    
    # Run the pipeline
    success = run_pipeline(args)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 