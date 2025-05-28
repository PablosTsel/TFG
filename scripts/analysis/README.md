# Error Analysis Scripts

This directory contains scripts for analyzing model performance across different disaster types.

## Scripts Overview

1. `group_by_disaster.py`: Organizes the xBD dataset into broader disaster categories
2. `evaluate_by_disaster.py`: Evaluates model performance on each disaster category

## How to Use

### Step 1: Group Data by Disaster Type

Run the following command from the project root directory:

```bash
./scripts/analysis/group_by_disaster.py
```

This will:
- Create a directory structure in `error_analysis/`
- Group specific disasters (e.g., hurricane-harvey, hurricane-michael) into broader categories (e.g., hurricane)
- Create symbolic links to the original data
- Generate a `disaster_mapping.json` file

Optional arguments:
- `--xbd_dir`: Path to the xBD data directory (default: `data/xBD`)
- `--output_dir`: Output directory for analysis results (default: `error_analysis`)

### Step 2: Evaluate Models by Disaster Type

Run the following command from the project root directory:

```bash
./scripts/analysis/evaluate_by_disaster.py
```

This will:
- Evaluate both the building detector and damage classifier on each disaster category
- Generate performance metrics (IoU scores) for each category
- Create visualizations comparing performance across categories
- Save results to `error_analysis/quantitative/disaster_metrics.csv` and `error_analysis/quantitative/disaster_comparison.png`

Optional arguments:
- `--analysis_dir`: Path to the analysis directory (default: `error_analysis`)
- `--building_detector_path`: Path to the building detector model weights (default: `output/building_detector/binary_building_best_new.pt`)
- `--damage_classifier_path`: Path to the damage classifier model weights (default: `output/dam_classifier/improved_damage_best_new.pt`)
- `--batch_size`: Batch size for evaluation (default: 8)

## Expected Output

After running these scripts, you'll have:

1. A structured directory organization in `error_analysis/`
2. Quantitative metrics showing how model performance varies across disaster types
3. Visualizations to help identify which disaster types the model handles well or poorly

This analysis will help identify potential areas for improvement in the models and guide further development. 