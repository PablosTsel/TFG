# Emergency Prioritization System

This system prioritizes areas for emergency response after disasters based on building damage assessment. It uses deep learning models to analyze satellite imagery and identify areas with damaged buildings that need urgent attention.

## Features

- **Building Detection**: Identifies buildings in satellite imagery
- **Damage Classification**: Classifies buildings by damage level
- **Area Prioritization**: Groups nearby areas and assigns priority levels
- **Interactive Maps**: Visualizes priority areas on interactive maps
- **Detailed Reports**: Generates PDF and text reports for emergency teams

## Prerequisites

- Python 3.6+
- PyTorch
- Required packages: folium, geopy, shapely, matplotlib, PIL

## Installation

Ensure you have all the required dependencies installed:

```bash
pip install torch torchvision folium geopy shapely matplotlib pillow tqdm
```

## Usage

### Running the Full Pipeline

The simplest way to use the system is to run the full pipeline with the `run_emergency_pipeline.py` script:

```bash
python scripts/send_emergency/run_emergency_pipeline.py data/xBD/hurricane-michael
```

This will:
1. Analyze the disaster area using pre-trained models
2. Group nearby areas and assign priorities
3. Generate visualizations and reports
4. Save all outputs to the specified directory

### Command-line Arguments

```
usage: run_emergency_pipeline.py [-h] [--building-model BUILDING_MODEL]
                                [--damage-model DAMAGE_MODEL]
                                [--output-dir OUTPUT_DIR]
                                [--max-distance MAX_DISTANCE]
                                [--building-threshold BUILDING_THRESHOLD]
                                [--text-only]
                                disaster_dir

positional arguments:
  disaster_dir          Path to disaster directory

optional arguments:
  -h, --help            show this help message and exit
  --building-model BUILDING_MODEL
                        Path to building detector model
  --damage-model DAMAGE_MODEL
                        Path to damage classifier model
  --output-dir OUTPUT_DIR
                        Output directory for all results
  --max-distance MAX_DISTANCE
                        Maximum distance (km) for grouping nearby images
  --building-threshold BUILDING_THRESHOLD
                        Threshold for building detection
  --text-only           Generate only text report (no PDF)
```

### Running Individual Components

If you prefer to run the components separately:

#### 1. Prioritize Disaster Areas

```bash
python scripts/send_emergency/prioritize.py data/xBD/hurricane-michael
```

#### 2. Generate Visualizations

```bash
python scripts/send_emergency/visualization.py output/emergency_prioritization/hurricane-michael_prioritization.json
```

#### 3. Generate Reports

```bash
python scripts/send_emergency/report_generator.py output/emergency_prioritization/hurricane-michael_prioritization.json
```

## Understanding the Output

The system will generate:

1. **JSON Results**: Detailed analysis of all images and groups
2. **Interactive Map**: HTML file with an interactive map showing priority areas
3. **Priority Charts**: Visualizations of priority and damage distributions
4. **PDF Report**: Comprehensive report with detailed information about high-priority areas
5. **Text Report**: Concise text-based summary for quick reference

## Priority Levels

Areas are classified into these priority levels:

- **High Priority** (Red): Critical damage, immediate response required
- **Medium Priority** (Yellow/Orange): Significant damage, monitor and respond as resources allow
- **Low Priority** (Green): Minimal damage, lower response priority

## Interpreting the Results

The key information to look for in the results:

1. **High Priority Areas**: Locations where buildings have sustained major damage or are destroyed
2. **Damage Distribution**: Percentage of buildings in each damage category
3. **Location Coordinates**: Precise locations for dispatching emergency teams

## Example Output

Here's what to expect in the output directory:

```
output/emergency_prioritization/
├── hurricane-michael_priority_map.html         # Interactive map
├── hurricane-michael_prioritization.json       # Raw data
├── hurricane-michael_priority_chart.png        # Priority visualization
├── hurricane-michael_damage_distribution.png   # Damage visualization
├── hurricane-michael_emergency_report.pdf      # PDF report
└── hurricane-michael_emergency_report.txt      # Text report
```

## License

This project is licensed under the MIT License. 