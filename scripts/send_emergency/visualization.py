#!/usr/bin/env python3
# Visualization tools for emergency prioritization

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import folium
from folium.plugins import MarkerCluster, HeatMap
import argparse
from pathlib import Path
import webbrowser

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

# Import our utility functions
from scripts.send_emergency.utils import create_custom_colormap


class PrioritizationVisualizer:
    """
    Class for visualizing emergency prioritization results.
    """
    # Define color mapping for priority levels
    PRIORITY_COLORS = {
        "High": "#ff0000",    # Red
        "Medium": "#ffaa00",  # Orange
        "Low": "#00cc00",     # Green
        "Unknown": "#999999"  # Gray
    }
    
    def __init__(self, results_file, output_dir=None):
        """
        Initialize the visualizer with prioritization results.
        
        Args:
            results_file: Path to the JSON file with prioritization results
            output_dir: Directory to save visualizations (default: same as results file)
        """
        self.results_file = results_file
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(results_file))
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        # Load results
        try:
            with open(results_file, 'r') as f:
                self.results = json.load(f)
                
            self.summary = self.results["summary"]
            self.groups = self.results["groups"]
            self.image_results = self.results["image_results"]
            
            print(f"Loaded results from {results_file}")
            print(f"Total groups: {self.summary['total_groups']}")
            print(f"High priority: {self.summary['high_priority_groups']}")
            print(f"Medium priority: {self.summary['medium_priority_groups']}")
            print(f"Low priority: {self.summary['low_priority_groups']}")
            
        except Exception as e:
            print(f"Error loading results file: {e}")
            self.results = None
            self.summary = None
            self.groups = None
            self.image_results = None
    
    def create_interactive_map(self, open_browser=True):
        """
        Create an interactive map with Folium showing prioritized areas.
        
        Args:
            open_browser: Whether to automatically open the map in a browser
            
        Returns:
            str: Path to the saved HTML file
        """
        if not self.groups:
            print("No group data available for visualization")
            return None
        
        # Filter out groups without valid coordinates
        valid_groups = [g for g in self.groups if g["center_latitude"] is not None and g["center_longitude"] is not None]
        
        if not valid_groups:
            print("No groups with valid coordinates found")
            return None
        
        # Calculate map center as the average of all group coordinates
        center_lat = np.mean([g["center_latitude"] for g in valid_groups])
        center_lng = np.mean([g["center_longitude"] for g in valid_groups])
        
        # Create a map
        m = folium.Map(location=[center_lat, center_lng], zoom_start=13)
        
        # Add a legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 130px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white; padding: 10px;
                    border-radius: 5px;">
                    
          <p style="margin-top: 0"><b>Priority Levels</b></p>
          <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="background-color: #ff0000; width: 20px; height: 20px; margin-right: 10px;"></div>
            <div>High Priority</div>
          </div>
          <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="background-color: #ffaa00; width: 20px; height: 20px; margin-right: 10px;"></div>
            <div>Medium Priority</div>
          </div>
          <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="background-color: #00cc00; width: 20px; height: 20px; margin-right: 10px;"></div>
            <div>Low Priority</div>
          </div>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Create a marker cluster for better visualization
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each group
        for i, group in enumerate(valid_groups):
            # Set circle color based on priority
            color = self.PRIORITY_COLORS.get(group["priority_level"], self.PRIORITY_COLORS["Unknown"])
            
            # Radius proportional to priority score (min 100m, max 500m)
            radius = 100 + (group["priority_score"] * 400)
            
            # Create a circle marker
            folium.Circle(
                location=[group["center_latitude"], group["center_longitude"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.5,
                popup=self._create_popup_html(group),
                tooltip=f"Group {group['group_id']}: {group['priority_level']} Priority"
            ).add_to(m)
            
            # Add a marker with details
            folium.Marker(
                location=[group["center_latitude"], group["center_longitude"]],
                popup=self._create_popup_html(group),
                tooltip=f"Group {group['group_id']}: {group['priority_level']} Priority",
                icon=folium.Icon(color=self._get_marker_color(group["priority_level"]), icon="info-sign")
            ).add_to(marker_cluster)
        
        # Add a heatmap layer based on priority scores
        heat_data = [
            [g["center_latitude"], g["center_longitude"], g["priority_score"]]
            for g in valid_groups
        ]
        
        HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
        
        # Add a layer control
        folium.LayerControl().add_to(m)
        
        # Save the map
        disaster_name = self.summary["disaster_dir"].split("/")[-1]
        map_path = os.path.join(self.output_dir, f"{disaster_name}_priority_map.html")
        m.save(map_path)
        
        print(f"Interactive map saved to {map_path}")
        
        if open_browser:
            webbrowser.open(f"file://{os.path.abspath(map_path)}")
            
        return map_path
    
    def create_priority_chart(self):
        """
        Create a chart showing the distribution of priority levels.
        
        Returns:
            str: Path to the saved chart image
        """
        if not self.summary:
            print("No summary data available for visualization")
            return None
        
        # Get priority counts
        counts = [
            self.summary["high_priority_groups"],
            self.summary["medium_priority_groups"],
            self.summary["low_priority_groups"]
        ]
        
        labels = ["High", "Medium", "Low"]
        colors = [self.PRIORITY_COLORS[label] for label in labels]
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, counts, color=colors)
        
        # Add labels and title
        plt.xlabel("Priority Level")
        plt.ylabel("Number of Areas")
        plt.title("Distribution of Priority Levels")
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom"
            )
        
        # Save the chart
        disaster_name = self.summary["disaster_dir"].split("/")[-1]
        chart_path = os.path.join(self.output_dir, f"{disaster_name}_priority_chart.png")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        
        print(f"Priority chart saved to {chart_path}")
        return chart_path
    
    def create_damage_distribution_chart(self):
        """
        Create a chart showing the overall damage distribution.
        
        Returns:
            str: Path to the saved chart image
        """
        if not self.groups:
            print("No group data available for visualization")
            return None
        
        # Aggregate damage counts across all groups
        damage_counts = {str(i): 0 for i in range(5)}
        
        for group in self.groups:
            group_counts = group.get("damage_counts", {})
            for cls, count in group_counts.items():
                damage_counts[cls] += count
        
        # Calculate percentages
        total = sum(damage_counts.values())
        if total == 0:
            print("No damage data available")
            return None
            
        damage_percentage = {int(cls): (count / total) * 100 for cls, count in damage_counts.items()}
        
        # Damage class labels
        labels = ["Background", "No Damage", "Minor Damage", "Major Damage", "Destroyed"]
        colors = ["gray", "green", "yellow", "orange", "red"]
        
        # Create the chart
        plt.figure(figsize=(12, 7))
        
        # Only plot non-zero values
        values = [damage_percentage.get(i, 0) for i in range(5)]
        non_zero_indices = [i for i, v in enumerate(values) if v > 0]
        
        bars = plt.bar(
            [labels[i] for i in non_zero_indices],
            [values[i] for i in non_zero_indices],
            color=[colors[i] for i in non_zero_indices]
        )
        
        # Add labels and title
        plt.xlabel("Damage Class")
        plt.ylabel("Percentage (%)")
        plt.title("Distribution of Damage Classes")
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom"
            )
        
        # Save the chart
        disaster_name = self.summary["disaster_dir"].split("/")[-1]
        chart_path = os.path.join(self.output_dir, f"{disaster_name}_damage_distribution.png")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        
        print(f"Damage distribution chart saved to {chart_path}")
        return chart_path
    
    def generate_all_visualizations(self):
        """
        Generate all available visualizations.
        
        Returns:
            list: Paths to all generated visualization files
        """
        paths = []
        
        # Create interactive map
        map_path = self.create_interactive_map(open_browser=False)
        if map_path:
            paths.append(map_path)
            
        # Create priority chart
        chart_path = self.create_priority_chart()
        if chart_path:
            paths.append(chart_path)
            
        # Create damage distribution chart
        damage_chart_path = self.create_damage_distribution_chart()
        if damage_chart_path:
            paths.append(damage_chart_path)
        
        return paths
    
    def _create_popup_html(self, group):
        """Create HTML content for map popups"""
        damage_distribution = group.get("damage_distribution", {})
        
        # Calculate percentages for each damage class
        damage_percentages = {}
        for cls in range(5):
            cls_str = str(cls)
            if cls_str in group.get("damage_counts", {}):
                count = group["damage_counts"][cls_str]
                total = sum(group["damage_counts"].values())
                if total > 0:
                    damage_percentages[cls] = (count / total) * 100
                else:
                    damage_percentages[cls] = 0
            else:
                damage_percentages[cls] = 0
        
        # HTML content
        content = f"""
        <div style="min-width:300px;">
            <h3>Area Group #{group['group_id']}</h3>
            <p><strong>Priority:</strong> <span style="color:{self.PRIORITY_COLORS[group['priority_level']]};">
                {group['priority_level']} (Score: {group['priority_score']:.2f})
            </span></p>
            <p><strong>Estimated Buildings:</strong> {group['estimated_buildings']}</p>
            <p><strong>Images in Group:</strong> {group['num_images']}</p>
            
            <h4>Damage Distribution:</h4>
            <table style="width:100%;">
                <tr>
                    <th>Damage Type</th>
                    <th>Percentage</th>
                </tr>
                <tr>
                    <td>No Damage</td>
                    <td>{damage_percentages.get(1, 0):.1f}%</td>
                </tr>
                <tr>
                    <td>Minor Damage</td>
                    <td>{damage_percentages.get(2, 0):.1f}%</td>
                </tr>
                <tr>
                    <td>Major Damage</td>
                    <td>{damage_percentages.get(3, 0):.1f}%</td>
                </tr>
                <tr>
                    <td>Destroyed</td>
                    <td>{damage_percentages.get(4, 0):.1f}%</td>
                </tr>
            </table>
            
            <p><strong>Location:</strong> {group['center_latitude']:.6f}, {group['center_longitude']:.6f}</p>
        </div>
        """
        return content
    
    def _get_marker_color(self, priority_level):
        """Get marker color name from priority level"""
        if priority_level == "High":
            return "red"
        elif priority_level == "Medium":
            return "orange"
        elif priority_level == "Low":
            return "green"
        else:
            return "gray"


def main():
    """Main function to run the visualization tools"""
    parser = argparse.ArgumentParser(description="Visualization Tools for Emergency Prioritization")
    parser.add_argument('results_file', type=str, help='Path to prioritization results JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.results_file):
        print(f"Error: Results file {args.results_file} does not exist")
        return
    
    # Initialize visualizer
    visualizer = PrioritizationVisualizer(
        results_file=args.results_file,
        output_dir=args.output_dir
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    # Create and open interactive map
    visualizer.create_interactive_map(open_browser=True)


if __name__ == "__main__":
    main() 