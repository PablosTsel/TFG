#!/usr/bin/env python3
# Report generator for emergency prioritization

import os
import sys
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)


class EmergencyReport:
    """
    Class for generating detailed emergency prioritization reports.
    """
    # Damage class labels
    DAMAGE_LABELS = [
        "Background",
        "No Damage",
        "Minor Damage",
        "Major Damage",
        "Destroyed"
    ]
    
    # Damage class colors
    DAMAGE_COLORS = ["gray", "green", "yellow", "orange", "red"]
    
    # Priority level colors
    PRIORITY_COLORS = {
        "High": "red",
        "Medium": "orange",
        "Low": "green",
        "Unknown": "gray"
    }
    
    def __init__(self, results_file, output_dir=None):
        """
        Initialize the report generator with prioritization results.
        
        Args:
            results_file: Path to the JSON file with prioritization results
            output_dir: Directory to save the report (default: same as results file)
        """
        self.results_file = results_file
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(results_file))
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load results
        try:
            with open(results_file, 'r') as f:
                self.results = json.load(f)
                
            self.summary = self.results["summary"]
            self.groups = self.results["groups"]
            self.image_results = self.results["image_results"]
            
            print(f"Loaded results from {results_file}")
            
        except Exception as e:
            print(f"Error loading results file: {e}")
            self.results = None
            self.summary = None
            self.groups = None
            self.image_results = None
    
    def generate_pdf_report(self):
        """
        Generate a comprehensive PDF report with prioritization results.
        
        Returns:
            str: Path to the saved PDF report
        """
        if not self.results:
            print("No results data available for report generation")
            return None
        
        # Create PDF
        disaster_name = os.path.basename(self.summary["disaster_dir"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"{disaster_name}_emergency_report_{timestamp}.pdf")
        
        with PdfPages(report_path) as pdf:
            # Cover page
            self._create_cover_page(pdf)
            
            # Summary page
            self._create_summary_page(pdf)
            
            # Priority distribution page
            self._create_priority_distribution_page(pdf)
            
            # Damage distribution page
            self._create_damage_distribution_page(pdf)
            
            # High priority areas page
            self._create_priority_areas_page(pdf, "High")
            
            # Medium priority areas page
            self._create_priority_areas_page(pdf, "Medium")
            
            # Detailed analysis pages for each high priority group
            for group in self.groups:
                if group["priority_level"] == "High":
                    self._create_group_detail_page(pdf, group)
        
        print(f"PDF report saved to {report_path}")
        return report_path
    
    def generate_text_report(self):
        """
        Generate a text-based summary report.
        
        Returns:
            str: Path to the saved text report
        """
        if not self.results:
            print("No results data available for report generation")
            return None
        
        # Create report filename
        disaster_name = os.path.basename(self.summary["disaster_dir"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"{disaster_name}_emergency_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            # Title
            f.write("="*80 + "\n")
            f.write(f"EMERGENCY PRIORITIZATION REPORT: {disaster_name.upper()}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Summary section
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Disaster Area: {self.summary['disaster_dir']}\n")
            f.write(f"Total Images Analyzed: {self.summary['total_images']}\n")
            f.write(f"Total Area Groups: {self.summary['total_groups']}\n")
            f.write(f"High Priority Areas: {self.summary['high_priority_groups']}\n")
            f.write(f"Medium Priority Areas: {self.summary['medium_priority_groups']}\n")
            f.write(f"Low Priority Areas: {self.summary['low_priority_groups']}\n\n")
            
            # Emergency response recommendations
            f.write("EMERGENCY RESPONSE RECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            
            if self.summary['high_priority_groups'] > 0:
                f.write("IMMEDIATE ACTION REQUIRED: High-priority areas identified.\n")
                f.write("Deploy emergency response teams to the high-priority locations listed below.\n\n")
            else:
                f.write("No high-priority areas identified. Monitor medium-priority areas.\n\n")
            
            # High priority areas
            f.write("HIGH PRIORITY AREAS\n")
            f.write("-"*80 + "\n")
            
            high_priority_groups = [g for g in self.groups if g["priority_level"] == "High"]
            if high_priority_groups:
                for i, group in enumerate(high_priority_groups):
                    f.write(f"Area #{i+1}\n")
                    f.write(f"  Location: {group['center_latitude']:.6f}, {group['center_longitude']:.6f}\n")
                    f.write(f"  Priority Score: {group['priority_score']:.2f}\n")
                    f.write(f"  Estimated Buildings: {group['estimated_buildings']}\n")
                    
                    # Damage breakdown
                    f.write("  Damage Breakdown:\n")
                    damage_counts = group.get("damage_counts", {})
                    total_damage = sum(int(count) for count in damage_counts.values())
                    
                    if total_damage > 0:
                        for cls in range(1, 5):  # Skip background
                            cls_str = str(cls)
                            if cls_str in damage_counts:
                                count = int(damage_counts[cls_str])
                                percentage = (count / total_damage) * 100
                                f.write(f"    {self.DAMAGE_LABELS[cls]}: {count} pixels ({percentage:.1f}%)\n")
                    
                    f.write("\n")
            else:
                f.write("No high-priority areas identified.\n\n")
            
            # Medium priority areas summary
            f.write("MEDIUM PRIORITY AREAS\n")
            f.write("-"*80 + "\n")
            
            medium_priority_groups = [g for g in self.groups if g["priority_level"] == "Medium"]
            if medium_priority_groups:
                for i, group in enumerate(medium_priority_groups):
                    f.write(f"Area #{i+1}\n")
                    f.write(f"  Location: {group['center_latitude']:.6f}, {group['center_longitude']:.6f}\n")
                    f.write(f"  Priority Score: {group['priority_score']:.2f}\n")
                    f.write(f"  Estimated Buildings: {group['estimated_buildings']}\n")
                    f.write("\n")
            else:
                f.write("No medium-priority areas identified.\n\n")
        
        print(f"Text report saved to {report_path}")
        return report_path
    
    def _create_cover_page(self, pdf):
        """Create the cover page for the PDF report"""
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Title
        disaster_name = os.path.basename(self.summary["disaster_dir"])
        plt.text(0.5, 0.8, f"Emergency Prioritization Report", 
                 ha='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.7, f"{disaster_name}", 
                 ha='center', fontsize=20)
        
        # Date and time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.text(0.5, 0.6, f"Generated on: {timestamp}", 
                 ha='center', fontsize=14)
        
        # Summary statistics
        plt.text(0.5, 0.5, f"Areas Analyzed: {self.summary['total_groups']}", 
                 ha='center', fontsize=14)
        plt.text(0.5, 0.45, f"High Priority Areas: {self.summary['high_priority_groups']}", 
                 ha='center', fontsize=14, color='red')
        plt.text(0.5, 0.4, f"Medium Priority Areas: {self.summary['medium_priority_groups']}", 
                 ha='center', fontsize=14, color='orange')
        plt.text(0.5, 0.35, f"Low Priority Areas: {self.summary['low_priority_groups']}", 
                 ha='center', fontsize=14, color='green')
        
        # Add border
        plt.gca().add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, linewidth=2))
        
        pdf.savefig()
        plt.close()
    
    def _create_summary_page(self, pdf):
        """Create the summary page for the PDF report"""
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Title
        plt.text(0.5, 0.95, "Summary of Findings", 
                ha='center', fontsize=20, weight='bold')
        
        # Disaster information
        disaster_name = os.path.basename(self.summary["disaster_dir"])
        plt.text(0.1, 0.9, f"Disaster Area: {disaster_name}", fontsize=12)
        plt.text(0.1, 0.87, f"Total Images Analyzed: {self.summary['total_images']}", fontsize=12)
        plt.text(0.1, 0.84, f"Grouping Distance: {self.summary['max_distance_km']} km", fontsize=12)
        
        # Priority statistics
        plt.text(0.1, 0.78, "Priority Distribution:", fontsize=14, weight='bold')
        plt.text(0.1, 0.75, f"High Priority Areas: {self.summary['high_priority_groups']}", 
                fontsize=12, color='red')
        plt.text(0.1, 0.72, f"Medium Priority Areas: {self.summary['medium_priority_groups']}", 
                fontsize=12, color='orange')
        plt.text(0.1, 0.69, f"Low Priority Areas: {self.summary['low_priority_groups']}", 
                fontsize=12, color='green')
        
        # Recommendations
        plt.text(0.1, 0.62, "Emergency Response Recommendations:", fontsize=14, weight='bold')
        
        if self.summary['high_priority_groups'] > 0:
            plt.text(0.1, 0.58, "IMMEDIATE ACTION REQUIRED:", fontsize=12, weight='bold', color='red')
            plt.text(0.1, 0.55, "Deploy emergency response teams to the high-priority", fontsize=12)
            plt.text(0.1, 0.52, "locations identified in this report.", fontsize=12)
            
            # Add details about the highest priority area
            high_priority_groups = [g for g in self.groups if g["priority_level"] == "High"]
            if high_priority_groups:
                top_priority = high_priority_groups[0]  # First one is highest priority
                plt.text(0.1, 0.46, "Highest Priority Area:", fontsize=12, weight='bold')
                plt.text(0.1, 0.43, f"Location: {top_priority['center_latitude']:.6f}, {top_priority['center_longitude']:.6f}", fontsize=12)
                plt.text(0.1, 0.40, f"Priority Score: {top_priority['priority_score']:.2f}", fontsize=12)
                plt.text(0.1, 0.37, f"Estimated Buildings: {top_priority['estimated_buildings']}", fontsize=12)
        else:
            plt.text(0.1, 0.58, "No high-priority areas identified.", fontsize=12)
            plt.text(0.1, 0.55, "Monitor medium-priority areas for developing situations.", fontsize=12)
        
        # Add notes
        plt.text(0.1, 0.28, "Notes:", fontsize=14, weight='bold')
        plt.text(0.1, 0.25, "- Priority levels are determined based on building damage assessment", fontsize=10)
        plt.text(0.1, 0.23, "- Areas with destroyed or major-damaged buildings are highest priority", fontsize=10)
        plt.text(0.1, 0.21, "- This analysis is based on satellite imagery and deep learning models", fontsize=10)
        plt.text(0.1, 0.19, "- Ground verification is recommended when possible", fontsize=10)
        
        pdf.savefig()
        plt.close()
    
    def _create_priority_distribution_page(self, pdf):
        """Create a page showing priority distribution charts"""
        plt.figure(figsize=(8.5, 11))
        
        # Title
        plt.suptitle("Priority Level Distribution", fontsize=16, y=0.98)
        
        # Bar chart of priority levels
        plt.subplot(2, 1, 1)
        counts = [
            self.summary["high_priority_groups"],
            self.summary["medium_priority_groups"],
            self.summary["low_priority_groups"]
        ]
        
        labels = ["High", "Medium", "Low"]
        colors = [self.PRIORITY_COLORS[label] for label in labels]
        
        bars = plt.bar(labels, counts, color=colors)
        
        plt.title("Number of Areas by Priority Level")
        plt.xlabel("Priority Level")
        plt.ylabel("Number of Areas")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom"
            )
        
        # Pie chart of priority distribution
        plt.subplot(2, 1, 2)
        
        # Only include non-zero counts
        non_zero_counts = []
        non_zero_labels = []
        non_zero_colors = []
        
        for i, (count, label, color) in enumerate(zip(counts, labels, colors)):
            if count > 0:
                non_zero_counts.append(count)
                non_zero_labels.append(label)
                non_zero_colors.append(color)
        
        if non_zero_counts:
            plt.pie(
                non_zero_counts, 
                labels=non_zero_labels, 
                colors=non_zero_colors,
                autopct='%1.1f%%',
                startangle=90
            )
            plt.axis('equal')
            plt.title("Percentage of Areas by Priority Level")
        else:
            plt.text(0.5, 0.5, "No priority data available", ha='center', va='center')
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
    
    def _create_damage_distribution_page(self, pdf):
        """Create a page showing damage distribution charts"""
        plt.figure(figsize=(8.5, 11))
        
        # Title
        plt.suptitle("Damage Distribution Analysis", fontsize=16, y=0.98)
        
        # Aggregate damage counts across all groups
        damage_counts = {str(i): 0 for i in range(5)}
        
        for group in self.groups:
            group_counts = group.get("damage_counts", {})
            for cls, count in group_counts.items():
                damage_counts[cls] += count
        
        # Calculate percentages
        total = sum(damage_counts.values())
        if total > 0:
            damage_percentage = {int(cls): (count / total) * 100 for cls, count in damage_counts.items()}
            
            # Bar chart of damage distribution
            plt.subplot(2, 1, 1)
            
            # Only plot non-zero values
            values = [damage_percentage.get(i, 0) for i in range(5)]
            non_zero_indices = [i for i, v in enumerate(values) if v > 0]
            
            bars = plt.bar(
                [self.DAMAGE_LABELS[i] for i in non_zero_indices],
                [values[i] for i in non_zero_indices],
                color=[self.DAMAGE_COLORS[i] for i in non_zero_indices]
            )
            
            plt.title("Overall Damage Distribution")
            plt.xlabel("Damage Class")
            plt.ylabel("Percentage (%)")
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.5,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom"
                )
            
            # Pie chart for major damage and destroyed
            plt.subplot(2, 1, 2)
            
            # Focus on significant damage (major and destroyed)
            significant_damage = [
                damage_percentage.get(3, 0),  # Major damage
                damage_percentage.get(4, 0)   # Destroyed
            ]
            
            other_damage = [
                damage_percentage.get(0, 0),  # Background
                damage_percentage.get(1, 0),  # No damage
                damage_percentage.get(2, 0)   # Minor damage
            ]
            
            # Only include if we have significant damage
            if sum(significant_damage) > 0:
                damage_types = ["Major Damage", "Destroyed"]
                
                plt.pie(
                    significant_damage,
                    labels=damage_types,
                    colors=[self.DAMAGE_COLORS[3], self.DAMAGE_COLORS[4]],
                    autopct='%1.1f%%',
                    startangle=90
                )
                plt.axis('equal')
                plt.title("Distribution of Significant Damage")
            else:
                plt.text(0.5, 0.5, "No significant damage detected", ha='center', va='center')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, "No damage data available", ha='center', va='center')
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
    
    def _create_priority_areas_page(self, pdf, priority_level):
        """Create a page listing areas of a specific priority level"""
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Title
        color = self.PRIORITY_COLORS.get(priority_level, 'black')
        plt.text(0.5, 0.95, f"{priority_level} Priority Areas", 
                ha='center', fontsize=20, weight='bold', color=color)
        
        # Filter groups by priority level
        groups = [g for g in self.groups if g["priority_level"] == priority_level]
        
        if groups:
            # Create a table-like layout with area information
            header_y = 0.9
            plt.text(0.1, header_y, "Area ID", fontsize=12, weight='bold')
            plt.text(0.25, header_y, "Location (Lat, Lng)", fontsize=12, weight='bold')
            plt.text(0.5, header_y, "Priority Score", fontsize=12, weight='bold')
            plt.text(0.65, header_y, "Buildings", fontsize=12, weight='bold')
            plt.text(0.8, header_y, "Major/Destroyed %", fontsize=12, weight='bold')
            
            # Horizontal line
            plt.axhline(y=header_y-0.02, xmin=0.1, xmax=0.9, color='black', alpha=0.5)
            
            # List each area
            for i, group in enumerate(groups[:20]):  # Limit to 20 areas per page
                y_pos = header_y - 0.05 * (i + 1)
                
                # Calculate percentage of major damage and destroyed
                damage_counts = group.get("damage_counts", {})
                total_damage = sum(int(count) for count in damage_counts.values())
                
                major_destroyed = 0
                if total_damage > 0:
                    major = int(damage_counts.get("3", 0))
                    destroyed = int(damage_counts.get("4", 0))
                    major_destroyed = ((major + destroyed) / total_damage) * 100
                
                # Add row data
                plt.text(0.1, y_pos, f"{group['group_id']}", fontsize=11)
                plt.text(0.25, y_pos, f"{group['center_latitude']:.6f}, {group['center_longitude']:.6f}", fontsize=11)
                plt.text(0.5, y_pos, f"{group['priority_score']:.2f}", fontsize=11)
                plt.text(0.65, y_pos, f"{group['estimated_buildings']}", fontsize=11)
                plt.text(0.8, y_pos, f"{major_destroyed:.1f}%", fontsize=11)
                
                # Light horizontal line
                if i < len(groups[:20]) - 1:
                    plt.axhline(y=y_pos-0.02, xmin=0.1, xmax=0.9, color='black', alpha=0.2)
            
            # Note if there are more areas
            if len(groups) > 20:
                plt.text(0.5, 0.15, f"+ {len(groups) - 20} more areas not shown", 
                        ha='center', fontsize=10, style='italic')
            
            # Action recommendations
            if priority_level == "High":
                plt.text(0.1, 0.08, "RECOMMENDED ACTION:", fontsize=12, weight='bold')
                plt.text(0.1, 0.05, "Immediately deploy emergency response teams to these locations.", fontsize=11)
                plt.text(0.1, 0.02, "Focus on areas with highest percentage of major damage and destroyed buildings.", fontsize=11)
            elif priority_level == "Medium":
                plt.text(0.1, 0.08, "RECOMMENDED ACTION:", fontsize=12, weight='bold')
                plt.text(0.1, 0.05, "Prepare teams for deployment once high-priority areas are addressed.", fontsize=11)
                plt.text(0.1, 0.02, "Monitor these areas for changing conditions.", fontsize=11)
        else:
            plt.text(0.5, 0.5, f"No {priority_level.lower()} priority areas identified.", 
                    ha='center', fontsize=14)
        
        pdf.savefig()
        plt.close()
    
    def _create_group_detail_page(self, pdf, group):
        """Create a detailed page for a specific high priority group"""
        plt.figure(figsize=(8.5, 11))
        
        # Title
        plt.suptitle(f"High Priority Area #{group['group_id']} Details", fontsize=16, y=0.98)
        
        # Location and general info
        plt.subplot(3, 1, 1)
        plt.axis('off')
        
        plt.text(0.05, 0.9, "Location Information:", fontsize=12, weight='bold')
        plt.text(0.05, 0.8, f"Coordinates: {group['center_latitude']:.6f}, {group['center_longitude']:.6f}", fontsize=11)
        plt.text(0.05, 0.7, f"Priority Score: {group['priority_score']:.2f}", fontsize=11)
        plt.text(0.05, 0.6, f"Estimated Buildings: {group['estimated_buildings']}", fontsize=11)
        plt.text(0.05, 0.5, f"Images in Group: {group['num_images']}", fontsize=11)
        
        plt.text(0.5, 0.9, "Emergency Response:", fontsize=12, weight='bold')
        plt.text(0.5, 0.8, "IMMEDIATE ACTION REQUIRED", fontsize=11, color='red')
        plt.text(0.5, 0.7, "Deploy search and rescue teams", fontsize=11)
        plt.text(0.5, 0.6, "Prepare medical evacuation", fontsize=11)
        plt.text(0.5, 0.5, "Establish emergency shelter nearby", fontsize=11)
        
        # Damage distribution chart
        plt.subplot(3, 1, 2)
        
        # Calculate damage distribution
        damage_counts = group.get("damage_counts", {})
        damage_values = [int(damage_counts.get(str(i), 0)) for i in range(5)]
        
        total_damage = sum(damage_values)
        if total_damage > 0:
            damage_percentage = [(count / total_damage) * 100 for count in damage_values]
            
            # Only show non-zero damage types
            non_zero_indices = [i for i, v in enumerate(damage_percentage) if v > 0]
            
            bars = plt.bar(
                [self.DAMAGE_LABELS[i] for i in non_zero_indices],
                [damage_percentage[i] for i in non_zero_indices],
                color=[self.DAMAGE_COLORS[i] for i in non_zero_indices]
            )
            
            plt.title("Damage Distribution in This Area")
            plt.xlabel("Damage Class")
            plt.ylabel("Percentage (%)")
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.5,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom"
                )
        else:
            plt.text(0.5, 0.5, "No damage data available", ha='center', va='center')
            plt.axis('off')
        
        # Risk assessment
        plt.subplot(3, 1, 3)
        plt.axis('off')
        
        plt.text(0.05, 0.9, "Risk Assessment:", fontsize=12, weight='bold')
        
        # Calculate risk factors
        risk_factors = []
        risk_level = "Unknown"
        
        if total_damage > 0:
            # Calculate percentage of major damage and destroyed
            major = damage_values[3]
            destroyed = damage_values[4]
            major_destroyed_pct = ((major + destroyed) / total_damage) * 100
            
            if major_destroyed_pct > 50:
                risk_level = "Severe"
                risk_factors.append("Majority of buildings severely damaged or destroyed")
                risk_factors.append("High probability of casualties")
                risk_factors.append("Critical infrastructure likely compromised")
            elif major_destroyed_pct > 25:
                risk_level = "High"
                risk_factors.append("Significant number of buildings severely damaged")
                risk_factors.append("Possible casualties")
                risk_factors.append("Infrastructure damage likely")
            else:
                risk_level = "Moderate"
                risk_factors.append("Some buildings damaged")
                risk_factors.append("Casualties less likely")
                risk_factors.append("Infrastructure may be partially affected")
            
            # Building density factor
            if group['estimated_buildings'] > 20:
                risk_factors.append("High building density increases risk")
            
            # Color based on risk level
            risk_color = 'black'
            if risk_level == "Severe":
                risk_color = 'darkred'
            elif risk_level == "High":
                risk_color = 'red'
            elif risk_level == "Moderate":
                risk_color = 'orange'
            
            plt.text(0.05, 0.8, f"Risk Level: {risk_level}", fontsize=11, color=risk_color, weight='bold')
            
            # List risk factors
            for i, factor in enumerate(risk_factors):
                plt.text(0.05, 0.7 - i * 0.1, f"• {factor}", fontsize=11)
        else:
            plt.text(0.05, 0.8, "Risk Level: Unknown (insufficient data)", fontsize=11)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()


def main():
    """Main function to run the report generator"""
    parser = argparse.ArgumentParser(description="Report Generator for Emergency Prioritization")
    parser.add_argument('results_file', type=str, help='Path to prioritization results JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for reports')
    parser.add_argument('--text-only', action='store_true',
                       help='Generate only text report (no PDF)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.results_file):
        print(f"Error: Results file {args.results_file} does not exist")
        return
    
    # Initialize report generator
    report_generator = EmergencyReport(
        results_file=args.results_file,
        output_dir=args.output_dir
    )
    
    # Generate reports
    report_generator.generate_text_report()
    
    if not args.text_only:
        report_generator.generate_pdf_report()


if __name__ == "__main__":
    main() 