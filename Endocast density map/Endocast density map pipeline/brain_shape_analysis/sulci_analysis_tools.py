import os
import numpy as np
import scipy.ndimage as scim
import logging
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logger = logging.getLogger("SulciAnalysisTools")

class SulciAnalysisTools:
    """Advanced analysis tools for sulci variability."""
    
    @staticmethod
    def compute_sulci_statistics(list_point_sets: List[List[List[float]]]) -> Dict[int, Dict[str, Any]]:
        """
        Compute statistics for each sulcus.
        
        Args:
            list_point_sets: List of point sets, each containing points [x, y, z, label]
            
        Returns:
            Dictionary mapping sulcus labels to statistics
        """
        # Identify all unique labels
        all_labels = set()
        for point_set in list_point_sets:
            for point in point_set:
                all_labels.add(int(point[3]))
        
        # Initialize statistics
        stats = {}
        
        for label in all_labels:
            stats[label] = {
                'occurrence_rate': 0.0,
                'mean_position': np.zeros(3),
                'std_position': np.zeros(3),
                'point_count': 0
            }
        
        # Compute statistics
        for label in all_labels:
            # Count sets containing this label
            sets_with_label = 0
            all_points = []
            
            for point_set in list_point_sets:
                points_with_label = [point[:3] for point in point_set if int(point[3]) == label]
                
                if points_with_label:
                    sets_with_label += 1
                    all_points.extend(points_with_label)
            
            # Compute occurrence rate
            stats[label]['occurrence_rate'] = sets_with_label / len(list_point_sets)
            
            # Compute position statistics
            if all_points:
                points_array = np.array(all_points)
                stats[label]['mean_position'] = np.mean(points_array, axis=0)
                stats[label]['std_position'] = np.std(points_array, axis=0)
                stats[label]['point_count'] = len(all_points)
        
        return stats
    
    @staticmethod
    def create_labeled_volume(list_point_sets: List[List[List[float]]],
                            image_size: int = 300,
                            smooth_sigma: float = 5.0) -> Dict[int, np.ndarray]:
        """
        Create labeled probability volumes for each sulcus.
        
        Args:
            list_point_sets: List of point sets, each containing points [x, y, z, label]
            image_size: Size of the output volume
            smooth_sigma: Sigma for Gaussian smoothing
            
        Returns:
            Dictionary mapping sulcus labels to probability volumes
        """
        # Identify all unique labels
        all_labels = set()
        for point_set in list_point_sets:
            for point in point_set:
                all_labels.add(int(point[3]))
        
        # Create probability volumes
        prob_volumes = {}
        
        for label in all_labels:
            # Initialize volume
            volume = np.zeros((image_size, image_size, image_size), dtype=np.float32)
            
            # Add points
            for point_set in list_point_sets:
                for point in point_set:
                    if int(point[3]) == label:
                        x, y, z = int(point[0]), int(point[1]), int(point[2])
                        
                        if 0 <= x < image_size and 0 <= y < image_size and 0 <= z < image_size:
                            volume[x, y, z] = 1000.0
            
            # Apply Gaussian smoothing
            volume = scim.gaussian_filter(volume, smooth_sigma, mode='constant')
            
            # Normalize to [0, 100]
            if volume.max() > 0:
                volume = 100.0 * volume / volume.max()
            
            prob_volumes[label] = volume
        
        return prob_volumes
    
    @staticmethod
    def generate_report(stats: Dict[int, Dict[str, Any]], 
                      output_file: str = "sulci_report.txt") -> None:
        """
        Generate a text report with sulci statistics.
        
        Args:
            stats: Dictionary mapping sulcus labels to statistics
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            f.write("Sulci Variability Analysis Report\n")
            f.write("================================\n\n")
            
            for label, label_stats in sorted(stats.items()):
                f.write(f"Sulcus {label}:\n")
                f.write(f"  Occurrence Rate: {label_stats['occurrence_rate']*100:.1f}%\n")
                f.write(f"  Points Count: {label_stats['point_count']}\n")
                f.write("  Mean Position (X, Y, Z): "
                       f"({label_stats['mean_position'][0]:.1f}, "
                       f"{label_stats['mean_position'][1]:.1f}, "
                       f"{label_stats['mean_position'][2]:.1f})\n")
                f.write("  Std. Dev. Position (X, Y, Z): "
                       f"({label_stats['std_position'][0]:.1f}, "
                       f"{label_stats['std_position'][1]:.1f}, "
                       f"{label_stats['std_position'][2]:.1f})\n")
                f.write("\n")