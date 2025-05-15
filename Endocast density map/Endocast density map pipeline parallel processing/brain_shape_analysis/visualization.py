import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional, Dict, Any, Union
import logging

logger = logging.getLogger("BrainViz")

class BrainVisualization:
    """Enhanced visualization tools for brain shape analysis."""
    
    def __init__(self, output_dir: str = "viz_output"):
        """
        Initialize visualization module.
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define custom colormaps
        self.sulci_cmap = LinearSegmentedColormap.from_list(
            'sulci', [(0, 0, 0, 0), (0, 0, 1, 0.7), (0, 0.5, 1, 0.8), (0, 1, 1, 0.9), (1, 1, 1, 1)], N=256
        )
        
        self.variability_cmap = LinearSegmentedColormap.from_list(
            'variability', [(0, 0, 0, 0), (0, 0.5, 0, 0.5), (1, 1, 0, 0.7), (1, 0, 0, 0.9)], N=256
        )
    
    def _normalize_for_display(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image for display.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if image.max() == image.min():
            return np.zeros_like(image)
        
        normalized = (image - image.min()) / (image.max() - image.min())
        return normalized
    
    def visualize_registration_results(self, fixed_image: np.ndarray, 
                                     moving_image: np.ndarray,
                                     registered_image: np.ndarray,
                                     slice_idx: Optional[Dict[str, int]] = None) -> None:
        """
        Visualize registration results with before/after comparison.
        
        Args:
            fixed_image: Fixed (target) image
            moving_image: Moving (source) image
            registered_image: Registered (deformed) moving image
            slice_idx: Dictionary with slice indices for x, y, z axes
        """
        if slice_idx is None:
            slice_idx = {
                'x': fixed_image.shape[0] // 2,
                'y': fixed_image.shape[1] // 2,
                'z': fixed_image.shape[2] // 2
            }
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Normalize images for visualization
        fixed_norm = self._normalize_for_display(fixed_image)
        moving_norm = self._normalize_for_display(moving_image)
        registered_norm = self._normalize_for_display(registered_image)
        
        # Define slice planes
        slices = [
            (0, slice_idx['x'], slice(None), slice(None)),  # x-slice
            (1, slice(None), slice_idx['y'], slice(None)),  # y-slice
            (2, slice(None), slice(None), slice_idx['z'])   # z-slice
        ]
        
        # Plot images
        for row, (axis, sl_x, sl_y, sl_z) in enumerate(slices):
            # Fixed image
            if axis == 0:
                axes[row, 0].imshow(fixed_norm[sl_x, sl_y, sl_z].T, cmap='gray', origin='lower')
            elif axis == 1:
                axes[row, 0].imshow(fixed_norm[sl_x, sl_y, sl_z].T, cmap='gray', origin='lower')
            else:
                axes[row, 0].imshow(fixed_norm[sl_x, sl_y, sl_z], cmap='gray', origin='lower')
            
            axes[row, 0].set_title(f"Fixed Image ({['X', 'Y', 'Z'][axis]}-slice)")
            
            # Moving image
            if axis == 0:
                axes[row, 1].imshow(moving_norm[sl_x, sl_y, sl_z].T, cmap='gray', origin='lower')
            elif axis == 1:
                axes[row, 1].imshow(moving_norm[sl_x, sl_y, sl_z].T, cmap='gray', origin='lower')
            else:
                axes[row, 1].imshow(moving_norm[sl_x, sl_y, sl_z], cmap='gray', origin='lower')
            
            axes[row, 1].set_title(f"Moving Image ({['X', 'Y', 'Z'][axis]}-slice)")
            
            # Registered image
            if axis == 0:
                axes[row, 2].imshow(registered_norm[sl_x, sl_y, sl_z].T, cmap='gray', origin='lower')
            elif axis == 1:
                axes[row, 2].imshow(registered_norm[sl_x, sl_y, sl_z].T, cmap='gray', origin='lower')
            else:
                axes[row, 2].imshow(registered_norm[sl_x, sl_y, sl_z], cmap='gray', origin='lower')
            
            axes[row, 2].set_title(f"Registered Image ({['X', 'Y', 'Z'][axis]}-slice)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'registration_results.png'), dpi=150)
        plt.close()
    
    def visualize_average_shape(self, average_image: np.ndarray, 
                              variability_x: np.ndarray,
                              variability_y: np.ndarray,
                              variability_z: np.ndarray) -> None:
        """
        Visualize average shape and variability maps.
        
        Args:
            average_image: Average shape image
            variability_x: Variability map in x direction
            variability_y: Variability map in y direction
            variability_z: Variability map in z direction
        """
        # Create combined variability map
        variability_combined = np.sqrt(variability_x**2 + variability_y**2 + variability_z**2)
        
        # Normalize for visualization
        average_norm = self._normalize_for_display(average_image)
        variability_norm = self._normalize_for_display(variability_combined)
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        
        # Define slice planes
        slices = [
            (0, average_image.shape[0] // 2, slice(None), slice(None)),  # x-slice
            (1, slice(None), average_image.shape[1] // 2, slice(None)),  # y-slice
            (2, slice(None), slice(None), average_image.shape[2] // 2)   # z-slice
        ]
        
        # Plot images
        for row, (axis, sl_x, sl_y, sl_z) in enumerate(slices):
            # Average image
            if axis == 0:
                axes[row, 0].imshow(average_norm[sl_x, sl_y, sl_z].T, cmap='gray', origin='lower')
            elif axis == 1:
                axes[row, 0].imshow(average_norm[sl_x, sl_y, sl_z].T, cmap='gray', origin='lower')
            else:
                axes[row, 0].imshow(average_norm[sl_x, sl_y, sl_z], cmap='gray', origin='lower')
            
            axes[row, 0].set_title(f"Average Shape ({['X', 'Y', 'Z'][axis]}-slice)")
            
            # Variability map
            if axis == 0:
                im = axes[row, 1].imshow(variability_norm[sl_x, sl_y, sl_z].T, 
                                         cmap=self.variability_cmap, origin='lower')
            elif axis == 1:
                im = axes[row, 1].imshow(variability_norm[sl_x, sl_y, sl_z].T, 
                                         cmap=self.variability_cmap, origin='lower')
            else:
                im = axes[row, 1].imshow(variability_norm[sl_x, sl_y, sl_z], 
                                         cmap=self.variability_cmap, origin='lower')
            
            axes[row, 1].set_title(f"Variability Map ({['X', 'Y', 'Z'][axis]}-slice)")
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Probability')
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for colorbar
        plt.savefig(os.path.join(self.output_dir, 'average_shape_variability.png'), dpi=150)
        plt.close()
    
    def visualize_sulci_probability_maps(self, 
                                     average_image: np.ndarray,
                                     sulci_maps: Dict[int, np.ndarray]) -> None:
        """
        Visualize sulci probability maps with overlays and side colourbar.
        
        Args:
            average_image: Average shape image
            sulci_maps: Dictionary mapping sulcus labels to probability maps
        """
        # Normalize average image for visualization
        average_norm = self._normalize_for_display(average_image)
        
        for label, prob_map in sulci_maps.items():
            # Normalize probability map
            prob_map_norm = self._normalize_for_display(prob_map)
    
            # Skip if empty
            if np.max(prob_map_norm) < 1e-5:
                print(f"Skipping sulcus {label} (empty map)")
                continue
    
            # Find centroid of probability map (maximum value)
            max_idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)
    
            # Define slices through the centroid
            slices = [
                (0, max_idx[0], slice(None), slice(None)),  # X-slice
                (1, slice(None), max_idx[1], slice(None)),  # Y-slice
                (2, slice(None), slice(None), max_idx[2])   # Z-slice
            ]
    
            # Create figure
            fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
            for row, (axis, sl_x, sl_y, sl_z) in enumerate(slices):
                # Average shape
                avg_img = average_norm[sl_x, sl_y, sl_z] if axis == 2 else average_norm[sl_x, sl_y, sl_z].T
                axes[row, 0].imshow(avg_img, cmap='gray', origin='lower')
                axes[row, 0].set_title(f"Average Shape ({['X', 'Y', 'Z'][axis]}-slice)")
    
                # Probability map
                prob_img = prob_map_norm[sl_x, sl_y, sl_z] if axis == 2 else prob_map_norm[sl_x, sl_y, sl_z].T
                im = axes[row, 1].imshow(prob_img, cmap=self.sulci_cmap, origin='lower', alpha=0.7)
                axes[row, 1].set_title(f"Sulcus {label} Probability Map ({['X', 'Y', 'Z'][axis]}-slice)")
    
                # Overlay on grayscale
                axes[row, 0].imshow(prob_img, cmap=self.sulci_cmap, origin='lower', alpha=0.5)
    
            # Add side colourbar
            fig.subplots_adjust(right=0.88, wspace=0.3)
            cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax, label='Probability')
    
            # Save figure
            out_path = os.path.join(self.output_dir, f"sulcus_{label}_probability.png")
            print(f"Saving sulcus {label} probability to:", out_path)
            plt.suptitle(f"Sulcus {label} Probability Maps", fontsize=14)
            plt.savefig(out_path, dpi=150)
            plt.close()

    
    def create_3d_rendering(self, average_image: np.ndarray, prob_maps: Dict[int, np.ndarray]) -> None:
        """
        Create 3D rendering of average shape and sulci probability maps.
        
        This function requires mayavi or plotly for 3D rendering.
        
        Args:
            average_image: Average shape image
            prob_maps: Dictionary mapping sulcus labels to probability maps
        """
        try:
            import plotly.graph_objects as go
            from skimage import measure
            
            # Extract isosurfaces
            average_norm = self._normalize_for_display(average_image)
            thresh = 0.3  # Slightly lower threshold for better visualization
            
            # Extract surface from average image
            verts, faces, _, _ = measure.marching_cubes(average_norm, thresh)
            
            # Compute dynamic axis limits based on average surface
            x_min, y_min, z_min = verts.min(axis=0)
            x_max, y_max, z_max = verts.max(axis=0)
            
            # Create figure
            fig = go.Figure()
            
            # Add average surface
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.3,
                color='gray',
                name='Average Shape'
            ))
            
            # Add probability maps
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
            
            for i, (label, prob_map) in enumerate(prob_maps.items()):
                prob_norm = self._normalize_for_display(prob_map)
                
                # Extract isosurface at 70% probability
                try:
                    verts, faces, _, _ = measure.marching_cubes(prob_norm, 0.7)
                    
                    # Add to figure
                    fig.add_trace(go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        opacity=0.7,
                        color=colors[i % len(colors)],
                        name=f'Sulcus {label}'
                    ))
                except:
                    logger.warning(f"Could not extract isosurface for sulcus {label}")
            
            # Update layout - only change the aspectratio to ensure full view
            fig.update_layout(
                title="3D Rendering of Average Shape and Sulci",
                scene=dict(
                    xaxis=dict(title="X"),
                    yaxis=dict(title="Y"),
                    zaxis=dict(title="Z"),
                    # Set aspect ratio to 'data' to ensure everything is visible
                    aspectratio=dict(x=1, y=1, z=1),
                    # Slightly adjust camera position for better view
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            # Save figure
            fig.write_html(os.path.join(self.output_dir, '3d_rendering.html'))
            
        except ImportError:
            logger.warning("3D rendering requires plotly and scikit-image. Skipping.")


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
        import scipy.ndimage as scim
        
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
    def compute_pairwise_distances(list_point_sets: List[List[List[float]]]) -> Dict[int, np.ndarray]:
        """
        Compute pairwise distances between corresponding sulci points.
        
        Args:
            list_point_sets: List of point sets, each containing points [x, y, z, label]
            
        Returns:
            Dictionary mapping sulcus labels to distance matrices
        """
        # Identify all unique labels
        all_labels = set()
        for point_set in list_point_sets:
            for point in point_set:
                all_labels.add(int(point[3]))
        
        # Compute distance matrices
        distance_matrices = {}
        
        for label in all_labels:
            # Extract points for current label
            labeled_points = []
            
            for point_set in list_point_sets:
                points = [point[:3] for point in point_set if int(point[3]) == label]
                
                if points:
                    # Use centroid if multiple points
                    centroid = np.mean(points, axis=0)
                    labeled_points.append(centroid)
            
            # Compute pairwise distances
            n_points = len(labeled_points)
            if n_points > 1:
                dist_matrix = np.zeros((n_points, n_points))
                
                for i in range(n_points):
                    for j in range(i+1, n_points):
                        dist = np.linalg.norm(np.array(labeled_points[i]) - np.array(labeled_points[j]))
                        dist_matrix[i, j] = dist
                        dist_matrix[j, i] = dist
                
                distance_matrices[label] = dist_matrix
        
        return distance_matrices
    
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