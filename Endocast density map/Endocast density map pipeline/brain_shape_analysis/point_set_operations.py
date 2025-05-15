import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as scim
import logging
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logger = logging.getLogger("PointSetOperations")


class PointSetOperations:
    """Operations on point sets representing anatomical landmarks."""
    
    @staticmethod
    def transport_points(pts_mov: List[List[float]], 
                        mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> List[List[float]]:
        """
        Transport points according to displacement fields.
        
        Args:
            pts_mov: List of points [x, y, z, label]
            mx, my, mz: Displacement fields
            
        Returns:
            Transported points
        """
        try:
            nx, ny, nz = mx.shape
            pts_def = []
            
            for i, point in enumerate(pts_mov):
                try:
                    # Get integer coordinates (ensure they're within bounds)
                    l = max(0, min(nx-1, int(point[0] + 0.5)))
                    m = max(0, min(ny-1, int(point[1] + 0.5)))
                    n = max(0, min(nz-1, int(point[2] + 0.5)))
                    
                    # Apply displacement
                    pts_def.append([
                        point[0] + mx[l, m, n],
                        point[1] + my[l, m, n],
                        point[2] + mz[l, m, n],
                        point[3]  # Keep the label unchanged
                    ])
                except IndexError:
                    logger.warning(f"Point {i} with coordinates {point[:3]} is out of bounds. Skipping.")
                    pts_def.append(point.copy())  # Keep original point
            
            return pts_def
        except Exception as e:
            logger.error(f"Error in transport_points: {str(e)}")
            raise
    
    @staticmethod
    def list_point_sets_to_image(list_point_sets: List[List[List[float]]], 
                               image_size: int = 300,
                               im_center_x: float = 0, 
                               im_center_y: float = 0, 
                               im_center_z: float = 0) -> sitk.Image:
        """
        Convert a list of point sets to a 3D image.
        
        Args:
            list_point_sets: List of point sets, each containing points [x, y, z, label]
            image_size: Size of the output image
            im_center_x, im_center_y, im_center_z: Image center coordinates
            
        Returns:
            SimpleITK image
        """
        try:
            # Create empty 3D array
            ref_np_image = np.zeros([image_size, image_size, image_size], dtype=np.int16)
            
            # Set pixel values based on point labels
            points_count = 0
            out_of_bounds_count = 0
            
            for point_set_idx, point_set in enumerate(list_point_sets):
                for point in point_set:
                    # Get coordinates relative to image center
                    x = int(point[0])
                    y = int(point[1])
                    z = int(point[2])
                    label = int(point[3])
                    
                    # Check if point is within image bounds
                    if (1 <= x < image_size-1) and (1 <= y < image_size-1) and (1 <= z < image_size-1):
                        ref_np_image[x, y, z] = label
                        points_count += 1
                    else:
                        out_of_bounds_count += 1
            
            if out_of_bounds_count > 0:
                logger.warning(f"{out_of_bounds_count} points were out of the ROI and not included in the image")
            
            logger.info(f"Created image from {points_count} points")
            
            # Convert to SimpleITK image
            ref_sitk_image = sitk.GetImageFromArray(ref_np_image)
            ref_sitk_image.SetSpacing([1, 1, 1])
            ref_sitk_image.SetOrigin([-im_center_x, -im_center_y, -im_center_z])
            
            return ref_sitk_image
        except Exception as e:
            logger.error(f"Error in list_point_sets_to_image: {str(e)}")
            raise
    
    @staticmethod
    def list_point_sets_to_images_and_surfaces(list_point_sets: List[List[List[float]]],
                                             average_surf_file: str,
                                             prefix_outputs: str,
                                             im_center_x: float = 0, 
                                             im_center_y: float = 0, 
                                             im_center_z: float = 0) -> None:
        """
        Convert a list of point sets to multiple images and surfaces, one per label.
        
        Args:
            list_point_sets: List of point sets, each containing points [x, y, z, label]
            average_surf_file: File path to the average surface
            prefix_outputs: Prefix for output file names
            im_center_x, im_center_y, im_center_z: Image center coordinates
        """
        try:
            # 1) Get all unique labels
            all_labels = set()
            for point_set in list_point_sets:
                for point in point_set:
                    all_labels.add(int(point[3]))
            
            logger.info(f"Found {len(all_labels)} unique labels")
            
            # 2) Process each label
            image_size = 300
            for curr_label in sorted(all_labels):
                logger.info(f"Processing label: {curr_label}")
                
                # 2.1) Compute the values
                ref_np_image = np.zeros([image_size, image_size, image_size], dtype=np.float32)
                
                points_count = 0
                for point_set in list_point_sets:
                    for point in point_set:
                        if (int(point[3]) == curr_label and 
                            1 <= point[0] < image_size-1 and 
                            1 <= point[1] < image_size-1 and 
                            1 <= point[2] < image_size-1):
                            
                            ref_np_image[int(point[0]), int(point[1]), int(point[2])] = 1000.0
                            points_count += 1
                
                logger.info(f"Added {points_count} points for label {curr_label}")
                
                # Apply Gaussian smoothing
                ref_np_image = scim.gaussian_filter(ref_np_image, 5, mode='constant')
                
                # Normalize to 0-100 range
                if ref_np_image.max() > 0:
                    ref_np_image *= 100.0 / ref_np_image.max()
                
                # 2.2) Save as NIFTI image
                ref_sitk_image = sitk.GetImageFromArray(ref_np_image)
                ref_sitk_image.SetSpacing([1, 1, 1])
                ref_sitk_image.SetOrigin([-im_center_x, -im_center_y, -im_center_z])
                
                output_file = f"{prefix_outputs}_DefPoints_Label_{curr_label}.nii"
                sitk.WriteImage(ref_sitk_image, output_file)
                logger.info(f"Saved image to {output_file}")
                
                # 2.3) Save as VTK surface
                output_vtk = f"{prefix_outputs}_DefPoints_Label_{curr_label}.vtk"
                
                # Import at function level to avoid circular imports
                from .surface_operations import SurfaceOperations
                
                SurfaceOperations.set_values_to_vtk_surf(
                    average_surf_file, ref_np_image, output_vtk,
                    im_center_x=im_center_x, im_center_y=im_center_y, im_center_z=im_center_z
                )
                logger.info(f"Saved surface to {output_vtk}")
        
        except Exception as e:
            logger.error(f"Error in list_point_sets_to_images_and_surfaces: {str(e)}")
            raise