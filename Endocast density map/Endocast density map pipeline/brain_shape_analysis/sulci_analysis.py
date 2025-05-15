import os
import numpy as np
import SimpleITK as sitk
import logging
from typing import List, Dict, Any, Optional, Tuple

# Import from other modules in the package
from .registration import BrainRegistration, RegistrationParameters, ImageRegistration
from .image_utils import ImageUtils
from .surface_operations import SurfaceOperations
from .point_set_operations import PointSetOperations

# Set up logging
logger = logging.getLogger("SulciAnalysis")


class SulciAnalysis:
    """Analysis of sulci variability across brain samples."""
    
    def __init__(self, registration_params: RegistrationParameters):
        """
        Initialize with registration parameters.
        
        Args:
            registration_params: Parameters for registration
        """
        self.reg_params = registration_params
        self.registration = BrainRegistration(registration_params)
        logger.info(f"Initialized sulci analysis with registration parameters: {self.reg_params}")
    
    def compute_sulci_variability(self, average_im: np.ndarray,
                                list_input_image_files: List[str],
                                list_input_curve_files: List[str],
                                ref_surface_file: str) -> List[List[List[float]]]:
        """
        Compute sulci variability by transporting sulci to the average shape.
        
        Args:
            average_im: Average brain image
            list_input_image_files: List of input image files
            list_input_curve_files: List of input curve files
            ref_surface_file: Reference surface file
            
        Returns:
            List of deformed point sets
        """
        try:
            # Validate inputs
            if len(list_input_image_files) != len(list_input_curve_files):
                raise ValueError("Number of input image files must match number of curve files")
            
            if not os.path.exists(ref_surface_file):
                raise FileNotFoundError(f"Reference surface file not found: {ref_surface_file}")
            
            # 1) Read data and allocate memory
            list_images = []
            for input_image_file in list_input_image_files:
                i_curr, nx, ny, nz = ImageUtils.create_np_3d(input_image_file)
                list_images.append(i_curr)
            
            list_point_sets = []
            for input_curve_file in list_input_curve_files:
                try:
                    points = np.genfromtxt(input_curve_file, delimiter=',')
                    list_point_sets.append(points.tolist())
                except Exception as e:
                    logger.error(f"Error reading curve file {input_curve_file}: {str(e)}")
                    raise
            
            # Initialize displacement fields for registration
            loc_reg_mx = np.zeros((nx, ny, nz))
            loc_reg_my = np.zeros((nx, ny, nz))
            loc_reg_mz = np.zeros((nx, ny, nz))
            
            loc_reg_fx = np.zeros((nx, ny, nz))
            loc_reg_fy = np.zeros((nx, ny, nz))
            loc_reg_fz = np.zeros((nx, ny, nz))
            
            # 2) Transport all point sets to the average shape
            list_def_point_sets = []
            
            for i in range(len(list_images)):
                curr_image = list_images[i]
                curr_points_set = list_point_sets[i]
                
                logger.info(f"Processing image and point set {i+1}/{len(list_images)}")
                
                # Reset displacement fields
                loc_reg_mx.fill(0.0)
                loc_reg_my.fill(0.0)
                loc_reg_mz.fill(0.0)
                
                # Register current image to average image
                loc_reg_mx, loc_reg_my, loc_reg_mz = self.registration.register_images(
                    curr_image, average_im, loc_reg_mx, loc_reg_my, loc_reg_mz,
                    loc_reg_fx, loc_reg_fy, loc_reg_fz
                )
                
                # Transport points using the computed displacement fields
                def_points = PointSetOperations.transport_points(
                    curr_points_set, loc_reg_mx, loc_reg_my, loc_reg_mz
                )
                list_def_point_sets.append(def_points)
                
                # Log displacement field statistics
                max_disp = np.max(np.sqrt(loc_reg_mx**2 + loc_reg_my**2 + loc_reg_mz**2))
                mean_disp = np.mean(np.sqrt(loc_reg_mx**2 + loc_reg_my**2 + loc_reg_mz**2))
                logger.info(f"Image {i+1}: Max displacement = {max_disp:.4f}, Mean displacement = {mean_disp:.4f}")
            
            # 3) Compute the average surface
            logger.info("Computing average surface")
            
            # Reset displacement fields
            loc_reg_mx.fill(0.0)
            loc_reg_my.fill(0.0)
            loc_reg_mz.fill(0.0)
            
            # Register first image to average image
            loc_reg_mx, loc_reg_my, loc_reg_mz = self.registration.register_images(
                list_images[0], average_im, loc_reg_mx, loc_reg_my, loc_reg_mz,
                loc_reg_fx, loc_reg_fy, loc_reg_fz
            )
            
            # Transport the reference surface
            SurfaceOperations.transport_vtk_surf(
                ref_surface_file, 'Result_averageSurf.vtk',
                loc_reg_mx, loc_reg_my, loc_reg_mz,
                im_center_x=150, im_center_y=150, im_center_z=150
            )
            
            # 4) Save results
            # Save all initial points in one image
            logger.info("Saving initial points")
            im_point_sets = PointSetOperations.list_point_sets_to_image(
                list_point_sets, im_center_x=150, im_center_y=150, im_center_z=150
            )
            sitk.WriteImage(im_point_sets, 'Result_InitPoints.nii')
            
            # Save all deformed points in one image
            logger.info("Saving deformed points")
            im_def_point_sets = PointSetOperations.list_point_sets_to_image(
                list_def_point_sets, im_center_x=150, im_center_y=150, im_center_z=150
            )
            sitk.WriteImage(im_def_point_sets, 'Result_DefPoints.nii')
            
            # Save deformed points in different images, each image for a given label
            logger.info("Saving deformed points by label")
            PointSetOperations.list_point_sets_to_images_and_surfaces(
                list_def_point_sets, 'Result_averageSurf.vtk', 'Result',
                im_center_x=150, im_center_y=150, im_center_z=150
            )
            
            logger.info("Sulci variability analysis complete")
            
            return list_def_point_sets
        
        except Exception as e:
            logger.error(f"Error in compute_sulci_variability: {str(e)}")
            raise