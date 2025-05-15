import os
import numpy as np
import SimpleITK as sitk
import logging
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logger = logging.getLogger("SurfaceOperations")


class SurfaceOperations:
    """Operations on VTK surfaces."""
    
    @staticmethod
    def transport_vtk_surf(input_vtk_file: str, output_vtk_file: str,
                         mx: np.ndarray, my: np.ndarray, mz: np.ndarray,
                         im_center_x: float = 0, im_center_y: float = 0, im_center_z: float = 0) -> None:
        """
        Transport a VTK surface according to displacement fields.
        
        Args:
            input_vtk_file: Input VTK surface file
            output_vtk_file: Output VTK surface file
            mx, my, mz: Displacement fields
            im_center_x, im_center_y, im_center_z: Image center coordinates
        """
        try:
            # Read the input VTK file
            with open(input_vtk_file, 'r') as f_in:
                lines = f_in.readlines()
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_vtk_file)), exist_ok=True)
            
            # Write the output VTK file
            with open(output_vtk_file, 'w') as f_out:
                consider_lines = False
                points_processed = 0
                
                for i, line in enumerate(lines):
                    line_parts = line.split(' ')
                    
                    if line_parts[0] == "POINTS":
                        f_out.write(line)
                        consider_lines = True
                    elif line_parts[0] == "POLYGONS":
                        f_out.write(line)
                        consider_lines = False
                    elif consider_lines:
                        try:
                            # Parse point coordinates
                            coords = [float(val) for val in line_parts[:3]]
                            
                            # Get displacement field indices
                            l = max(0, min(mx.shape[0]-1, int(coords[0] + 0.5 + im_center_x)))
                            m = max(0, min(mx.shape[1]-1, int(coords[1] + 0.5 + im_center_y)))
                            n = max(0, min(mx.shape[2]-1, int(coords[2] + 0.5 + im_center_z)))
                            
                            # Apply displacement
                            x = coords[0] + mx[l, m, n]
                            y = coords[1] + my[l, m, n]
                            z = coords[2] + mz[l, m, n]
                            
                            f_out.write(f"{x} {y} {z}\n")
                            points_processed += 1
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error processing point at line {i+1}: {str(e)}. Using original point.")
                            f_out.write(line)
                    else:
                        f_out.write(line)
            
            logger.info(f"Transported VTK surface with {points_processed} points")
        
        except Exception as e:
            logger.error(f"Error in transport_vtk_surf: {str(e)}")
            raise
    
    @staticmethod
    def set_values_to_vtk_surf(input_vtk_file: str, ref_np_image: np.ndarray, output_vtk_file: str,
                             im_center_x: float = 0, im_center_y: float = 0, im_center_z: float = 0,
                             mult_factor: float = 1.0) -> None:
        """
        Set values from a 3D image to a VTK surface.
        
        Args:
            input_vtk_file: Input VTK surface file
            ref_np_image: Reference 3D image
            output_vtk_file: Output VTK surface file
            im_center_x, im_center_y, im_center_z: Image center coordinates
            mult_factor: Multiplication factor for values
        """
        try:
            # Read the input VTK file
            with open(input_vtk_file, 'r') as f_in:
                lines = f_in.readlines()
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_vtk_file)), exist_ok=True)
            
            # Process the file to extract points and prepare for writing
            lines_flag1 = False
            lines_flag2 = False
            list_gl = []
            header_lines = []
            
            for i, line in enumerate(lines):
                line_parts = line.split(' ')
                
                if line_parts[0] == "POINTS":
                    header_lines.append(line)
                    lines_flag1 = True
                elif line_parts[0] == "POLYGONS":
                    header_lines.append(line)
                    lines_flag1 = False
                elif line_parts[0] == "LOOKUP_TABLE":
                    header_lines.append(line)
                    lines_flag2 = True
                elif lines_flag1:
                    # Copy the line to header
                    header_lines.append(line)
                    
                    # Get point coordinates
                    try:
                        coords = [float(val) for val in line_parts[:3]]
                        x = int(coords[0] + 0.5 + im_center_x)
                        y = int(coords[1] + 0.5 + im_center_y)
                        z = int(coords[2] + 0.5 + im_center_z)
                        
                        # Get corresponding gray level from image
                        if (1 <= x < ref_np_image.shape[0]-1) and (1 <= y < ref_np_image.shape[1]-1) and (1 <= z < ref_np_image.shape[2]-1):
                            list_gl.append(ref_np_image[x, y, z] * mult_factor)
                        else:
                            list_gl.append(0.0)
                    except (ValueError, IndexError):
                        list_gl.append(0.0)
                else:
                    if not lines_flag2:
                        header_lines.append(line)
            
            # Write the output VTK file
            with open(output_vtk_file, 'w') as f_out:
                # Write header and points
                for line in header_lines:
                    f_out.write(line)
                
                # Write the scalar values
                for gl in list_gl:
                    f_out.write(f"{gl}\n")
            
            logger.info(f"Set values to VTK surface with {len(list_gl)} points")
        
        except Exception as e:
            logger.error(f"Error in set_values_to_vtk_surf: {str(e)}")
            raise