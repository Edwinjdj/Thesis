# brain_shape_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy.ndimage as scim
import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("brain_shape_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BrainShapeAnalysis")


@dataclass
class RegistrationParameters:
    """Parameters for registration algorithm."""
    v1: float = 10.0  # Variance of Gaussian filter on forces
    v2: float = 1.0   # Variance of Gaussian filter on displacement
    delta_updates: float = 0.5  # Update step size
    it_nb: int = 20   # Number of iterations


@dataclass
class MeanEstimationParameters:
    """Parameters for mean shape estimation."""
    ref_im_nb: int  # Number of reference images
    it_nb: int = 20  # Number of iterations


class ImageUtils:
    """Utility functions for image handling."""
    
    @staticmethod
    def read_sitk_image(im_file: str) -> sitk.Image:
        """Read a SimpleITK image from file."""
        if not os.path.exists(im_file):
            raise FileNotFoundError(f"Image file not found: {im_file}")
        return sitk.Cast(sitk.ReadImage(im_file), sitk.sitkFloat32)
    
    @staticmethod
    def write_sitk_image(sitk_image: sitk.Image, im_file: str) -> None:
        """Write a SimpleITK image to file."""
        os.makedirs(os.path.dirname(os.path.abspath(im_file)), exist_ok=True)
        sitk.WriteImage(sitk_image, im_file)
    
    @staticmethod
    def create_np_3d(file_path: str) -> Tuple[np.ndarray, int, int, int]:
        """Create a 3D numpy array from a SimpleITK image file."""
        try:
            i_sitk = ImageUtils.read_sitk_image(file_path)
            i_np = sitk.GetArrayViewFromImage(i_sitk)
            nx, ny, nz = np.shape(i_np)
            return np.copy(i_np), nx, ny, nz
        except Exception as e:
            logger.error(f"Error creating 3D array from {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def save_np_array_as_image(np_array: np.ndarray, 
                              im_with_ref_domain: sitk.Image,
                              output_file: str) -> None:
        """Save a numpy array as a SimpleITK image."""
        try:
            i_itk = sitk.GetImageFromArray(np_array)
            i_itk.CopyInformation(im_with_ref_domain)
            ImageUtils.write_sitk_image(i_itk, output_file)
        except Exception as e:
            logger.error(f"Error saving array as image to {output_file}: {str(e)}")
            raise
    
    @staticmethod
    def display_slice(im_3d: np.ndarray, layer: int, axis: int) -> None:
        """Display a slice from a 3D image array."""
        if axis not in [1, 2, 3]:
            raise ValueError(f"Invalid axis: {axis}. Must be 1, 2, or 3.")
            
        plt.figure(figsize=(10, 8))
        
        if axis == 1 and 0 <= layer-1 < im_3d.shape[0]:
            plt.imshow(im_3d[layer-1, :, :])
            plt.title(f"Slice {layer} along axis {axis}")
            plt.colorbar()
            plt.show()
        elif axis == 2 and 0 <= layer-1 < im_3d.shape[1]:
            plt.imshow(im_3d[:, layer-1, :])
            plt.title(f"Slice {layer} along axis {axis}")
            plt.colorbar()
            plt.show()
        elif axis == 3 and 0 <= layer-1 < im_3d.shape[2]:
            plt.imshow(im_3d[:, :, layer-1])
            plt.title(f"Slice {layer} along axis {axis}")
            plt.colorbar()
            plt.show()
        else:
            logger.warning(f"Slice {layer} out of bounds for axis {axis}")


class ImageRegistration:
    """Image registration functionality."""
    
    @staticmethod
    def transport_image(imov: np.ndarray, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> np.ndarray:
        """
        Transport a moving image according to displacement fields.
        
        Args:
            imov: Moving image
            mx, my, mz: Displacement fields
            
        Returns:
            Deformed image
        """
        try:
            nx, ny, nz = np.shape(imov)
            idef = np.zeros((nx, ny, nz))
            j, i, k = np.meshgrid(range(ny), range(nx), range(nz), indexing='ij')
            i, j, k = i.transpose(1, 0, 2), j.transpose(1, 0, 2), k.transpose(1, 0, 2)
            
            # Calculate new indices
            l = (i + mx[i, j, k] + 0.5)
            m = (j + my[i, j, k] + 0.5)
            n = (k + mz[i, j, k] + 0.5)
            
            # Boundary conditions
            l = np.clip(l, 0, nx-1)
            m = np.clip(m, 0, ny-1)
            n = np.clip(n, 0, nz-1)
            
            l = l.astype(int)
            m = m.astype(int)
            n = n.astype(int)
            
            # Apply displacement
            idef[i, j, k] = imov[l[i, j, k], m[i, j, k], n[i, j, k]]
            
            return idef
        except Exception as e:
            logger.error(f"Error in transport_image: {str(e)}")
            raise
    
    @staticmethod
    def ssd(i: np.ndarray, j: np.ndarray) -> float:
        """Calculate Sum of Squared Differences between two images."""
        return np.sum((i - j) ** 2)
    
    @staticmethod
    def trilinear_interpolation(mx: np.ndarray, my: np.ndarray, mz: np.ndarray,
                             fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trilinear interpolation for force fields.
        
        Args:
            mx, my, mz: Displacement fields
            fx, fy, fz: Force fields
            
        Returns:
            Interpolated force fields
        """
        try:
            nx, ny, nz = np.shape(mx)
            j, i, k = np.meshgrid(range(ny), range(nx), range(nz), indexing='ij')
            i, j, k = i.transpose(1, 0, 2), j.transpose(1, 0, 2), k.transpose(1, 0, 2)
            
            # Create new indices
            l = (i + mx[i, j, k])
            m = (j + my[i, j, k])
            n = (k + mz[i, j, k])
            
            # Boundary conditions
            l = np.clip(l, 0, nx-1.01)
            m = np.clip(m, 0, ny-1.01)
            n = np.clip(n, 0, nz-1.01)
            
            fl = np.floor(l).astype(int)
            fm = np.floor(m).astype(int)
            fn = np.floor(n).astype(int)
            
            cl = fl + 1
            cm = fm + 1
            cn = fn + 1
            
            # Ensure indices are within bounds
            cl = np.minimum(cl, nx-1)
            cm = np.minimum(cm, ny-1)
            cn = np.minimum(cn, nz-1)
            
            cx = cl - l
            cy = cm - m
            cz = cn - n
            
            # Compute interpolation for Fx
            fx_x1 = fx[fl, fm, fn] * cx + fx[np.minimum(fl+1, nx-1), fm, fn] * (1 - cx)
            fx_x2 = fx[fl, np.minimum(fm+1, ny-1), fn] * cx + fx[np.minimum(fl+1, nx-1), np.minimum(fm+1, ny-1), fn] * (1 - cx)
            fx_x3 = fx[fl, fm, np.minimum(fn+1, nz-1)] * cx + fx[np.minimum(fl+1, nx-1), fm, np.minimum(fn+1, nz-1)] * (1 - cx)
            fx_x4 = fx[fl, np.minimum(fm+1, ny-1), np.minimum(fn+1, nz-1)] * cx + fx[np.minimum(fl+1, nx-1), np.minimum(fm+1, ny-1), np.minimum(fn+1, nz-1)] * (1 - cx)
            
            fx_y1 = fx_x1 * cy + fx_x2 * (1 - cy)
            fx_y2 = fx_x3 * cy + fx_x4 * (1 - cy)
            
            fx_z = fx_y1 * cz + fx_y2 * (1 - cz)
            
            # Compute interpolation for Fy
            fy_x1 = fy[fl, fm, fn] * cx + fy[np.minimum(fl+1, nx-1), fm, fn] * (1 - cx)
            fy_x2 = fy[fl, np.minimum(fm+1, ny-1), fn] * cx + fy[np.minimum(fl+1, nx-1), np.minimum(fm+1, ny-1), fn] * (1 - cx)
            fy_x3 = fy[fl, fm, np.minimum(fn+1, nz-1)] * cx + fy[np.minimum(fl+1, nx-1), fm, np.minimum(fn+1, nz-1)] * (1 - cx)
            fy_x4 = fy[fl, np.minimum(fm+1, ny-1), np.minimum(fn+1, nz-1)] * cx + fy[np.minimum(fl+1, nx-1), np.minimum(fm+1, ny-1), np.minimum(fn+1, nz-1)] * (1 - cx)
            
            fy_y1 = fy_x1 * cy + fy_x2 * (1 - cy)
            fy_y2 = fy_x3 * cy + fy_x4 * (1 - cy)
            
            fy_z = fy_y1 * cz + fy_y2 * (1 - cz)
            
            # Compute interpolation for Fz
            fz_x1 = fz[fl, fm, fn] * cx + fz[np.minimum(fl+1, nx-1), fm, fn] * (1 - cx)
            fz_x2 = fz[fl, np.minimum(fm+1, ny-1), fn] * cx + fz[np.minimum(fl+1, nx-1), np.minimum(fm+1, ny-1), fn] * (1 - cx)
            fz_x3 = fz[fl, fm, np.minimum(fn+1, nz-1)] * cx + fz[np.minimum(fl+1, nx-1), fm, np.minimum(fn+1, nz-1)] * (1 - cx)
            fz_x4 = fz[fl, np.minimum(fm+1, ny-1), np.minimum(fn+1, nz-1)] * cx + fz[np.minimum(fl+1, nx-1), np.minimum(fm+1, ny-1), np.minimum(fn+1, nz-1)] * (1 - cx)
            
            fz_y1 = fz_x1 * cy + fz_x2 * (1 - cy)
            fz_y2 = fz_x3 * cy + fz_x4 * (1 - cy)
            
            fz_z = fz_y1 * cz + fz_y2 * (1 - cz)
            
            return fx_z, fy_z, fz_z
        except Exception as e:
            logger.error(f"Error in trilinear_interpolation: {str(e)}")
            raise
    
    @staticmethod
    def det_jac(mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the minimum and maximum determinants of the Jacobian.
        
        Args:
            mx, my, mz: Displacement fields
            
        Returns:
            Minimum and maximum Jacobian determinants
        """
        try:
            nx, ny, nz = np.shape(mx)
            v_min = float('inf')
            v_max = float('-inf')
            
            # Create Jacobian matrix for each point
            for i in range(nx-1):
                for j in range(ny-1):
                    for k in range(nz-1):
                        jac_p = np.zeros((3, 3))
                        
                        jac_p[0, 0] = mx[i+1, j, k] - mx[i, j, k]
                        jac_p[1, 1] = my[i, j+1, k] - my[i, j, k]
                        jac_p[2, 2] = mz[i, j, k+1] - mz[i, j, k]
                        
                        jac_p[1, 0] = mx[i, j+1, k] - mx[i, j, k]
                        jac_p[2, 0] = mx[i, j, k+1] - mx[i, j, k]
                        
                        jac_p[0, 1] = my[i+1, j, k] - my[i, j, k]
                        jac_p[2, 1] = my[i, j, k+1] - my[i, j, k]
                        
                        jac_p[0, 2] = mz[i+1, j, k] - mz[i, j, k]
                        jac_p[1, 2] = mz[i, j+1, k] - mz[i, j, k]
                        
                        # Calculate determinant
                        deter = np.linalg.det(jac_p)
                        
                        # Update min and max values
                        v_min = min(v_min, deter)
                        v_max = max(v_max, deter)
            
            return v_min+1, v_max+1
        except Exception as e:
            logger.error(f"Error in det_jac: {str(e)}")
            raise
    
    @staticmethod
    def calc_forces(i_f: np.ndarray, i_d: np.ndarray, v1: float,
                   mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate force fields for registration.
        
        Args:
            i_f: Fixed image
            i_d: Deformed moving image
            v1: Force smoothing variance
            mx, my, mz: Displacement fields
            
        Returns:
            Force fields in x, y, z directions
        """
        try:
            # Calculate forces using gradient
            gx, gy, gz = np.gradient(i_d)
            fx = 2 * (i_f - i_d) * gx
            fy = 2 * (i_f - i_d) * gy
            fz = 2 * (i_f - i_d) * gz
            
            # Trilinear interpolation
            fx, fy, fz = ImageRegistration.trilinear_interpolation(mx, my, mz, fx, fy, fz)
            
            # Smooth forces
            fx = scim.gaussian_filter(fx, v1, mode='constant')
            fy = scim.gaussian_filter(fy, v1, mode='constant')
            fz = scim.gaussian_filter(fz, v1, mode='constant')
            
            return fx, fy, fz
        except Exception as e:
            logger.error(f"Error in calc_forces: {str(e)}")
            raise