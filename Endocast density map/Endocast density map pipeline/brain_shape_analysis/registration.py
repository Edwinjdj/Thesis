import numpy as np
import scipy.ndimage as scim
import logging
from dataclasses import dataclass
from typing import List, Tuple

# Set up logging
logger = logging.getLogger("Registration")

@dataclass
class RegistrationParameters:
    """Parameters for registration algorithm."""
    v1: float = 10.0  # Variance of Gaussian filter on forces
    v2: float = 1.0   # Variance of Gaussian filter on displacement
    delta_updates: float = 0.5  # Update step size
    it_nb: int = 20   # Number of iterations
    
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
            
class BrainRegistration:
    """Brain image registration implementation."""
    
    def __init__(self, registration_params: RegistrationParameters):
        """
        Initialize with registration parameters.
        
        Args:
            registration_params: Parameters for the registration algorithm
        """
        self.params = registration_params
        logger.info(f"Initialized registration with parameters: {self.params}")
    
    def register_images(self, fixed_image: np.ndarray, moving_image: np.ndarray,
                       mx: np.ndarray, my: np.ndarray, mz: np.ndarray,
                       fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> List[np.ndarray]:
        """
        Register a moving image to a fixed image.
        
        Args:
            fixed_image: Target image
            moving_image: Image to be registered
            mx, my, mz: Displacement fields (initialized with zeros)
            fx, fy, fz: Force fields (temporary variables)
            
        Returns:
            Updated displacement fields
        """
        try:
            # Initialize displacement fields (already done by caller)
            # Gradient descent
            for k in range(self.params.it_nb):
                # Transport moving image
                i_def = ImageRegistration.transport_image(moving_image, mx, my, mz)
                
                # Calculate forces
                fx, fy, fz = ImageRegistration.calc_forces(fixed_image, i_def, self.params.v1, mx, my, mz)
                
                # Determine optimal step size
                if k == 0:
                    curr_max_def = np.max(np.sqrt(fx**2 + fy**2 + fz**2))
                    if curr_max_def > 1e-5:
                        p = 0.5 / curr_max_def
                    else:
                        p = 1.0
                
                # Update displacement fields
                mx = mx + p * fx
                my = my + p * fy
                mz = mz + p * fz
                
                # Smooth displacement fields
                mx = scim.gaussian_filter(mx, self.params.v2, mode='constant')
                my = scim.gaussian_filter(my, self.params.v2, mode='constant')
                mz = scim.gaussian_filter(mz, self.params.v2, mode='constant')
                
                # Log progress every few iterations
                if (k + 1) % 5 == 0 or k == 0:
                    max_disp = np.max(np.sqrt(mx**2 + my**2 + mz**2))
                    logger.info(f"Registration iteration {k+1}/{self.params.it_nb}: max displacement = {max_disp:.4f}")
            
            return [mx, my, mz]
        
        except Exception as e:
            logger.error(f"Error in register_images: {str(e)}")
            raise