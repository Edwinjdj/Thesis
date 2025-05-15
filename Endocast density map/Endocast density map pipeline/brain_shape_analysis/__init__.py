# Import key components for easy access
from .image_utils import ImageUtils
from .registration import BrainRegistration, RegistrationParameters
from .average_shape import BrainShapeAnalysis, MeanEstimationParameters
from .sulci_analysis import SulciAnalysis
from .surface_operations import SurfaceOperations
from .point_set_operations import PointSetOperations
from .sulci_analysis_tools import SulciAnalysisTools  # Add this line
from .memory_optimization import MemoryMonitor, BatchProcessor, optimize_numpy_memory
from .visualization import BrainVisualization