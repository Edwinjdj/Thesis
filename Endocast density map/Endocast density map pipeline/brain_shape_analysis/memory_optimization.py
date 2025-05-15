import os
import gc
import psutil
import numpy as np
import logging
from typing import List, Dict, Any, Callable, Optional
from functools import wraps

logger = logging.getLogger("MemoryOptimization")

class MemoryMonitor:
    """Monitor and optimize memory usage during processing."""
    
    def __init__(self, enable_monitoring: bool = True, 
                memory_threshold: float = 0.85,
                clear_unused_images: bool = True):
        """
        Initialize memory monitor.
        
        Args:
            enable_monitoring: Whether to enable memory monitoring
            memory_threshold: Threshold of memory usage (0-1) to trigger optimization
            clear_unused_images: Whether to clear unused images from memory
        """
        self.enable_monitoring = enable_monitoring
        self.memory_threshold = memory_threshold
        self.clear_unused_images = clear_unused_images
        self.cached_objects = {}
        
        if enable_monitoring:
            try:
                import psutil
                self.process = psutil.Process(os.getpid())
            except ImportError:
                logger.warning("psutil not available. Memory monitoring disabled.")
                self.enable_monitoring = False
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage as a ratio of total available memory.
        
        Returns:
            Memory usage ratio (0-1)
        """
        if not self.enable_monitoring:
            return 0.0
        
        try:
            memory_info = self.process.memory_info()
            total_memory = psutil.virtual_memory().total
            return memory_info.rss / total_memory
        except Exception as e:
            logger.warning(f"Error getting memory usage: {str(e)}")
            return 0.0
    
    def check_memory(self) -> bool:
        """
        Check if memory usage is above threshold.
        
        Returns:
            True if memory optimization is needed
        """
        if not self.enable_monitoring:
            return False
        
        memory_usage = self.get_memory_usage()
        if memory_usage > self.memory_threshold:
            logger.warning(f"High memory usage detected: {memory_usage:.2%}. Optimization needed.")
            return True
        return False
    
    def optimize_memory(self) -> None:
        """Perform memory optimization."""
        if not self.enable_monitoring:
            return
        
        logger.info("Running memory optimization")
        
        # Clear unused objects
        if self.clear_unused_images:
            for key in list(self.cached_objects.keys()):
                if not self.cached_objects[key]['in_use']:
                    del self.cached_objects[key]
                    logger.info(f"Cleared cached object {key}")
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Memory usage after optimization: {self.get_memory_usage():.2%}")
    
    def monitor_function(self, func: Callable) -> Callable:
        """
        Decorator to monitor memory usage during function execution.
        
        Args:
            func: Function to monitor
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enable_monitoring:
                return func(*args, **kwargs)
            
            # Check memory before function call
            pre_memory = self.get_memory_usage()
            logger.info(f"Memory usage before {func.__name__}: {pre_memory:.2%}")
            
            # Call function
            result = func(*args, **kwargs)
            
            # Check memory after function call
            post_memory = self.get_memory_usage()
            logger.info(f"Memory usage after {func.__name__}: {post_memory:.2%}")
            
            # Optimize memory if needed
            if post_memory > self.memory_threshold:
                logger.warning(f"High memory usage after {func.__name__}: {post_memory:.2%}")
                self.optimize_memory()
            
            return result
        
        return wrapper
    
    def cache_object(self, key: str, obj: Any) -> None:
        """
        Cache an object for potential reuse.
        
        Args:
            key: Unique key for the object
            obj: Object to cache
        """
        if not self.enable_monitoring:
            return
        
        self.cached_objects[key] = {
            'object': obj,
            'in_use': True
        }
    
    def get_cached_object(self, key: str) -> Optional[Any]:
        """
        Get a cached object.
        
        Args:
            key: Unique key for the object
            
        Returns:
            Cached object or None if not found
        """
        if not self.enable_monitoring or key not in self.cached_objects:
            return None
        
        self.cached_objects[key]['in_use'] = True
        return self.cached_objects[key]['object']
    
    def release_object(self, key: str) -> None:
        """
        Release a cached object.
        
        Args:
            key: Unique key for the object
        """
        if not self.enable_monitoring or key not in self.cached_objects:
            return
        
        self.cached_objects[key]['in_use'] = False


class BatchProcessor:
    """Process data in batches to reduce memory usage."""
    
    def __init__(self, batch_size: int = 5, memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Maximum number of items to process in a batch
            memory_monitor: Memory monitor instance
        """
        self.batch_size = batch_size
        self.memory_monitor = memory_monitor
    
    def process_in_batches(self, items: List[Any], 
                         process_func: Callable[[List[Any]], List[Any]]) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            process_func: Function to process a batch of items
            
        Returns:
            List of processed items
        """
        results = []
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i+self.batch_size]
            logger.info(f"Processing batch {(i//self.batch_size)+1}/{(len(items)-1)//self.batch_size+1} "
                       f"(items {i+1}-{min(i+self.batch_size, len(items))})")
            
            # Process batch
            batch_results = process_func(batch)
            results.extend(batch_results)
            
            # Check memory
            if self.memory_monitor and self.memory_monitor.check_memory():
                self.memory_monitor.optimize_memory()
        
        return results


def optimize_numpy_memory(array_list: List[np.ndarray]) -> None:
    """
    Optimize memory usage of numpy arrays.
    
    Args:
        array_list: List of numpy arrays to optimize
    """
    for i, arr in enumerate(array_list):
        # Check if array needs to be downcasted
        if arr.dtype == np.float64:
            array_list[i] = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            # Check if values fit in int32
            if np.min(arr) >= np.iinfo(np.int32).min and np.max(arr) <= np.iinfo(np.int32).max:
                array_list[i] = arr.astype(np.int32)
        
        # Force contiguous memory layout
        if not arr.flags.c_contiguous:
            array_list[i] = np.ascontiguousarray(arr)


def memory_efficient_registration(brain_registration, fixed_images, moving_images, 
                                memory_monitor=None, batch_size=5):
    """
    Memory-efficient implementation of multi-image registration.
    
    Args:
        brain_registration: BrainRegistration instance
        fixed_images: List of fixed images
        moving_images: List of moving images
        memory_monitor: MemoryMonitor instance
        batch_size: Maximum batch size
        
    Returns:
        List of displacement fields
    """
    results = []
    batch_processor = BatchProcessor(batch_size, memory_monitor)
    
    def process_batch(batch_indices):
        batch_results = []
        
        for i in batch_indices:
            # Allocate displacement fields
            nx, ny, nz = fixed_images[i].shape
            mx = np.zeros((nx, ny, nz), dtype=np.float32)
            my = np.zeros((nx, ny, nz), dtype=np.float32)
            mz = np.zeros((nx, ny, nz), dtype=np.float32)
            
            # Allocate force fields
            fx = np.zeros((nx, ny, nz), dtype=np.float32)
            fy = np.zeros((nx, ny, nz), dtype=np.float32)
            fz = np.zeros((nx, ny, nz), dtype=np.float32)
            
            # Register images
            mx, my, mz = brain_registration.register_images(
                fixed_images[i], moving_images[i], mx, my, mz, fx, fy, fz
            )
            
            batch_results.append((mx, my, mz))
            
            # Clear force fields to save memory
            del fx, fy, fz
            
            # Optimize memory
            if memory_monitor and memory_monitor.check_memory():
                memory_monitor.optimize_memory()
        
        return batch_results
    
    # Process all indices in batches
    indices = list(range(len(fixed_images)))
    results = batch_processor.process_in_batches(indices, process_batch)
    
    return results