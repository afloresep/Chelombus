"""Chelombus: Billion-scale molecular clustering and visualization.

This package provides tools for clustering and visualizing ultra-large
molecular datasets using Product Quantization and nested TMAPs.
"""

from .streamer.data_streamer import DataStreamer
from .encoder.encoder import PQEncoder
from .utils.fingerprints import FingerprintCalculator
from .utils.helper_functions import save_chunk, format_time

# Optional imports - only available if dependencies are installed
try:
    from .clustering.PyQKmeans import PQKMeans
except ImportError:
    PQKMeans = None  # pqkmeans not installed

__version__ = "0.1.0"
__all__ = [
    "DataStreamer",
    "PQEncoder",
    "FingerprintCalculator",
    "PQKMeans",
    "save_chunk",
    "format_time",
]
