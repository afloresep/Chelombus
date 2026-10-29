"""Chelombus: Billion-scale molecular clustering and visualization.

This package provides tools for clustering and visualizing ultra-large
molecular datasets using Product Quantization and nested TMAPs.
"""

from .streamer.data_streamer import DataStreamer
from .utils.fingerprints import FingerprintCalculator
from .utils.helper_functions import save_chunk, format_time


def __getattr__(name):
    # PQEncoder and PQKMeans pull in torch, numba and pqkmeans, so they load on
    # first use. This keeps `import chelombus` fast, which matters most for
    # fingerprint worker processes on Windows and macOS (each re-imports it).
    if name == "PQEncoder":
        from .encoder.encoder import PQEncoder as value
    elif name == "PQKMeans":
        try:
            from .clustering.PyQKmeans import PQKMeans as value
        except ImportError as e:
            import warnings
            warnings.warn(f"PQKMeans not available: {e}")
            value = None  # pqkmeans or numba not installed
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value

# Cluster I/O utilities (require duckdb)
try:
    from .utils.cluster_io import (
        query_cluster,
        export_cluster,
        export_all_clusters,
        get_cluster_stats,
        get_cluster_ids,
        get_total_molecules,
        query_clusters_batch,
        sample_from_cluster,
    )
    _CLUSTER_IO_AVAILABLE = True
except ImportError:
    _CLUSTER_IO_AVAILABLE = False
    # Define stubs that raise helpful errors
    def _cluster_io_not_available(*args, **kwargs):
        raise ImportError(
            "Cluster I/O functions require duckdb. "
            "Install with: pip install duckdb"
        )
    query_cluster = _cluster_io_not_available
    export_cluster = _cluster_io_not_available
    export_all_clusters = _cluster_io_not_available
    get_cluster_stats = _cluster_io_not_available
    get_cluster_ids = _cluster_io_not_available
    get_total_molecules = _cluster_io_not_available
    query_clusters_batch = _cluster_io_not_available
    sample_from_cluster = _cluster_io_not_available

__version__ = "0.2.2"
__all__ = [
    # Core classes
    "DataStreamer",
    "PQEncoder",
    "FingerprintCalculator",
    "PQKMeans",
    # Helper functions
    "save_chunk",
    "format_time",
    # Cluster I/O (require duckdb)
    "query_cluster",
    "export_cluster",
    "export_all_clusters",
    "get_cluster_stats",
    "get_cluster_ids",
    "get_total_molecules",
    "query_clusters_batch",
    "sample_from_cluster",
]
