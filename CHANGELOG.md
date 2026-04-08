# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-04-08

### Added
- Full GPU acceleration for the PQ pipeline (fit, transform, predict) via PyTorch + Triton kernels
- Triton JIT-compiled `_pq_assign_kernel` for cluster assignment — never materializes the N×K distance matrix
- GPU-accelerated `PQEncoder.fit()` with `device='gpu'|'auto'` parameter
- GPU-accelerated `PQEncoder.transform()` with automatic VRAM-aware batching
- GPU-accelerated `PQKMeans.fit()` and `.predict()` with automatic CPU fallback
- Early-stopping tolerance (`tol`) parameter for `PQKMeans` GPU training path
- New `_update_centers()` function with chunked histogram accumulation for bounded memory
- New scripts: `benchmark_1B_pipeline.py`, `benchmark_gpu_predict.py`, `cluster_smiles.py`, `k_selection_gpu.py`
- Comprehensive GPU test suite in `test_clustering.py`

### Changed
- `PQEncoder.fit()` now accepts `device` parameter (`'cpu'`, `'gpu'`, `'auto'`)
- `PQEncoder.transform()` automatically uses GPU when available
- `PQKMeans` uses GPU path when CUDA/Triton are detected, with transparent CPU fallback
- MQN fingerprint dtype changed to `int16` for correctness
- Migrated from `tmap-silicon` + `faerun` to `tmap2` for visualization
- Rewrote `visualization.py` to use tmap2's `TMAP`, `TmapViz`, and chemistry utilities
- Removed `pandarallel` dependency (parallelism now handled by tmap2)
- Default Morgan fingerprint bits changed from 1024 to 2048 (tmap2 default)

### Fixed
- Stale codebook cache bug in encoder
- PyTorch monkeypatch guard for CI compatibility

## [0.2.1] - 2026-03-06

### Added
- Numba JIT-compiled parallel `predict` for PQKMeans (~3x speedup)
- `numba>=0.57.0` as a core dependency

## [0.2.0] - 2025-XX-XX

### Added
- DuckDB-based cluster I/O utilities (`query_cluster`, `export_cluster`, `export_all_clusters`)
- Batch cluster querying with `query_clusters_batch`
- Random sampling from clusters with `sample_from_cluster`
- TMAP visualization CLI (`chelombus-tmap`)
- Representative TMAP generation with `representative_tmap`
- Configurable molecular properties for TMAP coloring
- Project URLs in package metadata

### Changed
- Improved memory efficiency for large-scale processing
- Enhanced error messages for missing dependencies

### Fixed
- Version string now correctly reports 0.2.0

## [0.1.0] - 2025-XX-XX

### Added
- Initial release
- PQEncoder for Product Quantization encoding
- PQKMeans clustering wrapper
- DataStreamer for memory-efficient data loading
- FingerprintCalculator for MQN and Morgan fingerprints
- Basic helper functions (`save_chunk`, `format_time`)
