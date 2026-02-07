# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
