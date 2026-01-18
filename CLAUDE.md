# Chelombus (formerly SPQR/SPiQ) - Project Context for AI-Assisted Development

> **Last Updated**: 2026-01-18
> **Status**: Pre-release, preparing for paper publication
> **Goal**: Production-ready library for billion-scale molecular clustering

---

## 1. PROJECT OVERVIEW

### What is Chelombus?

Chelombus is a Python library for **large-scale clustering and visualization of molecular datasets** (up to billions of molecules) on commodity hardware. It implements the "Nested TMAP" framework described in the accompanying paper.

**Core Pipeline**:
```
SMILES → MQN Fingerprints → PQ Encoding → PQk-means Clustering → Nested TMAP Visualization
```

**Key Innovation**: Using Product Quantization (PQ) to compress 42-dimensional MQN vectors into 6-byte codes, enabling clustering of 9.6 billion molecules on a single workstation (AMD Ryzen 7, 64GB RAM).

### Paper Reference
- **Title**: "Nested Tree-maps to visualize Billions of Molecules"
- **Authors**: Alejandro Flores Sepúlveda, Jean-Louis Reymond
- **Website**: https://chelombus.gdb.tools
- **Publication Timeline**: ~1 month

---

## 2. NAMING DECISION

**Old Names**: SPQR, SPiQ, spiq
**New Name**: **Chelombus**

This rename must be applied throughout:
- Package name: `spiq/` → `chelombus/`
- Module imports
- README, documentation
- GitHub repository
- pyproject.toml, setup.py
- All references in code/comments

---

## 3. ARCHITECTURE OVERVIEW

### Current Module Structure
```
chelombus/
├── streamer/
│   └── data_streamer.py      # Memory-efficient SMILES streaming
├── encoder/
│   ├── encoder_base.py       # Abstract base class (sklearn-compatible)
│   └── encoder.py            # PQEncoder - Product Quantization
├── clustering/
│   └── PyQKmeans.py          # Wrapper around pqkmeans C++ library
├── utils/
│   ├── fingerprints.py       # Morgan & MQN fingerprint calculators
│   ├── helper_functions.py   # save_chunk, format_time, etc.
│   └── visualization.py      # TMAP generation (needs major work)
└── __init__.py               # Package exports
```

### Data Flow
```
1. DataStreamer.parse_input() → yields SMILES chunks
2. FingerprintCalculator.FingerprintFromSmiles() → MQN/Morgan FPs
3. PQEncoder.fit() → trains on sample, creates codebook
4. PQEncoder.transform() → converts FPs to PQ codes (6 bytes each)
5. PQKMeans.fit() → clusters PQ codes (uses symmetric distance)
6. PQKMeans.predict() → assigns all molecules to clusters
7. TMAP visualization → primary (cluster reps) + nested (cluster contents)
```

### Memory Efficiency
- Original MQN: 42 dimensions × 4 bytes = 168 bytes/molecule
- PQ Code: 6 bytes/molecule (m=6 subvectors, k=256 centroids)
- **Compression ratio**: 28x (actual: up to 256x with uint8 dtype)

---

## 4. CRITICAL ISSUES TO FIX (Priority 1)

### 4.1 Bugs
| File | Line | Issue |
|------|------|-------|
| `scripts/generate_pqcodes.py` | 72 | Global variable `pq_encoder` used before definition |
| `scripts/get_training_data.py` | 104 | Undefined variable `fp_df_sample` in .npy branch |
| `spiq/utils/visualization.py` | 220 | `file.split(f'/n')` → should be `file.read().split('\n')` |
| `setup.py` | 14 | Typo `"tqmd"` → `"tqdm"` |
| `setup.py` | 19 | Dead console script references non-existent `spq.main:main` |
| `spiq/utils/visualization.py` | 164 | References non-existent column `"Fraction Aromatic_atoms"` |

### 4.2 Code Debt
- `spiq/clustering/PyQKmeans.py`: 80+ lines of commented-out incomplete Python implementation (lines 52-141)
- `spiq/utils/fingerprints.py`: Invalid SMILES returns random fingerprints instead of None/error (dangerous for data integrity)
- Missing `__init__.py` exports in encoder/, streamer/, clustering/

### 4.3 Missing Tests
- `tests/test_clustering.py`: Empty file (0 tests for PQKMeans)
- `tests/test_trainer.py`: Empty file
- No integration tests for full pipeline
- No tests for visualization module

---

## 5. IMPROVEMENT ROADMAP (v1.0 Release)

### Week 1: Critical Fixes & Rename
- [ ] Fix bug: `generate_pqcodes.py:72` - global variable
- [ ] Fix bug: `get_training_data.py:104` - undefined variable
- [ ] Fix bug: `visualization.py:220` - file.split syntax
- [ ] Fix bug: `visualization.py:164` - wrong column name
- [ ] Fix bug: `setup.py:14` - tqmd typo
- [ ] Fix bug: `setup.py:19` - dead console script
- [ ] Remove 80+ lines commented code in `PyQKmeans.py`
- [ ] Fix `fingerprints.py` - return None for invalid SMILES (not random FP)
- [ ] **Rename**: `spiq/` → `chelombus/` (all files, imports, docs)
- [ ] Update `pyproject.toml` and `setup.py` with new name
- [ ] Update README.md

### Week 2: Core Functionality & Tests
- [ ] Add `__init__.py` exports to all submodules
- [ ] Write tests for `PQKMeans` (test_clustering.py)
- [ ] Add integration test (full pipeline: SMILES → clusters)
- [ ] Add `duckdb` dependency
- [ ] Implement `export_cluster()` function
- [ ] Implement `export_all_clusters()` function
- [ ] Pin dependency versions in requirements.txt

### Week 3: Pipeline & Visualization
- [ ] Create unified `Pipeline` class with high-level API
- [ ] Add `pipeline.get_cluster(id)` method
- [ ] Add `pipeline.export_cluster(id, path)` method
- [ ] Fix visualization module bugs
- [ ] Create `generate_tmap_for_cluster(cluster_id)` function
- [ ] Add type hints to core public APIs
- [ ] Add proper logging (replace print statements)

### Week 4: Polish & Release
- [ ] CLI commands: `chelombus cluster`, `chelombus export-cluster`, `chelombus visualize`
- [ ] Update tutorial notebook with new API
- [ ] Create quick-start example (small dataset, <5 min)
- [ ] Verify installation works: `pip install -e .`
- [ ] Final testing on real data
- [ ] Tag v1.0.0 release

### Future (v2.0)
- [ ] Pure Python PQKMeans implementation
- [ ] Additional fingerprint types (ECFP, RDKit descriptors)
- [ ] GPU acceleration (faiss integration)
- [ ] Nearest neighbor search with PQTable
- [ ] Distributed processing (Ray/Spark)

---

## 6. API DESIGN GOALS

### Current API (Fragmented)
```python
# Current - requires manual orchestration
from chelombus import DataStreamer, FingerprintCalculator, PQEncoder

streamer = DataStreamer(path, chunksize=100000)
fp_calc = FingerprintCalculator()
encoder = PQEncoder(k=256, m=6)

# Manual chunking, saving, loading...
```

### Target API (Unified)
```python
from chelombus import Pipeline

# Initialize pipeline with parameters
pipeline = Pipeline(
    fingerprint='mqn',        # 'mqn' or 'morgan'
    n_clusters=100000,
    n_subvectors=6,           # m parameter for PQ
    n_centroids=256,          # k parameter for PQ
    chunksize=100000          # streaming chunk size
)

# Step 1: Train PQ encoder on sample data
pipeline.fit_encoder(training_data_path, n_samples=50_000_000)

# Step 2: Train PQk-means on PQ codes
pipeline.fit_clusterer(pq_codes_path, n_training=1_000_000_000)

# Step 3: Process all data (streaming) - saves chunks with cluster_id
pipeline.transform(input_path, output_dir='results/')

# Query cluster contents (returns pandas DataFrame)
df = pipeline.get_cluster(42)

# Export single cluster to file
pipeline.export_cluster(42, 'cluster_42.parquet')

# Export all clusters (for HPC/SLURM batch processing)
pipeline.export_all_clusters('clusters/')

# Generate visualization for a cluster
pipeline.visualize_cluster(42, output='cluster_42_tmap.html')

# Generate primary TMAP (cluster representatives)
pipeline.visualize_overview(output='primary_tmap.html')
```

### Convenience Functions (Module-Level)
```python
import chelombus

# Quick cluster query without full pipeline
df = chelombus.query_cluster('results/', cluster_id=42)

# Export cluster
chelombus.export_cluster('results/', cluster_id=42, output='cluster_42.parquet')

# Generate TMAP from SMILES file
chelombus.create_tmap('cluster_42.parquet', output='cluster_42_tmap.html')
```

### CLI Design
```bash
# Full pipeline (most users)
chelombus run --input data.smi --output results/ --n-clusters 100000

# Step-by-step (advanced users)
chelombus fingerprint --input data.smi --output fps/ --type mqn
chelombus train-encoder --input fps/ --output encoder.pkl --samples 50000000
chelombus encode --input fps/ --output pqcodes/ --encoder encoder.pkl
chelombus train-clusterer --input pqcodes/ --output clusterer.pkl --n-clusters 100000
chelombus assign --input pqcodes/ --output results/ --clusterer clusterer.pkl

# Cluster utilities
chelombus query --input results/ --cluster 42                    # Print to stdout
chelombus export-cluster --input results/ --cluster 42 --output cluster_42.parquet
chelombus export-all --input results/ --output clusters/         # For HPC batch

# Visualization
chelombus visualize --input results/ --cluster 42 --output cluster_42_tmap.html
chelombus visualize-overview --input results/ --output primary_tmap.html
```

### Low-Level API (For Custom Workflows)
```python
from chelombus import DataStreamer, FingerprintCalculator, PQEncoder, PQKMeans
from chelombus.utils import export_cluster, create_tmap

# Individual components remain accessible for advanced users
streamer = DataStreamer(path, chunksize=100000)
fp_calc = FingerprintCalculator()
encoder = PQEncoder(k=256, m=6)
clusterer = PQKMeans(encoder, n_clusters=100000)
```

---

## 7. KEY TECHNICAL DECISIONS

### Why MQN Fingerprints?
- 42-dimensional (compact, divisible by 6)
- Count-based (works with L2 distance in PQ)
- Interpretable (atom types, bonds, rings, etc.)
- Fast to compute

### Why Product Quantization?
- Compresses vectors dramatically (168 → 6 bytes)
- Enables billion-scale clustering in RAM
- Preserves distance relationships (Symmetric Distance ≈ Euclidean)
- Well-studied algorithm with proven performance

### Why PQk-means?
- Works directly on PQ codes (no reconstruction)
- Uses Symmetric Distance (lookup-table based)
- Scales to billions of data points
- Reference implementation available

### Why TMAP for Visualization?
- Handles millions of points
- Creates interpretable tree structure
- Interactive web-based display
- Already proven for molecular visualization

---

## 8. DEPENDENCIES

### Core (Required)
```
numpy>=1.20.0
pandas>=1.3.0
rdkit>=2022.03
scikit-learn>=1.0.0
tqdm>=4.60.0
```

### Clustering (Required for full pipeline)
```
pqkmeans  # C++ library, pip install from GitHub
```

### Visualization (Optional)
```
tmap>=1.0.0
faerun>=0.4.0
pandarallel>=1.5.0
```

### Development
```
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
isort>=5.10.0
mypy>=0.950
```

---

## 9. FILE FORMAT CONVENTIONS

### Input Formats
- `.smi`, `.txt`: SMILES strings (one per line, or with tab-separated columns)
- `.sdf`, `.sd`: MDL SD files (streamed with RDKit)
- `.csv`, `.parquet`: DataFrames with SMILES column

### Intermediate Formats
- `.npy`: NumPy arrays (fastest for fingerprints)
- `.parquet`: When SMILES association needed

### Output Formats
- `.pkl`, `.joblib`: Model serialization
- `.npy`: PQ codes, cluster assignments
- `.html`: TMAP visualizations

---

## 10. PERFORMANCE BENCHMARKS (Reference)

From the paper (AMD Ryzen 7 8700F, 64GB RAM):

| Stage | Data Size | Time |
|-------|-----------|------|
| MQN Calculation | 9.6B molecules | ~hours (streaming) |
| PQ Codebook Training | 50M vectors | ~70 minutes |
| PQ Encoding | 9.6B vectors | ~3 hours |
| PQk-means Training | 1B PQ codes | ~2 days 5 hours |
| Cluster Assignment | 9.6B PQ codes | ~4 hours |
| Primary TMAP | 100K points | ~1 minute |
| Secondary TMAP | per cluster | ~1 minute each |

---

## 11. KNOWN LIMITATIONS

1. **MQN-dominated clustering**: Large count values (HAC, number of carbons) dominate distance. Normalized/weighted MQN could improve results.

2. **Fixed k**: Number of clusters must be set a-priori. No automatic k selection.

3. **External dependency**: Requires pqkmeans C++ library (installation can be tricky).

4. **No incremental updates**: Cannot add new molecules without re-clustering.

5. **Visualization scalability**: Secondary TMAPs must be generated per-cluster (100K clusters = 100K TMAPs if all pre-computed).

---

## 12. TESTING STRATEGY

### Unit Tests (Current: 24 tests)
- `test_data_streamer.py`: 5 tests ✓
- `test_encoder.py`: 8 tests ✓
- `test_fingerprints.py`: 7 tests ✓
- `test_utils.py`: 4 tests ✓
- `test_clustering.py`: 0 tests ✗ (empty)
- `test_trainer.py`: 0 tests ✗ (empty)

### Needed Tests
- [ ] Clustering module tests
- [ ] Integration tests (full pipeline)
- [ ] Visualization tests
- [ ] Performance regression tests
- [ ] Edge cases (empty files, invalid data, etc.)

### Running Tests
```bash
pytest tests/ -v
pytest tests/test_encoder.py -v  # Single file
```

---

## 13. DESIGN DECISIONS (RESOLVED)

### 13.1 Cluster File Organization
**Decision**: Chunked output files + DuckDB for on-demand queries

**Rationale**:
- Writing to 100k files per chunk causes massive I/O overhead
- Parquet doesn't support appending (must read-modify-write)
- Parallel workers can't safely write to same file

**Implementation**:
```
Pipeline output:
  results/chunk_00001.parquet  (smiles, cluster_id)
  results/chunk_00002.parquet  (smiles, cluster_id)
  ...

On-demand export:
  DuckDB query: SELECT smiles FROM 'results/*.parquet' WHERE cluster_id = 42
  → cluster_42.parquet
```

**API**:
```python
# Query cluster contents (returns DataFrame)
pipeline.get_cluster(42)

# Export cluster to file
pipeline.export_cluster(42, "cluster_42.parquet")

# Batch export all clusters (for SLURM)
pipeline.export_all_clusters("clusters/")
```

**CLI**:
```bash
chelombus export-cluster --input results/ --cluster 42 --output cluster_42.parquet
chelombus export-all --input results/ --output clusters/  # For HPC batch
```

**Dependencies**: Add `duckdb` to requirements (lightweight, no server needed)

### 13.2 Secondary TMAP Generation
**Decision**: On-demand generation only

**Rationale**:
- 100k clusters × 1 minute each = ~70 days to pre-compute
- Not feasible on single workstation
- HPC users can batch with SLURM using exported cluster files

**Workflow**:
1. User clicks cluster in primary TMAP (or uses CLI)
2. System queries cluster contents via DuckDB
3. Generates TMAP on-the-fly (~1 minute)
4. Caches result for future access (optional)

### 13.3 Feature Scope for v1.0
**Decision**: Limit features, ship reliable core

**Included in v1.0**:
- MQN fingerprints (primary use case)
- Morgan fingerprints (already implemented)
- PQ encoding with sklearn KMeans
- PQk-means clustering (C++ library wrapper)
- Primary TMAP visualization
- On-demand secondary TMAP generation
- Cluster export utilities

**Deferred to v2.0**:
- Pure Python PQk-means implementation
- Additional fingerprint types (ECFP, RDKit descriptors)
- GPU acceleration (faiss integration)
- Nearest neighbor search with PQTable
- Distributed processing support

### 13.4 Dependencies
**Decision**: pqkmeans C++ dependency is acceptable for v1.0

**Core dependencies** (required):
```
numpy>=1.20.0
pandas>=1.3.0
rdkit>=2022.03
scikit-learn>=1.0.0
tqdm>=4.60.0
duckdb>=0.9.0  # NEW - for cluster queries
```

**Clustering** (required for full pipeline):
```
pqkmeans  # C++ library
```

**Visualization** (optional):
```
tmap>=1.0.0
faerun>=0.4.0
```

---

## 14. CONVENTIONS FOR THIS PROJECT

### Code Style
- Black formatter (line length 88)
- isort for imports
- Type hints for public APIs
- Google-style docstrings

### Git Workflow
- Main branch: `master`
- Feature branches: `feature/description`
- Commits: Conventional commits format

### Documentation
- README.md: User-facing documentation
- CLAUDE.md: This file (AI context)
- docs/: Sphinx API documentation

---

## 15. CONTEXT FOR AI CONVERSATIONS

When resuming work on this project:

1. **Read this file first** to understand the project state
2. **Check git log** for recent changes
3. **Run tests** to verify nothing is broken: `pytest tests/`
4. **Refer to Phase markers** in Section 5 for priority

Key files to understand:
- `spiq/encoder/encoder.py`: Core PQ implementation
- `spiq/utils/fingerprints.py`: Fingerprint calculation
- `spiq/streamer/data_streamer.py`: Memory-efficient streaming
- `docs/nestedTMAPs-2.pdf`: The academic paper

---

## 16. IMPLEMENTATION SNIPPETS

### Cluster Export with DuckDB
```python
# chelombus/utils/cluster_io.py
import duckdb
from pathlib import Path

def query_cluster(results_dir: str, cluster_id: int) -> 'pd.DataFrame':
    """Query all molecules from a specific cluster."""
    import pandas as pd
    query = f"""
        SELECT smiles, cluster_id
        FROM '{results_dir}/*.parquet'
        WHERE cluster_id = {cluster_id}
    """
    return duckdb.query(query).df()

def export_cluster(results_dir: str, cluster_id: int, output_path: str) -> int:
    """Export a single cluster to a file. Returns molecule count."""
    query = f"""
        SELECT smiles
        FROM '{results_dir}/*.parquet'
        WHERE cluster_id = {cluster_id}
    """
    result = duckdb.query(query)

    if output_path.endswith('.parquet'):
        result.write_parquet(output_path)
    elif output_path.endswith('.csv'):
        result.write_csv(output_path)
    else:
        raise ValueError("Output must be .parquet or .csv")

    return result.fetchone()[0] if result.fetchone() else 0

def export_all_clusters(results_dir: str, output_dir: str, format: str = 'parquet') -> dict:
    """Export all clusters to individual files. Returns {cluster_id: count}."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all cluster IDs
    cluster_ids = duckdb.query(f"""
        SELECT DISTINCT cluster_id
        FROM '{results_dir}/*.parquet'
        ORDER BY cluster_id
    """).fetchall()

    counts = {}
    for (cid,) in tqdm(cluster_ids, desc="Exporting clusters"):
        ext = 'parquet' if format == 'parquet' else 'csv'
        output_path = f"{output_dir}/cluster_{cid:06d}.{ext}"
        counts[cid] = export_cluster(results_dir, cid, output_path)

    return counts

def get_cluster_stats(results_dir: str) -> 'pd.DataFrame':
    """Get statistics for all clusters."""
    query = f"""
        SELECT
            cluster_id,
            COUNT(*) as molecule_count
        FROM '{results_dir}/*.parquet'
        GROUP BY cluster_id
        ORDER BY cluster_id
    """
    return duckdb.query(query).df()
```

### Pipeline Transform (Chunked Output)
```python
# In Pipeline.transform()
def transform(self, input_path: str, output_dir: str):
    """Process all data and save with cluster assignments."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for chunk_idx, smiles_chunk in enumerate(self.streamer.parse_input(input_path)):
        # Calculate fingerprints
        fps = self.fp_calc.FingerprintFromSmiles(smiles_chunk, fp=self.fingerprint)

        # Encode to PQ codes
        pq_codes = self.encoder.transform(fps, verbose=0)

        # Assign clusters
        cluster_ids = self.clusterer.predict(pq_codes)

        # Save chunk with cluster assignments
        df = pd.DataFrame({
            'smiles': smiles_chunk,
            'cluster_id': cluster_ids
        })
        df.to_parquet(f"{output_dir}/chunk_{chunk_idx:05d}.parquet", index=False)

    self._results_dir = output_dir
```

---

## CHANGELOG

### 2026-01-18 - Critical Bug Fixes & Package Rename (Session 1)

**Bugs Fixed**:
- `scripts/generate_pqcodes.py`: Removed global variable, pass `pq_encoder` as parameter
- `scripts/get_training_data.py`: Fixed undefined variable `fp_df_sample`, added missing for loop
- `chelombus/utils/visualization.py`: Fixed `file.split(f'/n')` → proper file reading
- `chelombus/utils/visualization.py`: Fixed wrong column names in `representative_tmap()`
- `setup.py`: Fixed typo `tqmd` → `tqdm`, removed dead console script
- `chelombus/clustering/PyQKmeans.py`: Removed 80+ lines of commented dead code
- `chelombus/utils/fingerprints.py`: Invalid SMILES now returns `None` instead of random FP

**Package Rename**:
- Renamed `spiq/` → `chelombus/`
- Updated all imports across codebase (Python files, notebooks, scripts)
- Updated `pyproject.toml` with new name, optional dependencies, URLs
- Updated `setup.py` with new name, classifiers, optional dependencies
- Updated `README.md` with new branding, quick start, citation info

**Configuration**:
- Added optional dependency groups: `[clustering]`, `[visualization]`, `[io]`, `[dev]`, `[all]`
- Made PQKMeans import optional (doesn't fail if pqkmeans not installed)
- Updated `requirements.txt` with version pins

**Tests**:
- Updated `test_calculate_morgan_fp_invalid` to expect `None` (not random FP)
- 22/24 tests passing (2 failures due to missing optional deps, not code issues)

---

### 2026-01-18 - Initial Analysis & Design Decisions
- Created CLAUDE.md with comprehensive project analysis
- Identified 6 critical bugs
- Documented 24 existing tests, 2 empty test files
- **Resolved**: Cluster file organization → Chunked output + DuckDB queries
- **Resolved**: Secondary TMAP → On-demand generation only
- **Resolved**: v1.0 scope → Limited features, ship reliable core
- **Resolved**: Dependencies → pqkmeans C++ acceptable, add duckdb
- Defined unified Pipeline API design
- Created 4-week implementation roadmap
- Added implementation snippets for cluster export
