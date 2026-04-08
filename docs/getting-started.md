# Getting Started

## Installation

### From PyPI (recommended)

```bash
pip install chelombus
```

### GPU Support (optional)

GPU acceleration requires PyTorch and Triton. Install them separately for your CUDA version:

```bash
pip install torch triton
```

When CUDA is available, `PQEncoder.fit()`, `.transform()`, `PQKMeans.fit()`, and `.predict()` automatically use the GPU. No code changes needed — the library falls back to CPU transparently if CUDA is not detected.

### From Source

```bash
git clone https://github.com/afloresep/chelombus.git
cd chelombus
pip install -e .
```

!!! warning "Apple Silicon (M1/M2/M3)"
    The `pqkmeans` library is not currently supported on Apple Silicon Macs. Clustering functionality requires an x86_64 system.

## Quick Start

```python
from chelombus import DataStreamer, FingerprintCalculator, PQEncoder, PQKMeans

# 1. Stream SMILES in chunks
streamer = DataStreamer()
smiles_gen = streamer.parse_input('molecules.smi', chunksize=100000, smiles_col=0)

# 2. Calculate MQN fingerprints
fp_calc = FingerprintCalculator()
for smiles_chunk in smiles_gen:
    fingerprints = fp_calc.FingerprintFromSmiles(smiles_chunk, fp='mqn')
    # Save fingerprints...

# 3. Train PQ encoder (uses GPU automatically if available)
encoder = PQEncoder(k=256, m=6, iterations=20)
encoder.fit(training_fingerprints, device='auto')

# 4. Transform all fingerprints to PQ codes
pq_codes = encoder.transform(fingerprints)

# 5. Cluster with PQk-means
clusterer = PQKMeans(encoder, k=100000)
labels = clusterer.fit_predict(pq_codes)
```

## GPU Acceleration

When PyTorch and Triton are installed with CUDA support, the entire PQ pipeline runs on GPU.

**GPU benchmarks on 10M Enamine MQN fingerprints** (RTX 4070 Ti SUPER 16GB, k=10,000):

| Stage | Time |
|:---|---:|
| `encoder.fit()` (10M, m=6, k=256) | 12.1s |
| `encoder.transform()` (10M) | 1.8s |
| `clusterer.fit()` (10M, k=10K, 5 iters) | 15.4s |
| `clusterer.predict()` (10M) | 2.1s |
| **Total** | **31.4s** |

At 1B scale with k=100K, the full pipeline completes in ~2.9 hours on the same GPU.

### Explicit device control

```python
# Force GPU
encoder.fit(X_train, device='gpu')

# Force CPU
encoder.fit(X_train, device='cpu')

# Auto-detect (default for fit; transform/predict auto-detect always)
encoder.fit(X_train, device='auto')
```

The GPU path uses VRAM-aware batching — it queries free GPU memory and sizes batches automatically so you don't need to worry about OOM errors.

## Visualization

Chelombus uses [tmap2](https://github.com/afloresep/tmap2) for interactive TMAP visualizations.

### From Python

```python
from chelombus import sample_from_cluster
from chelombus.utils.visualization import create_tmap, representative_tmap

# Visualize molecules from a cluster
df = sample_from_cluster('results/', cluster_id=42, n=1000)
create_tmap(
    smiles=df['smiles'].tolist(),
    fingerprint='morgan',
    properties=['mw', 'logp', 'qed', 'n_rings'],
    tmap_name='cluster_42',
)

# Representative TMAP with cluster IDs
representative_tmap(
    smiles=rep_smiles,
    cluster_ids=rep_cluster_ids,
    fingerprint='mqn',
    tmap_name='representatives',
)
```

### From CLI

```bash
chelombus-tmap --smiles molecules.smi --fingerprint morgan --output my_tmap
chelombus-tmap --cluster-file representatives.csv --output rep_tmap
```

## Pipeline Scripts

For large-scale processing, use the pipeline scripts in `scripts/`:

| Script | Description |
|---|---|
| `cluster_smiles.py` | End-to-end: SMILES to clustered parquet |
| `benchmark_1B_pipeline.py` | Full pipeline benchmark at billion scale |
| `benchmark_gpu_predict.py` | GPU vs CPU predict benchmarks |
| `k_selection_gpu.py` | GPU-accelerated k hyperparameter sweep |

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_encoder.py -v
```
