# Getting Started

## Installation

### From PyPI (recommended)

```bash
pip install chelombus
```

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
streamer = DataStreamer(path='molecules.smi', chunksize=100000)

# 2. Calculate MQN fingerprints
fp_calc = FingerprintCalculator()
for smiles_chunk in streamer.parse_input():
    fingerprints = fp_calc.FingerprintFromSmiles(smiles_chunk, fp='mqn')
    # Save fingerprints...

# 3. Train PQ encoder on sample
encoder = PQEncoder(k=256, m=6, iterations=20)
encoder.fit(training_fingerprints)

# 4. Transform all fingerprints to PQ codes
pq_codes = encoder.transform(fingerprints)

# 5. Cluster with PQk-means
clusterer = PQKMeans(encoder, k=100000)
labels = clusterer.fit_predict(pq_codes)
```

## Pipeline Scripts

For large-scale processing, use the pipeline scripts in `scripts/`:

| Script | Description |
|---|---|
| `calculate_fingerprints.py` | Compute MQN fingerprints from SMILES |
| `train_encoder.py` | Train the PQ encoder on a sample |
| `generate_pqcodes.py` | Transform fingerprints to PQ codes |
| `fit_pqkmeans.py` | Fit PQk-means clustering |
| `assign_clusters.py` | Assign molecules to clusters |
| `select_k.py` | Hyperparameter sweep for choosing k |
| `run_pipeline.py` | Run the full pipeline end-to-end |

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_encoder.py -v
```
