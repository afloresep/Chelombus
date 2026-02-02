# Chelombus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/afloresep/chelombus)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Billion-scale molecular clustering and visualization on commodity hardware.**

Chelombus enables interactive exploration of ultra-large chemical datasets (up to billions of molecules) using Product Quantization and nested TMAPs. Process the entire Enamine REAL database (9.6B molecules) on a single workstation.

**Live Demo**: [https://chelombus.gdb.tools](https://chelombus.gdb.tools)

## Overview

Chelombus implements the "Nested TMAP" framework for visualizing billion-sized molecular datasets:

```
SMILES → MQN Fingerprints → PQ Encoding → PQk-means Clustering → Nested TMAPs
```

**Key Features**:
- **Scalability**: Stream billions of molecules without loading everything into memory
- **Efficiency**: Compress 42-dimensional MQN vectors to 6-byte PQ codes (28x compression)
- **Visualization**: Navigate from global overview to individual molecules in two clicks
- **Accessibility**: Runs on commodity hardware (tested: AMD Ryzen 7, 64GB RAM)

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

## Platform Notes

**Apple Silicon (M1/M2/M3)**: The `pqkmeans` library is not currently supported on Apple Silicon Macs. My plan is to rewrite pqkmeans with Silicon and GPU support but that's for a future release... For now, clustering functionality requires an x86_64 system.

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

## Project Structure

```
chelombus/
├── chelombus/
│   ├── encoder/          # Product Quantization encoder
│   ├── clustering/       # PQk-means wrapper
│   ├── streamer/         # Memory-efficient data streaming
│   └── utils/            # Fingerprints, visualization, helpers
├── scripts/              # Pipeline scripts
├── examples/             # Tutorial notebooks
└── tests/                # Unit tests
```


## Documentation

- **Tutorial**: See `examples/tutorial.ipynb` for a hands-on introduction
- **Large-scale example**: See `examples/enamine_1B_clustering.ipynb`
- **API Reference**: Generated from docstrings using Sphinx (see `docs/`)

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_encoder.py -v
```

## Citation

If you use Chelombus in your research, please cite:

```bibtex
@article{chelombus2025,
  title={Nested TMAPs to visualize Billions of Molecules},
  author={Flores Sepulveda, Alejandro and Reymond, Jean-Louis},
  journal={},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for new functionality
4. Submit a pull request

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [PQk-means](https://github.com/DwangoMediaVillage/pqkmeans) by Matsui et al.
- [TMAP](https://github.com/reymond-group/tmap) by Probst & Reymond
- [RDKit](https://www.rdkit.org/) for cheminformatics functionality
- Swiss National Science Foundation (grant no. 200020_178998)
