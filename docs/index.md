# Chelombus

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
