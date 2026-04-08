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
- **GPU Acceleration**: Full CUDA pipeline via PyTorch + Triton kernels for fit, transform, and predict
- **Visualization**: Interactive TMAPs powered by [tmap2](https://github.com/afloresep/tmap2) — navigate from global overview to individual molecules in two clicks
- **Accessibility**: Runs on commodity hardware (tested: AMD Ryzen 7, 64GB RAM); GPU optional

## Project Structure

```
chelombus/
├── chelombus/
│   ├── encoder/          # Product Quantization encoder (CPU + GPU)
│   ├── clustering/       # PQk-means wrapper + Triton kernels
│   ├── streamer/         # Memory-efficient data streaming
│   └── utils/            # Fingerprints, visualization, cluster I/O
├── scripts/              # Pipeline and benchmark scripts
├── examples/             # Tutorial notebooks
└── tests/                # Unit tests
```
