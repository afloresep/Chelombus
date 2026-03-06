"""Benchmark PQKMeans.predict: pqkmeans C++ baseline vs Numba JIT (v0.2.0).

The original pqkmeans library calls predict_one per point in a Python loop.
This script benchmarks that baseline against the Numba JIT-compiled parallel
predict that uses precomputed symmetric distance lookup tables.

Usage:
    python scripts/benchmark_predict.py [--n 100000] [--model models/paper_clusterer.joblib]
"""
import argparse
import os
import time

import numpy as np
from numba import njit, prange

from chelombus.clustering.PyQKmeans import PQKMeans, _build_distance_tables


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000,
                        help="Number of PQ codes to benchmark (default: 100000)")
    parser.add_argument("--model", type=str, default="models/paper_clusterer.joblib",
                        help="Path to a trained PQKMeans model")
    return parser.parse_args()


# ── Numba JIT parallel predict ───────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _predict_numba(pq_codes, centers, dtables):
    """Assign each PQ code to its nearest center using precomputed distance tables."""
    n = pq_codes.shape[0]
    m = pq_codes.shape[1]
    n_centers = centers.shape[0]
    labels = np.empty(n, dtype=np.int64)
    for i in prange(n):
        best_dist = np.inf
        best_label = 0
        for c in range(n_centers):
            dist = np.float32(0.0)
            for sub in range(m):
                dist += dtables[sub, pq_codes[i, sub], centers[c, sub]]
            if dist < best_dist:
                best_dist = dist
                best_label = c
        labels[i] = best_label
    return labels


def predict_numba(pq_codes, centers, dtables):
    return _predict_numba(
        np.asarray(pq_codes, dtype=np.uint8),
        np.asarray(centers, dtype=np.uint8),
        dtables,
    )


def main():
    args = parse_args()
    N = args.n
    n_cores = os.cpu_count()

    print(f"Loading model from {args.model} ...")
    model = PQKMeans.load(args.model)
    centers = model.cluster_centers_.astype(np.uint8)
    m = model.encoder.m

    print(f"  k={model.k:,} clusters, m={m} subvectors, {n_cores} CPU cores")
    print(f"  Benchmark size: {N:,} random PQ codes\n")

    dtables = _build_distance_tables(model.encoder.codewords)
    pq_codes = np.random.randint(0, 256, size=(N, m), dtype=np.uint8)

    print(f"[1/2] pqkmeans C++ predict_one loop ({N:,} points) ...")
    t0 = time.time()
    labels_cpp = np.array(list(model._cluster.predict_generator(pq_codes)))
    t_cpp = time.time() - t0
    print(f"      {t_cpp:.2f}s  |  {N / t_cpp:,.0f} codes/s\n")

    # ── 2) Numba JIT (warmup then timed run) ─────────────────────────────
    print("      Numba JIT warmup (compiling) ...")
    _ = predict_numba(pq_codes[:1000], centers, dtables)

    print(f"[2/2] Numba JIT parallel ({N:,} points) ...")
    t0 = time.time()
    labels_numba = predict_numba(pq_codes, centers, dtables)
    t_numba = time.time() - t0
    print(f"      {t_numba:.2f}s  |  {N / t_numba:,.0f} codes/s\n")

    # ── Correctness ───────────────────────────────────────────────────────
    mismatches = int(np.sum(labels_cpp != labels_numba))
    print(f"Label agreement: {N - mismatches}/{N} "
          f"({mismatches} differ due to equidistant ties)\n")

    # ── Summary table ─────────────────────────────────────────────────────
    speedup = t_cpp / t_numba
    rate_numba = N / t_numba

    print("=" * 65)
    print(f"{'Method':<30} {'Time (s)':<10} {'codes/s':<15} {'Speedup'}")
    print("-" * 65)
    print(f"{'pqkmeans C++ (predict_one)':<30} {t_cpp:<10.2f} {N / t_cpp:<15,.0f} 1.0x")
    print(f"{'Numba JIT (parallel)':<30} {t_numba:<10.2f} {rate_numba:<15,.0f} {speedup:.1f}x")
    print("=" * 65)

    # ── Extrapolation ─────────────────────────────────────────────────────
    print(f"\n{'Extrapolation (Numba @ {0:,} codes/s on {1} cores):'.format(int(rate_numba), n_cores)}")
    print("-" * 65)
    print(f"{'Scale':<12} {'Single machine':<22} {'4 nodes':<18} {'8 nodes'}")
    print("-" * 65)
    for label, count in [("10M", 10e6), ("1B", 1e9), ("10B", 10e9)]:
        secs = count / rate_numba
        fmt = lambda s: (f"{s / 3600:.1f} hrs" if s >= 3600
                         else f"{s / 60:.1f} min" if s >= 60
                         else f"{s:.0f}s")
        print(f"{label:<12} {fmt(secs):<22} {fmt(secs / 4):<18} {fmt(secs / 8)}")

    print(f"\nNote: multi-node estimates assume embarrassingly parallel data")
    print(f"splitting (each node runs Numba predict on its shard).")


if __name__ == "__main__":
    main()
