"""Hyperparameter selection for PQk-means: sweep over k values on a subsample.

Fits PQk-means for several k values on a subsample of PQ codes and reports
inertia, cluster size distribution, and empty cluster fraction. Results are
saved to a CSV and optionally plotted.

The subsample approach works because the *relative* ranking of k values
(elbow shape) is stable across representative samples — we compare k values
against each other, not absolute inertia.

Usage:
    python scripts/select_k.py \\
        --pq-codes data/pq_codes.npy \\
        --encoder models/encoder.joblib \\
        --n-subsample 10000000 \\
        --k-values 10000 25000 50000 100000 200000 500000 \\
        --iterations 10 \\
        --output results/k_selection.csv \\
        --plot results/k_selection.png

    # Quick test with random data (no --pq-codes):
    python scripts/select_k.py \\
        --encoder models/encoder.joblib \\
        --n-subsample 1000000 \\
        --k-values 1000 5000 10000 50000
"""
import argparse
import os
import time

import numpy as np
from numba import njit, prange

from chelombus import PQEncoder
from chelombus import FingerprintCalculator
from chelombus.clustering.PyQKmeans import PQKMeans, _build_distance_tables
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pq-codes", type=str, default=None,
                    help="Path to .npy file with PQ codes (N, m) uint8. "
                         "If omitted, random codes are generated.")
    p.add_argument("--encoder", type=str, required=True,
                    help="Path to a trained PQEncoder (.joblib)")
    p.add_argument("--n-subsample", type=int, default=10_000_000,
                    help="Subsample size for fitting (default: 10M)")
    p.add_argument("--k-values", type=int, nargs="+",
                    default=[10_000, 25_000, 50_000, 100_000, 200_000, 500_000],
                    help="k values to test")
    p.add_argument("--iterations", type=int, default=10,
                    help="PQk-means iterations per k (default: 10)")
    p.add_argument("--output", type=str, default="k_selection.csv",
                    help="Output CSV path")
    p.add_argument("--plot", type=str, default=None,
                    help="Save elbow plot to this path (requires matplotlib)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@njit(parallel=True, cache=True)
def _predict_with_inertia(pq_codes, centers, dtables):
    """Assign labels AND compute per-point distance to its center."""
    n = pq_codes.shape[0]
    m = pq_codes.shape[1]
    n_centers = centers.shape[0]
    labels = np.empty(n, dtype=np.int64)
    distances = np.empty(n, dtype=np.float32)
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
        distances[i] = best_dist
    return labels, distances


def compute_metrics(labels, distances, k):
    """Compute clustering quality metrics from labels and distances."""
    inertia = float(np.sum(distances))
    avg_dist = float(np.mean(distances))
    counts = np.bincount(labels.astype(np.intp), minlength=k)
    n_empty = int(np.sum(counts == 0))
    sizes = counts[counts > 0]
    return {
        "k": k,
        "inertia": inertia,
        "avg_dist": avg_dist,
        "n_empty_clusters": n_empty,
        "pct_empty": 100.0 * n_empty / k,
        "cluster_size_mean": float(np.mean(sizes)),
        "cluster_size_median": float(np.median(sizes)),
        "cluster_size_std": float(np.std(sizes)),
        "cluster_size_min": int(np.min(sizes)),
        "cluster_size_max": int(np.max(sizes)),
    }


def make_plot(results, path):
    """Generate elbow plot with inertia and empty cluster fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in results]
    inertias = [r["inertia"] for r in results]
    pct_empty = [r["pct_empty"] for r in results]
    fit_times = [r["fit_time_s"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Elbow plot
    ax = axes[0]
    ax.plot(ks, inertias, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Inertia (sum of distances)")
    ax.set_title("Elbow Plot")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Empty clusters
    ax = axes[1]
    ax.bar(range(len(ks)), pct_empty, color="coral", alpha=0.8)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"{k//1000}K" for k in ks], rotation=45)
    ax.set_ylabel("Empty clusters (%)")
    ax.set_title("Empty Cluster Fraction")
    ax.set_ylim(0, max(max(pct_empty) * 1.3, 1.0))
    ax.grid(True, alpha=0.3, axis="y")

    # Fit time
    ax = axes[2]
    ax.plot(ks, [t / 60 for t in fit_times], "s-", color="seagreen", linewidth=2)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Fit time (min)")
    ax.set_title("Fit Time")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"PQk-means k Selection (n={results[0]['n_subsample']:,})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {path}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n_cores = os.cpu_count()
    print(f"Loading encoder from {args.encoder} ...")
    try:
        encoder = PQEncoder.load(args.encoder)
    except (TypeError, KeyError):
        # Encoder is embedded inside a PQKMeans model
        model = PQKMeans.load(args.encoder)
        encoder = model.encoder
    m = encoder.m
    print(f"  m={m} subvectors, k'={encoder.k} codewords\n")

    if args.pq_codes is not None:
        clusterer = PQKMeans.load('/home/afloresep/work/Chelombus/models/paper_clusterer.joblib')
        encoder = PQEncoder.load('/home/afloresep/work/Chelombus/models/paper_encoder.joblib')

        from chelombus.streamer.data_streamer import DataStreamer

        all_codes= np.zeros((120_000_000, 42))
        print("Calculating fingerprints")
        ds =DataStreamer()
        fp = FingerprintCalculator()
        for i, smiles in enumerate(ds.parse_input('/mnt/10tb_hdd/cleaned_enamine_10b/output_file_0.cxsmiles', chunksize=1_000_000, smiles_col=1)):
            all_codes[i*1_000_000:(i+1)*1_000_000, :]= fp.FingerprintFromSmiles(smiles, 'mqn')
            print(f'Smiles Chunk {((i+1)*1_000_000):,} done')
            if i == 99:
                break
    
        all_codes = all_codes[:100_000_000]
        n_total = all_codes.shape[0]
        n_sub = min(args.n_subsample, n_total)
        if n_sub < n_total:
            idx = rng.choice(n_total, size=n_sub, replace=False)
            idx.sort()
            fingerprints = all_codes[idx]
        else:
            fingerprints = all_codes
        print(f"  Computed {n_total:,} fingerprints, subsampled to {n_sub:,}")
        print(f"  PQ-encoding {fingerprints.shape[0]:,} fingerprints ...")
        pq_codes = encoder.transform(fingerprints)
        print(f"  PQ codes shape: {pq_codes.shape}, dtype: {pq_codes.dtype}\n")
    else:
        n_sub = args.n_subsample
        print(f"No --pq-codes provided, generating {n_sub:,} random PQ codes\n")
        pq_codes = rng.integers(0, 256, size=(n_sub, m), dtype=np.uint8)

    # Precompute distance tables 
    dtables = _build_distance_tables(encoder.codewords)

    # Numba warmup
    print("Numba JIT warmup ...")
    _dummy_centers = rng.integers(0, 256, size=(10, m), dtype=np.uint8)
    _ = _predict_with_inertia(pq_codes[:100], _dummy_centers, dtables)

    # Load checkpoint if exist
    k_values = sorted(args.k_values)
    results = []
    completed_ks = set()

    import csv
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert types back from CSV strings
                parsed = {}
                for key, val in row.items():
                    try:
                        if "." in val:
                            parsed[key] = float(val)
                        else:
                            parsed[key] = int(val)
                    except (ValueError, TypeError):
                        parsed[key] = val
                results.append(parsed)
                completed_ks.add(int(float(row["k"])))
        if completed_ks:
            print(f"Checkpoint loaded: {len(completed_ks)} k values already done: "
                  f"{sorted(completed_ks)}")
            print(f"Remaining: {[k for k in k_values if k not in completed_ks]}\n")

    # Sweep over k values 
    print(f"{'k':>10}  {'fit':>8}  {'predict':>8}  {'inertia':>14}  "
          f"{'avg_dist':>9}  {'empty%':>7}  {'size_med':>9}  {'size_std':>9}")
    print("-" * 90)

    # Print already-completed rows
    for r in results:
        k = int(r["k"])
        print(f"{k:>10,}  {'done':>8}  {'done':>8}  {r['inertia']:>14,.0f}  "
              f"{r['avg_dist']:>9.3f}  {r['pct_empty']:>6.1f}%  "
              f"{r['cluster_size_median']:>9.0f}  {r['cluster_size_std']:>9.0f}")

    for k in k_values:
        if k in completed_ks:
            continue
        if k >= n_sub:
            print(f"{k:>10,}  SKIPPED (k >= n_subsample)")
            continue

        # Fit
        clusterer = PQKMeans(encoder, k=k, iteration=args.iterations, verbose=False)
        t0 = time.time()
        clusterer.fit(pq_codes)
        t_fit = time.time() - t0

        # Predict + inertia
        centers = clusterer.cluster_centers_.astype(np.uint8)
        t0 = time.time()
        labels, distances = _predict_with_inertia(pq_codes, centers, dtables)
        t_pred = time.time() - t0

        # Metrics
        metrics = compute_metrics(labels, distances, k)
        metrics["fit_time_s"] = t_fit
        metrics["predict_time_s"] = t_pred
        metrics["n_subsample"] = n_sub
        metrics["iterations"] = args.iterations
        results.append(metrics)

        print(f"{k:>10,}  {t_fit:>7.0f}s  {t_pred:>7.0f}s  {metrics['inertia']:>14,.0f}  "
              f"{metrics['avg_dist']:>9.3f}  {metrics['pct_empty']:>6.1f}%  "
              f"{metrics['cluster_size_median']:>9.0f}  {metrics['cluster_size_std']:>9.0f}")

        # Checkpoint: append to CSV after each k 
        write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
        with open(args.output, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
        print(f"           checkpoint saved to {args.output}")

    # Final plot (uses all results including checkpoint) 
    if results:
        # Sort by k for consistent plotting
        results.sort(key=lambda r: int(r["k"]))

        if args.plot:
            make_plot(results, args.plot)

    print("\nDone.")


if __name__ == "__main__":
    main()
