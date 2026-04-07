"""GPU k-selection benchmark: sweep k values on 100M Enamine fingerprints.

Loads MQN fingerprints from chunks, encodes to PQ codes on GPU,
then fits PQKMeans at each k value on GPU and computes clustering metrics.
All heavy steps (fit + predict) run on GPU.

Usage:
    nohup python -u scripts/k_selection_gpu.py --chunks /path/to/mqn_chunks > k_selection_results.txt 2>&1 &
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chelombus import PQEncoder
from chelombus.clustering.PyQKmeans import PQKMeans, _build_distance_tables


def compute_avg_distance(pq_codes, labels, centers, dtables):
    """Compute average distance from each point to its assigned center.

    O(N*M) — just a lookup per point, no sweep over K.
    """
    N, m = pq_codes.shape
    total = 0.0
    centers_u8 = centers.astype(np.uint8)
    # Vectorized: for each subvector, look up distance between point code and center code
    for sub in range(m):
        point_codes = pq_codes[:, sub]
        center_codes = centers_u8[labels, sub]
        total += dtables[sub, point_codes, center_codes].sum()
    return total / N


def fmt(secs):
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        return f"{secs/60:.1f} min"
    return f"{secs/3600:.1f} hrs"


def iter_chunks(chunk_dir, n_total):
    chunks = sorted(Path(chunk_dir).glob("*.npy"))
    n = 0
    for p in chunks:
        if n >= n_total:
            break
        arr = np.load(p)
        need = n_total - n
        if arr.shape[0] > need:
            arr = arr[:need]
        start = n
        n += arr.shape[0]
        yield arr, start, n


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=str, required=True,
                        help="Directory with .npy MQN fingerprint chunks")
    parser.add_argument("--encoder", type=str, default="models/paper_encoder.joblib")
    parser.add_argument("--n-fingerprints", type=int, default=100_000_000)
    parser.add_argument("--n-subsample", type=int, default=100_000_000)
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[10_000, 25_000, 50_000, 100_000, 200_000])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"n_fingerprints={args.n_fingerprints:,}, n_subsample={args.n_subsample:,}", flush=True)
    print(flush=True)

    encoder = PQEncoder.load(args.encoder)
    print(f"Encoder: m={encoder.m}, k={encoder.k}", flush=True)

    # Encode fingerprints to PQ codes (streamed from disk)
    print(f"Encoding {args.n_fingerprints:,} fingerprints on GPU...", flush=True)
    m = encoder.m
    pq_codes = np.empty((args.n_fingerprints, m), dtype=np.uint8)
    actual_end = 0
    t0 = time.perf_counter()
    for arr, start, end in iter_chunks(args.chunks, args.n_fingerprints):
        codes = encoder.transform(arr.astype(np.float32), verbose=0, device='gpu')
        pq_codes[start:end] = codes
        actual_end = end
        if end % 10_000_000 == 0:
            print(f"  {end:,}/{args.n_fingerprints:,}", flush=True)
    pq_codes = pq_codes[:actual_end]
    print(f"  Encoded {actual_end:,} in {fmt(time.perf_counter() - t0)}", flush=True)

    # Subsample if needed
    rng = np.random.default_rng(args.seed)
    n_total = pq_codes.shape[0]
    n_sub = min(args.n_subsample, n_total)
    if n_sub < n_total:
        idx = rng.choice(n_total, size=n_sub, replace=False)
        idx.sort()
        pq_codes = pq_codes[idx]
    print(f"  Using {pq_codes.shape[0]:,} PQ codes\n", flush=True)

    dtables = _build_distance_tables(encoder.codewords)

    # Sweep k values (all GPU)
    sep = "=" * 85
    print(f"{sep}", flush=True)
    print(f"  k-SELECTION: n={n_sub:,}, iters={args.iterations}, device=gpu", flush=True)
    print(f"{sep}\n", flush=True)

    print(f"{'k':>10}  {'Fit Time':>10}  {'Avg Dist':>10}  {'Empty%':>8}  "
          f"{'Med Size':>10}", flush=True)
    print("-" * 58, flush=True)

    results = []
    for k in sorted(args.k_values):
        if k >= n_sub:
            print(f"{k:>10,}  SKIPPED (k >= n)", flush=True)
            continue

        # Fit on GPU
        clusterer = PQKMeans(encoder, k=k, iteration=args.iterations, verbose=False)
        t0 = time.perf_counter()
        clusterer.fit(pq_codes, device='gpu')
        t_fit = time.perf_counter() - t0

        # Predict on GPU
        labels = clusterer.predict(pq_codes, device='gpu')

        # Compute metrics (cheap O(N*M) lookups, no K sweep)
        centers = clusterer.cluster_centers_
        avg_dist = compute_avg_distance(pq_codes, labels, centers, dtables)
        counts = np.bincount(labels.astype(np.intp), minlength=k)
        n_empty = int(np.sum(counts == 0))
        pct_empty = 100.0 * n_empty / k
        sizes = counts[counts > 0]
        med_size = float(np.median(sizes))

        print(f"{k:>10,}  {fmt(t_fit):>10}  {avg_dist:>10.2f}  {pct_empty:>7.1f}%  "
              f"{med_size:>10,.0f}", flush=True)

        results.append({
            "k": k, "fit_time": t_fit, "avg_dist": avg_dist,
            "pct_empty": pct_empty, "med_size": med_size,
        })

    # Summary table for README
    print(f"\n{sep}", flush=True)
    print("  README TABLE:", flush=True)
    print(f"{sep}\n", flush=True)
    print("| k | Avg Distance | Empty Clusters | Median Cluster Size | Fit Time (GPU) |", flush=True)
    print("|---:|---:|---:|---:|---:|", flush=True)
    for r in results:
        print(f"| {r['k']:,} | {r['avg_dist']:.2f} | {r['pct_empty']:.1f}% | "
              f"{r['med_size']:,.0f} | {fmt(r['fit_time'])} |", flush=True)

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
