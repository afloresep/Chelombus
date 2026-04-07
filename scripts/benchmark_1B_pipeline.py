"""Benchmark the full GPU pipeline on real MQN fingerprints.

Streams fingerprint chunks from disk (never loading all into RAM at once),
runs all 4 pipeline steps on GPU, and reports wall-clock times.

Input: a directory of .npy files, each containing an (N_chunk, 42) array
of MQN fingerprints (any integer dtype). Files are loaded in sorted order
until --n fingerprints have been consumed.

Usage:
    # Full 1B benchmark
    python scripts/benchmark_1B_pipeline.py --chunks /path/to/mqn_chunks --n 1000000000

    # Quick test on 100M
    python scripts/benchmark_1B_pipeline.py --chunks /path/to/mqn_chunks --n 100000000

    # Overnight (recommended)
    nohup python -u scripts/benchmark_1B_pipeline.py --chunks /path/to/mqn_chunks > benchmark_results.txt 2>&1 &
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chelombus import PQEncoder
from chelombus.clustering.PyQKmeans import PQKMeans


def fmt(secs):
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        return f"{secs/60:.1f} min"
    if secs < 86400:
        return f"{secs/3600:.1f} hrs"
    return f"{secs/86400:.1f} days"


def iter_chunks(chunk_dir, n_total):
    """Yield (chunk_array, start_idx, end_idx) from .npy files on disk."""
    chunks = sorted(Path(chunk_dir).glob("*.npy"))
    if not chunks:
        raise FileNotFoundError(f"No .npy files found in {chunk_dir}")
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


def load_training_sample(chunk_dir, n_train):
    """Load first n_train fingerprints for encoder training."""
    loaded = []
    n = 0
    for arr, _, end in iter_chunks(chunk_dir, n_train):
        loaded.append(arr)
        n = end
    return np.vstack(loaded)[:n_train]


def encode_chunked(encoder, chunk_dir, n_total):
    """Encode fingerprints in streaming chunks on GPU, return PQ codes."""
    m = encoder.m
    pq_codes = np.empty((n_total, m), dtype=np.uint8)
    t0 = time.perf_counter()
    actual_end = 0

    for arr, start, end in iter_chunks(chunk_dir, n_total):
        codes = encoder.transform(arr.astype(np.float32), verbose=0, device='gpu')
        pq_codes[start:end] = codes
        actual_end = end
        elapsed = time.perf_counter() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta = (n_total - end) / rate if rate > 0 else 0
        print(f"    {end:>13,}/{n_total:,} ({end/n_total*100:5.1f}%)  "
              f"rate={rate:,.0f} pts/s  ETA={fmt(eta)}", flush=True)

    total = time.perf_counter() - t0
    return pq_codes[:actual_end], total


def bench(label, fn):
    print(f"  {label}...", end=" ", flush=True)
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    print(f"{elapsed:.2f}s ({fmt(elapsed)})", flush=True)
    return elapsed, result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chunks", type=str, required=True,
                        help="Directory containing .npy MQN fingerprint chunks (N, 42)")
    parser.add_argument("--n", type=int, default=1_000_000_000,
                        help="Number of fingerprints to use (default: 1B)")
    parser.add_argument("--k", type=int, default=100_000,
                        help="Number of clusters (default: 100K)")
    parser.add_argument("--fit-iters", type=int, default=5,
                        help="Cluster training iterations (default: 5)")
    parser.add_argument("--train-sample", type=int, default=50_000_000,
                        help="Encoder training sample size (default: 50M)")
    args = parser.parse_args()

    import torch
    gpu_name = torch.cuda.get_device_name(0)
    free_mb = torch.cuda.mem_get_info()[0] / 1024**2
    print(f"GPU: {gpu_name} ({free_mb:.0f} MB free)", flush=True)
    print(f"N={args.n:,}  K={args.k:,}  fit_iters={args.fit_iters}", flush=True)
    print(f"Chunks: {args.chunks}", flush=True)
    print(flush=True)

    sep = "=" * 70
    results = {}

    # ── 1. Encoder training ──────────────────────────────────────────────
    print(f"{sep}\n  STEP 1: Encoder training ({args.train_sample:,} sample)\n{sep}", flush=True)

    print("  Loading training sample...", flush=True)
    train_sample = load_training_sample(args.chunks, args.train_sample).astype(np.float32)
    print(f"  Sample shape: {train_sample.shape}", flush=True)

    encoder = PQEncoder(k=256, m=6, iterations=20)
    t_enc, _ = bench("GPU", lambda: encoder.fit(train_sample, verbose=1, device='gpu'))
    results["Encoder training"] = t_enc
    del train_sample

    # ── 2. PQ encoding ───────────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 2: PQ encoding ({args.n:,} fingerprints, streamed)\n{sep}", flush=True)

    # Warmup
    dummy = np.random.randint(0, 255, (1000, 42), dtype=np.int16)
    encoder.transform(dummy.astype(np.float32), verbose=0, device='gpu')
    del dummy

    pq_codes, t_encode = encode_chunked(encoder, args.chunks, args.n)
    N = pq_codes.shape[0]
    print(f"  Total: {fmt(t_encode)} ({N/t_encode:,.0f} pts/s)", flush=True)
    results["PQ encoding"] = t_encode

    print(f"  PQ codes: {pq_codes.shape} = {pq_codes.nbytes / 1e9:.1f} GB in RAM", flush=True)

    # ── 3. Cluster training ──────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 3: Cluster training ({N:,}, K={args.k:,}, {args.fit_iters} iters, tol=0)\n{sep}", flush=True)

    clusterer = PQKMeans(encoder, k=args.k, iteration=args.fit_iters, tol=0, verbose=True)
    t_fit, _ = bench("GPU", lambda: clusterer.fit(pq_codes, device='gpu'))
    results["Cluster training"] = t_fit

    # ── 4. Label assignment ──────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 4: Label assignment ({N:,}, K={args.k:,})\n{sep}", flush=True)

    clusterer.predict(pq_codes[:10000], device='gpu')  # warmup

    t_pred, labels = bench("GPU", lambda: clusterer.predict(pq_codes, device='gpu'))
    results["Label assignment"] = t_pred

    n_unique = len(np.unique(labels))
    print(f"  Unique clusters: {n_unique:,} / {args.k:,}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────
    total = sum(results.values())
    print(f"\n{sep}\n  GPU RESULTS AT {N:,}\n{sep}", flush=True)
    print(f"  {'Stage':<35} {'Time':>12}", flush=True)
    print(f"  {'-'*49}", flush=True)
    for stage, t in results.items():
        print(f"  {stage:<35} {fmt(t):>12}", flush=True)
    print(f"  {'-'*49}", flush=True)
    print(f"  {'TOTAL':<35} {fmt(total):>12}", flush=True)

    print(f"\n  Note: Cluster training uses tol=0 (no early stopping).", flush=True)
    print(f"  With default tol=1e-3, training typically converges in fewer", flush=True)
    print(f"  iterations, reducing cluster training time further.", flush=True)
    print(f"\n  Done.", flush=True)


if __name__ == "__main__":
    main()
