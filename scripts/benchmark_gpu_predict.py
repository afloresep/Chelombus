"""Benchmark: GPU vs CPU for PQ Transform and Predict.

Pre-computes fingerprints to data/20M_fingerprints.npy so they can be
reloaded without recomputing. Fingerprint time is NOT included in the report.

Usage:
    python scripts/benchmark_gpu_predict.py [--n-points N] [--runs R]
    python scripts/benchmark_gpu_predict.py                    # full 20M
    python scripts/benchmark_gpu_predict.py --n-points 1000000 # quick test
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chelombus import PQEncoder, FingerprintCalculator
from chelombus.clustering.PyQKmeans import PQKMeans
from chelombus.utils import format_time


FP_CACHE = Path("data/20M_fingerprints.npy")


def load_fingerprints(n_points: int) -> np.ndarray:
    """Load pre-computed fingerprints, or compute and cache them."""
    if FP_CACHE.exists():
        print(f"Loading cached fingerprints from {FP_CACHE}...")
        fps = np.load(FP_CACHE)
        if n_points < len(fps):
            fps = fps[:n_points]
        print(f"  shape={fps.shape}, dtype={fps.dtype}")
        return fps

    # Compute and save — try multiple SMILES sources
    smiles_path = None
    for candidate in [
        Path("data/20M_smiles.txt"),
        Path("data/10M_smiles.txt"),
    ]:
        if candidate.exists():
            smiles_path = candidate
            break
    # Try decompressing a gzipped version
    if smiles_path is None:
        for gz_candidate in [
            Path("data/20M_smiles.txt.gz"),
            Path("data/10M_smiles.txt.gz"),
        ]:
            if gz_candidate.exists():
                smiles_path = gz_candidate.with_suffix("").with_suffix(".txt")
                print(f"Decompressing {gz_candidate}...")
                import gzip, shutil
                with gzip.open(gz_candidate, "rt") as gz, open(smiles_path, "w") as out:
                    shutil.copyfileobj(gz, out)
                break
    if smiles_path is None or not smiles_path.exists():
        raise FileNotFoundError(
            "No SMILES file found in data/. Expected 10M_smiles.txt(.gz) or 20M_smiles.txt(.gz)."
        )
    print(f"No cached fingerprints found. Computing from {smiles_path}...")
    smiles = []
    with open(smiles_path) as f:
        for line in f:
            smiles.append(line.strip())
    fp_calc = FingerprintCalculator()
    fps = fp_calc.FingerprintFromSmiles(smiles, "mqn")
    np.save(FP_CACHE, fps)
    print(f"  Saved to {FP_CACHE} ({FP_CACHE.stat().st_size / 1024**2:.1f} MB)")
    if n_points < len(fps):
        fps = fps[:n_points]
    return fps


def timed_runs(fn, n_runs, label):
    """Run fn() n_runs times, print each run, return median time."""
    times = []
    result = None
    for r in range(n_runs):
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if result is None:
            result = out
        print(f"  {label} run {r+1}: {elapsed:.4f}s")
    return np.median(times), result


def benchmark_transform(encoder, fps, n_warmup, n_runs):
    N = fps.shape[0]
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"TRANSFORM: {N:,} fingerprints → PQ codes (m={encoder.m}, k={encoder.k})")
    print(sep)

    results = {}
    for device in ["gpu", "cpu"]:
        tag = device.upper()

        # Warmup
        for _ in range(n_warmup):
            encoder.transform(fps[:1000], verbose=0, device=device)

        med, codes = timed_runs(
            lambda d=device: encoder.transform(fps, verbose=0, device=d),
            n_runs, tag,
        )
        throughput = N / med
        results[device] = {"median": med, "throughput": throughput, "codes": codes}
        print(f"  → {tag} median: {med:.3f}s  ({throughput:,.0f} mol/s)")

    gpu, cpu = results["gpu"], results["cpu"]
    mismatches = int(np.sum(gpu["codes"] != cpu["codes"]))

    print(f"\n  Speedup: {cpu['median']/gpu['median']:.1f}x")
    print(f"  Correctness: {mismatches}/{gpu['codes'].size} mismatches "
          f"({100*mismatches/gpu['codes'].size:.3f}% — float32 tie-breaking)")
    print(f"  Extrapolation:")
    for label, total in [("1B", 1e9), ("9.6B", 9.6e9)]:
        print(f"    {label}: GPU {format_time(total/gpu['throughput'])} | CPU {format_time(total/cpu['throughput'])}")

    return gpu["codes"]


def benchmark_predict(clusterer, pq_codes, n_warmup, n_runs):
    N = pq_codes.shape[0]
    K = clusterer.k
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"PREDICT: {N:,} PQ codes → cluster labels (K={K:,})")
    print(sep)

    results = {}
    for device in ["gpu", "cpu"]:
        tag = device.upper()
        # CPU with K=100K is very slow; limit subset
        n_bench = N 
        bench_codes = pq_codes[:n_bench]

        # Warmup
        for _ in range(n_warmup):
            clusterer.predict(bench_codes[:1000], device=device)

        med, labels = timed_runs(
            lambda d=device, bc=bench_codes: clusterer.predict(bc, device=d),
            n_runs, tag,
        )
        throughput = n_bench / med
        results[device] = {
            "median": med, "throughput": throughput, "labels": labels, "n": n_bench,
        }
        print(f"  → {tag} median: {med:.3f}s for {n_bench:,} points  ({throughput:,.0f} codes/sec)")

    gpu, cpu = results["gpu"], results["cpu"]
    overlap = min(gpu["n"], cpu["n"])
    match = int(np.sum(gpu["labels"][:overlap] == cpu["labels"][:overlap]))

    print(f"\n  Speedup: {gpu['throughput']/cpu['throughput']:.1f}x")
    print(f"  Correctness: {match:,}/{overlap:,} match ({100*match/overlap:.2f}%)")
    print(f"  Extrapolation:")
    for label, total in [("1B", 1e9), ("9.6B", 9.6e9)]:
        print(f"    {label}: GPU {format_time(total/gpu['throughput'])} | CPU {format_time(total/cpu['throughput'])}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU: Transform + Predict")
    parser.add_argument("--n-points", type=int, default=0,
                        help="Number of fingerprints (0 = use all cached, default: all 20M)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--encoder", default="models/paper_encoder.joblib")
    parser.add_argument("--clusterer", default="models/paper_clusterer.joblib")
    args = parser.parse_args()

    print("Loading models...")
    encoder = PQEncoder.load(args.encoder)
    clusterer = PQKMeans.load(args.clusterer)
    print(f"  Encoder: m={encoder.m}, k={encoder.k}")
    print(f"  Clusterer: K={clusterer.k:,}")

    n = args.n_points if args.n_points > 0 else 20_000_001
    fps = load_fingerprints(n)

    pq_codes = benchmark_transform(encoder, fps, args.warmup, args.runs)
    benchmark_predict(clusterer, pq_codes, args.warmup, args.runs)


if __name__ == "__main__":
    main()
