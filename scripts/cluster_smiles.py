"""Cluster SMILES using pre-trained PQEncoder + PQKMeans models.

Usage:
    python scripts/cluster_smiles.py \
        --input /mnt/10tb_hdd/cleaned_enamine_10b/output_file_0.cxsmiles \
        --output /mnt/samsung_2tb/tmp/ \
        --smiles-col 1 \
        --chunksize 1000000

Uses GPU for cluster assignment when available (device='auto').
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chelombus import PQEncoder, DataStreamer, FingerprintCalculator
from chelombus.clustering.PyQKmeans import PQKMeans


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Cluster SMILES with pre-trained models")
    parser.add_argument("--input", required=True, help="Input SMILES file")
    parser.add_argument("--output", required=True, help="Output directory for parquet files")
    parser.add_argument("--encoder", default="models/paper_encoder.joblib")
    parser.add_argument("--clusterer", default="models/paper_clusterer.joblib")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--smiles-col", type=int, default=0,
                        help="Column index for SMILES (0-indexed)")
    parser.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--resume", action="store_true",
                        help="Skip chunks that already have output files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading encoder from {args.encoder}")
    encoder = PQEncoder.load(args.encoder)
    print(f"Loading clusterer from {args.clusterer}")
    clusterer = PQKMeans.load(args.clusterer)
    print(f"  k={clusterer.k:,} clusters, m={encoder.m} subvectors")

    stream = DataStreamer()
    fp_calc = FingerprintCalculator()

    total_molecules = 0
    total_time = 0
    start = time.perf_counter()

    for i, chunk in enumerate(stream.parse_input(
        args.input, chunksize=args.chunksize,
        smiles_col=args.smiles_col, verbose=0,
    )):
        # Resume support: skip if output file exists
        out_file = os.path.join(args.output, f"chunk_{i:05d}.parquet")
        if args.resume and os.path.exists(out_file):
            total_molecules += len(chunk)
            continue

        t0 = time.perf_counter()

        # SMILES to MQN fingerprints
        fps = fp_calc.FingerprintFromSmiles(chunk, "mqn")

        # MQN to PQ codes (GPU when available)
        pq_codes = encoder.transform(fps, verbose=0, device=args.device)

        # PQ codes to cluster labels (GPU when available)
        labels = clusterer.predict(pq_codes, device=args.device)

        # Build output only include rows where fingerprint succeeded
        if len(fps) == len(chunk):
            table = pa.table({"smiles": chunk, "cluster_id": labels})
        else:
            table = pa.table({"cluster_id": labels})

        pq.write_table(table, out_file)

        elapsed = time.perf_counter() - t0
        total_molecules += len(chunk)
        total_time += elapsed
        rate = total_molecules / (time.perf_counter() - start)

        print(
            f"\rChunk {i:>5d} | {total_molecules:>12,} molecules | "
            f"{rate:,.0f} mol/s | chunk: {elapsed:.1f}s | "
            f"ETA: {format_time((9_600_000_000 - total_molecules) / rate) if rate > 0 else '?'}",
            end="", flush=True,
        )

        del chunk, fps, pq_codes, labels, table

    elapsed_total = time.perf_counter() - start
    print(f"\n\nDone: {total_molecules:,} molecules in {format_time(elapsed_total)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
