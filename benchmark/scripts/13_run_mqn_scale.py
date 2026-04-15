#!/usr/bin/env python3
"""MQN scaling benchmark for Chelombus PQ-KMeans.

Stages:
  A. Encoder sub-benchmark (train PQEncoder at N_train = 10M/30M/50M/100M)
  B. Streaming encode (PREP — not timed for the main report)
  C. Sanity test: fit_predict vs fit+predict at N=100M, K=20k
  D. Main grid: (N, K) in [100M, 250M, 500M, 1B] x [20k, 50k, 100k]

All stages are resumable via per-artifact existence checks. Re-running
the script skips any work whose output already exists on disk.

Run:
    python benchmark/scripts/13_run_mqn_scale.py 2>&1 | tee -a benchmark/logs/13_run_mqn_scale.log

See docs/superpowers/specs/2026-04-15-chelombus-mqn-scale-benchmark-design.md
for the full design.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chelombus.encoder.encoder import PQEncoder  # noqa: E402
from chelombus.clustering.PyQKmeans import PQKMeans  # noqa: E402

from _mqn_scale_utils import (  # noqa: E402
    RSSSampler,
    fmt_time,
    gpu_peak_vram_gb,
    log,
    reset_gpu_vram,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_DIR = Path("/mnt/samsung_2tb/tmp")
CHUNK_GLOB = "enamine-chunk_{:05d}.npy"
CHUNK_SIZE = 10_000_000  # rows per chunk
WORK_DIR = Path("/mnt/10tb_hdd/chelombus_mqn_bench")
LABELS_DIR = WORK_DIR / "labels"
RESULTS_DIR = REPO_ROOT / "benchmark" / "results" / "mqn_scale"
ENCODER_BENCH_DIR = RESULTS_DIR / "encoder_bench"

PQ_K = 256
PQ_M = 6
PQ_ITERATIONS = 20

ENCODER_TRAIN_SIZES = [10_000_000, 30_000_000, 50_000_000, 100_000_000]
DOWNSTREAM_ENCODER_N = 50_000_000  # the one reported in the paper

PQ_SLICES = [
    (100_000_000, "100M"),
    (250_000_000, "250M"),
    (500_000_000, "500M"),
    (1_000_000_000, "1B"),
]

K_VALUES = [(20_000, "20k"), (50_000, "50k"), (100_000, "100k")]

SANITY_N = 100_000_000
SANITY_N_TAG = "100M"
SANITY_K = 20_000


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _n_tag(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}B"
    return f"{n // 1_000_000}M"


def chunk_path(i: int) -> Path:
    return CHUNK_DIR / CHUNK_GLOB.format(i)


def load_chunks(n_rows: int) -> np.ndarray:
    """Load the first n_rows of MQN as a single contiguous int16 ndarray.

    n_rows must be a multiple of CHUNK_SIZE. Raises if any chunk is missing.
    """
    if n_rows % CHUNK_SIZE != 0:
        raise ValueError(f"n_rows={n_rows} must be a multiple of {CHUNK_SIZE}")
    n_chunks = n_rows // CHUNK_SIZE
    parts: list[np.ndarray] = []
    for i in range(n_chunks):
        p = chunk_path(i)
        if not p.exists():
            raise FileNotFoundError(p)
        parts.append(np.load(p))
    arr = np.concatenate(parts, axis=0)
    del parts
    gc.collect()
    return arr


def encoder_path(n_train: int) -> Path:
    tag = _n_tag(n_train)
    return WORK_DIR / f"encoder_N{tag}_k{PQ_K}_m{PQ_M}.joblib"


def pq_codes_path(N_tag: str) -> Path:
    return WORK_DIR / f"pq_codes_{N_tag}.npy"


def labels_path(N_tag: str, K_tag: str) -> Path:
    return LABELS_DIR / f"N{N_tag}_K{K_tag}.npy"


def _ensure_dirs() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    ENCODER_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage A: encoder sub-benchmark
# ---------------------------------------------------------------------------

def stage_a_encoder_bench() -> None:
    log("=" * 72)
    log("Stage A — encoder sub-benchmark")
    log("=" * 72)

    for n_train in ENCODER_TRAIN_SIZES:
        n_tag = _n_tag(n_train)
        bench_path = ENCODER_BENCH_DIR / f"N{n_tag}.json"
        enc_out = encoder_path(n_train)

        if bench_path.exists() and enc_out.exists():
            log(f"[Stage A] N_train={n_tag}: SKIP (already done)")
            continue

        log(f"[Stage A] N_train={n_tag}: loading {n_train // CHUNK_SIZE} chunks")
        X_train = load_chunks(n_train)
        assert X_train.shape == (n_train, 42)
        assert X_train.dtype == np.int16
        log(f"  X_train shape={X_train.shape} dtype={X_train.dtype} "
            f"size={X_train.nbytes / 1e9:.2f} GB")

        reset_gpu_vram()
        enc = PQEncoder(k=PQ_K, m=PQ_M, iterations=PQ_ITERATIONS)
        t0 = time.perf_counter()
        with RSSSampler() as rss:
            enc.fit(X_train, verbose=0, device="auto")
        fit_time = time.perf_counter() - t0
        fit_vram = gpu_peak_vram_gb()
        log(f"  fit time={fmt_time(fit_time)}  "
            f"rss_peak={rss.peak_gb:.2f} GB  vram={fit_vram:.2f} GB")

        enc.save(enc_out)
        log(f"  encoder saved to {enc_out}")

        summary = {
            "stage": "A",
            "n_train": n_train,
            "n_train_tag": n_tag,
            "pq_k": PQ_K,
            "pq_m": PQ_M,
            "pq_iterations": PQ_ITERATIONS,
            "fit_seconds": round(fit_time, 2),
            "fit_peak_vram_gb": round(fit_vram, 3),
            "rss_start_gb": round(rss.start_gb, 3),
            "rss_peak_gb": round(rss.peak_gb, 3),
            "rss_delta_gb": round(rss.delta_gb, 3),
        }
        bench_path.write_text(json.dumps(summary, indent=2))
        log(f"  summary → {bench_path}")

        del X_train, enc
        gc.collect()
        reset_gpu_vram()
        log("")


# ---------------------------------------------------------------------------
# Stage B: streaming encode (PREP — not in the main report tables)
# ---------------------------------------------------------------------------

def _all_pq_files_exist() -> bool:
    return all(pq_codes_path(tag).exists() for _, tag in PQ_SLICES)


def stage_b_streaming_encode() -> None:
    log("=" * 72)
    log("Stage B — streaming encode (PREP)")
    log("=" * 72)

    if _all_pq_files_exist():
        log("[Stage B] SKIP (all pq_codes_*.npy already on disk)")
        return

    downstream_enc_path = encoder_path(DOWNSTREAM_ENCODER_N)
    if not downstream_enc_path.exists():
        raise FileNotFoundError(
            f"Downstream encoder missing at {downstream_enc_path}. "
            f"Run Stage A first (at least up to N_train={DOWNSTREAM_ENCODER_N})."
        )

    log(f"[Stage B] loading downstream encoder {downstream_enc_path}")
    enc = PQEncoder.load(downstream_enc_path)
    assert enc.is_trained

    total_rows = PQ_SLICES[-1][0]  # 1B
    n_chunks = total_rows // CHUNK_SIZE

    log(f"[Stage B] allocating pq_full shape=({total_rows:,}, {PQ_M}) uint8 "
        f"= {total_rows * PQ_M / 1e9:.1f} GB")
    pq_full = np.empty((total_rows, PQ_M), dtype=np.uint8)

    t0 = time.perf_counter()
    for i in range(n_chunks):
        p = chunk_path(i)
        if not p.exists():
            raise FileNotFoundError(p)
        mqn = np.load(p)
        assert mqn.shape == (CHUNK_SIZE, 42)
        codes = enc.transform(mqn, verbose=0, device="auto")
        start = i * CHUNK_SIZE
        pq_full[start:start + CHUNK_SIZE] = codes
        del mqn, codes
        if (i + 1) % 10 == 0 or i == n_chunks - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) * CHUNK_SIZE / elapsed
            log(f"  encoded chunk {i + 1}/{n_chunks}  "
                f"elapsed={fmt_time(elapsed)}  rate={rate / 1e6:.1f} M/s")
    total_elapsed = time.perf_counter() - t0
    log(f"[Stage B] stream-encode complete in {fmt_time(total_elapsed)}")

    for n_rows, tag in PQ_SLICES:
        out = pq_codes_path(tag)
        if out.exists():
            log(f"  [skip] {out} already exists")
            continue
        log(f"  saving {out} ({n_rows * PQ_M / 1e9:.2f} GB)")
        np.save(out, pq_full[:n_rows])

    del pq_full, enc
    gc.collect()
    log("")


def main() -> None:
    _ensure_dirs()
    import torch
    log(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU={torch.cuda.get_device_name(0)}  "
            f"VRAM_total={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    stage_a_encoder_bench()
    stage_b_streaming_encode()


if __name__ == "__main__":
    main()
