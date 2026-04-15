"""Shared helpers for the MQN scaling benchmark (script 13).

See docs/superpowers/specs/2026-04-15-chelombus-mqn-scale-benchmark-design.md
for the rationale on RSS sampling vs. ru_maxrss.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Timestamped logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_time(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {sec:.1f}s"


# ---------------------------------------------------------------------------
# RSS sampler — polls /proc/self/statm in a daemon thread
# ---------------------------------------------------------------------------

class RSSSampler:
    """Context manager that samples resident set size in a background thread.

    Usage:
        with RSSSampler(interval_s=0.1) as rss:
            do_work()
        print(rss.start_gb, rss.peak_gb, rss.delta_gb)

    resource.getrusage().ru_maxrss is monotone for the lifetime of the
    process and cannot be reset, so it is useless for per-cell attribution
    in a long-running benchmark script. /proc/self/statm reports the
    current resident set, which we poll and max over the sampling window.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self._page_size = os.sysconf("SC_PAGESIZE")
        self._start_bytes = 0
        self._max_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_rss_bytes(self) -> int:
        with open("/proc/self/statm", "rb") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * self._page_size

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._read_rss_bytes()
            if rss > self._max_bytes:
                self._max_bytes = rss
            if self._stop.wait(self.interval_s):
                return

    def __enter__(self) -> "RSSSampler":
        self._start_bytes = self._read_rss_bytes()
        self._max_bytes = self._start_bytes
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        final = self._read_rss_bytes()
        if final > self._max_bytes:
            self._max_bytes = final

    @property
    def start_gb(self) -> float:
        return self._start_bytes / 1024**3

    @property
    def peak_gb(self) -> float:
        return self._max_bytes / 1024**3

    @property
    def delta_gb(self) -> float:
        return (self._max_bytes - self._start_bytes) / 1024**3


# ---------------------------------------------------------------------------
# GPU / VRAM helpers
# ---------------------------------------------------------------------------

def reset_gpu_vram() -> None:
    import torch
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def gpu_peak_vram_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / 1024**3


# ---------------------------------------------------------------------------
# Iteration counter: parse "iter N/20 ..." lines out of PQKMeans verbose stdout
# ---------------------------------------------------------------------------

_ITER_LINE_RE = re.compile(r"iter\s+(\d+)\s*/\s*\d+", re.IGNORECASE)


def parse_iters_from_log(text: str) -> int:
    """Return the highest iteration number printed in a PQKMeans verbose log.

    Returns 0 if no iter lines are found.
    """
    best = 0
    for match in _ITER_LINE_RE.finditer(text):
        n = int(match.group(1))
        if n > best:
            best = n
    return best


# ---------------------------------------------------------------------------
# Cluster shape statistics
# ---------------------------------------------------------------------------

def cluster_stats(labels: np.ndarray, K: int) -> dict[str, Any]:
    """Compute the cluster-shape fields used in the per-run summary JSON.

    Args:
        labels: (N,) integer cluster assignments in [0, K).
        K: declared number of clusters (may exceed observed clusters if
            some were empty).
    """
    sizes = np.bincount(labels, minlength=K)
    nonempty = sizes[sizes > 0]
    n_singletons = int((sizes == 1).sum())
    n_empty = int((sizes == 0).sum())
    n_gt10 = int((sizes > 10).sum())
    n_gt100 = int((sizes > 100).sum())
    largest = int(sizes.max())
    total = int(len(labels))

    return {
        "num_clusters_nonempty": int((sizes > 0).sum()),
        "num_empty": n_empty,
        "num_singletons": n_singletons,
        "pct_singletons": round(n_singletons / K * 100, 4) if K else 0.0,
        "num_clusters_gt10": n_gt10,
        "num_clusters_gt100": n_gt100,
        "largest_cluster": largest,
        "largest_cluster_pct_of_total": round(largest / total * 100, 4) if total else 0.0,
        "mean_size_nonempty": round(float(nonempty.mean()), 2) if nonempty.size else 0.0,
        "median_size_nonempty": int(np.median(nonempty)) if nonempty.size else 0,
        "p25_size_nonempty": int(np.percentile(nonempty, 25)) if nonempty.size else 0,
        "p75_size_nonempty": int(np.percentile(nonempty, 75)) if nonempty.size else 0,
    }
