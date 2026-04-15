"""Unit tests for benchmark/scripts/_mqn_scale_utils.py."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from _mqn_scale_utils import (  # noqa: E402
    RSSSampler,
    cluster_stats,
    parse_iters_from_log,
)


def test_rss_sampler_detects_allocation_peak():
    """RSSSampler should observe a peak above start_gb when we allocate
    a chunk of memory inside its context."""
    with RSSSampler(interval_s=0.01) as rss:
        buf = np.zeros(200 * 1024 * 1024, dtype=np.uint8)
        buf[::4096] = 1  # touch pages so they become resident
        time.sleep(0.1)
        del buf

    assert rss.peak_gb > rss.start_gb
    assert rss.delta_gb > 0.1  # at least ~100 MB delta observed


def test_parse_iters_from_log_returns_highest():
    log_text = (
        "Some preamble\n"
        "  iter 1/20  assign=2.1s  update=0.3s  changed=60000/120000 (50.00%)\n"
        "  iter 2/20  assign=2.0s  update=0.3s  changed=40000/120000 (33.33%)\n"
        "  iter 3/20  assign=2.0s  update=0.3s  changed=100/120000 (0.08%)\n"
        "  Stopped at iteration 3 (converged)\n"
    )
    assert parse_iters_from_log(log_text) == 3


def test_parse_iters_from_log_empty():
    assert parse_iters_from_log("") == 0
    assert parse_iters_from_log("no iter lines here") == 0


def test_cluster_stats_basic():
    labels = np.array([0] * 7 + [1] * 3, dtype=np.int32)
    stats = cluster_stats(labels, K=10)

    assert stats["num_clusters_nonempty"] == 2
    assert stats["num_empty"] == 8
    assert stats["num_singletons"] == 0
    assert stats["largest_cluster"] == 7
    assert stats["num_clusters_gt10"] == 0
    assert stats["median_size_nonempty"] == 5  # median of [7, 3]
