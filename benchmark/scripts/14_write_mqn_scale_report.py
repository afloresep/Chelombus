#!/usr/bin/env python3
"""Assemble benchmark/REPORT_MQN_SCALE.md from the JSONs written by
benchmark/scripts/13_run_mqn_scale.py.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmark" / "results" / "mqn_scale"
COMBINED = RESULTS_DIR / "mqn_scale_summary.json"
REPORT = REPO_ROOT / "benchmark" / "REPORT_MQN_SCALE.md"

N_ORDER = ["100M", "250M", "500M", "1B"]
K_ORDER = ["20k", "50k", "100k"]
TRAIN_ORDER = ["10M", "30M", "50M", "100M"]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _fmt_seconds(s: float) -> str:
    if s >= 3600:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        return f"{h}h {m:02d}m"
    if s >= 60:
        m = int(s // 60)
        sec = int(s % 60)
        return f"{m}m {sec:02d}s"
    return f"{s:.1f}s"


def _encoder_bench_table(data: dict) -> str:
    lines = [
        "| N_train | fit wall | fit VRAM | RSS start | RSS peak | RSS delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for tag in TRAIN_ORDER:
        row = data.get(tag)
        if row is None:
            lines.append(f"| {tag} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {tag} "
            f"| {_fmt_seconds(row['fit_seconds'])} "
            f"| {row['fit_peak_vram_gb']:.2f} GB "
            f"| {row['rss_start_gb']:.2f} GB "
            f"| {row['rss_peak_gb']:.2f} GB "
            f"| {row['rss_delta_gb']:.2f} GB |"
        )
    return "\n".join(lines)


def _sanity_table(sanity: dict | None) -> str:
    if sanity is None:
        return "_(sanity test not yet run)_"
    a = sanity["fit_predict"]
    b = sanity["fit_then_predict"]
    return "\n".join([
        "| Path | wall | VRAM (peak) | RSS delta | iters |",
        "|---|---:|---:|---:|---:|",
        (
            f"| `fit_predict()` "
            f"| {_fmt_seconds(a['total_seconds'])} "
            f"| {a['peak_vram_gb']:.2f} GB "
            f"| {a['rss_delta_gb']:.2f} GB "
            f"| {a['iters_to_converge']} |"
        ),
        (
            f"| `fit()` + `predict()` "
            f"| {_fmt_seconds(b['total_seconds'])} "
            f"(fit {_fmt_seconds(b['fit_seconds'])} + "
            f"predict {_fmt_seconds(b['predict_seconds'])}) "
            f"| fit {b['fit_peak_vram_gb']:.2f} / "
            f"predict {b['predict_peak_vram_gb']:.2f} GB "
            f"| fit {b['fit_rss_delta_gb']:.2f} / "
            f"predict {b['predict_rss_delta_gb']:.2f} GB "
            f"| {b['iters_to_converge']} |"
        ),
    ])


def _main_grid_timing_table(grid: dict) -> str:
    header = ["| N \\ K |"] + [f" {k} |" for k in K_ORDER]
    sep = ["|---|"] + [":---:|" for _ in K_ORDER]
    lines = ["".join(header), "".join(sep)]
    for n_tag in N_ORDER:
        row = [f"| **{n_tag}** |"]
        for k_tag in K_ORDER:
            cell = grid.get(f"N{n_tag}_K{k_tag}")
            if cell is None:
                row.append(" — |")
                continue
            row.append(
                f" fit {_fmt_seconds(cell['fit_seconds'])} / "
                f"predict {_fmt_seconds(cell['predict_seconds'])}<br>"
                f"VRAM {cell['fit_peak_vram_gb']:.2f}/"
                f"{cell['predict_peak_vram_gb']:.2f} GB<br>"
                f"RSS peak {cell['rss_peak_gb']:.2f} GB "
                f"(Δ {cell['rss_delta_gb']:.2f})<br>"
                f"iters {cell['iters_to_converge']} |"
            )
        lines.append("".join(row))
    return "\n".join(lines)


def _cluster_shape_table(grid: dict) -> str:
    header = (
        "| N | K | nonempty | empty | singletons | >10 | >100 | largest | "
        "largest% | mean | median | Q1 | Q3 |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for n_tag in N_ORDER:
        for k_tag in K_ORDER:
            cell = grid.get(f"N{n_tag}_K{k_tag}")
            if cell is None:
                lines.append(
                    f"| {n_tag} | {k_tag} | — | — | — | — | — | — | — | — | — | — | — |"
                )
                continue
            lines.append(
                f"| {n_tag} | {k_tag} "
                f"| {cell['num_clusters_nonempty']} "
                f"| {cell['num_empty']} "
                f"| {cell['num_singletons']} "
                f"| {cell['num_clusters_gt10']} "
                f"| {cell['num_clusters_gt100']} "
                f"| {cell['largest_cluster']} "
                f"| {cell['largest_cluster_pct_of_total']}% "
                f"| {cell['mean_size_nonempty']} "
                f"| {cell['median_size_nonempty']} "
                f"| {cell['p25_size_nonempty']} "
                f"| {cell['p75_size_nonempty']} |"
            )
    return "\n".join(lines)


def main() -> None:
    if not COMBINED.exists():
        raise FileNotFoundError(
            f"{COMBINED} not found. Run benchmark/scripts/13_run_mqn_scale.py first."
        )
    data = json.loads(COMBINED.read_text())

    md = []
    md.append("# Chelombus MQN scaling benchmark\n")
    md.append(f"Date: {time.strftime('%Y-%m-%d')}")
    md.append(f"Repo commit: `{_git_commit()}`")
    md.append(
        "\nSee `docs/superpowers/specs/2026-04-15-chelombus-mqn-scale-benchmark-design.md` "
        "for the design and methodology."
    )

    md.append("\n## Encoder training scaling\n")
    md.append(
        "`PQEncoder(k=256, m=6, iterations=20).fit()` on MQN int16 "
        "(42-dim), device=auto (GPU). One row per training-set size."
    )
    md.append("")
    md.append(_encoder_bench_table(data.get("encoder_bench", {})))
    md.append(
        "\n**Note on RSS at N_train=100M:** `PQEncoder._fit_gpu` casts "
        "the full training matrix to float32 in one shot "
        "(`chelombus/encoder/encoder.py:157`), so peak RSS briefly holds "
        "both the int16 persistent buffer and the float32 copy. At 100M "
        "that's ~8.4 GB + ~16.8 GB ≈ 25 GB. This is a transient "
        "encoder-internal allocation, not something a downstream user "
        "would see from the outside."
    )

    md.append("\n## Sanity test: `fit_predict()` vs `fit()` + `predict()`\n")
    md.append(
        "Same PQ codes, same K, independent `PQKMeans` instances. "
        "N = 100M, K = 20k."
    )
    md.append("")
    md.append(_sanity_table(data.get("sanity")))

    md.append("\n## Main grid — timing & memory\n")
    md.append(
        "Every cell is a `PQKMeans(encoder_50M, k=K, iteration=20, tol=1e-3)` "
        "run: `fit()` then `predict()`, both on GPU. RSS is sampled via "
        "`/proc/self/statm` during the fit+predict span."
    )
    md.append("")
    md.append(_main_grid_timing_table(data.get("main_grid", {})))

    md.append("\n## Main grid — cluster shape\n")
    md.append(_cluster_shape_table(data.get("main_grid", {})))

    md.append("")
    REPORT.write_text("\n".join(md))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
