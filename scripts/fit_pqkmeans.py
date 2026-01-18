"""CLI utility to fit PQKMeans models from pre-computed PQ codes.

Example
-------
python scripts/fit_pqkmeans.py \
    --pq-codes data/pq_codes.npy \
    --encoder scripts/pq_trained_model_300iterations.pkl \
    --k 10000 --iterations 20
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from chelombus.utils.helper_functions import format_time, _process_input
except ModuleNotFoundError as exc:
    if exc.name != "pqkmeans":
        raise
    helper_spec = importlib.util.spec_from_file_location(
        "spiq.utils.helper_functions",
        REPO_ROOT / "spiq" / "utils" / "helper_functions.py")
    if helper_spec is None or helper_spec.loader is None:
        raise ImportError("Could not load spiq.utils.helper_functions.") from exc
    helper_module = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper_module)
    format_time = helper_module.format_time  # type: ignore[attr-defined]
    _process_input = helper_module._process_input  # type: ignore[attr-defined]


SUPPORTED_EXTENSIONS = {".npy", ".npz", ".parquet", ".csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit PQKMeans on previously generated PQ codes.")
    parser.add_argument(
        "--pq-codes",
        nargs="+",
        required=True,
        help="File(s) or directories containing PQ codes. Supported formats: "
             "npy, npz, parquet, csv.")
    parser.add_argument(
        "--encoder",
        required=True,
        help="Path to the serialized PQEncoder (joblib dump).")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of clusters to fit.")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of iterations for PQKMeans (default: 20).")
    parser.add_argument(
        "--pq-columns",
        nargs="+",
        help="Explicit list of PQ-code column names when loading tabular files."
             " Overrides automatic prefix-based discovery.")
    parser.add_argument(
        "--pq-prefix",
        default="pq_code_",
        help="Column prefix that identifies PQ-code columns in parquet/csv files "
             "(default: pq_code_). Ignored when --pq-columns is passed.")
    parser.add_argument(
        "--npz-key",
        default=None,
        help="Array key to load from npz files. Required when multiple arrays are "
             "stored per npz file.")
    parser.add_argument(
        "--csv-delimiter",
        default=",",
        help="Delimiter used for CSV file inputs (default: ',').")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a tqdm progress bar while reading multiple input files.")
    parser.add_argument(
        "--kmeans-verbose",
        action="store_true",
        help="Enable verbose output emitted by pqkmeans during fitting.")
    parser.add_argument(
        "--save-assignments",
        help="Optional path to save predicted cluster labels as a .npy file.")
    parser.add_argument(
        "--save-model",
        help="Optional path to persist the fitted PQKMeans object via joblib.")
    return parser.parse_args()


def resolve_input_files(inputs: Sequence[str]) -> List[str]:
    """Expand the provided file/directory inputs into a flat file list."""
    files = []
    for file_path in _process_input(inputs):
        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported formats: {SUPPORTED_EXTENSIONS}")
        files.append(file_path)
    if not files:
        raise ValueError("No PQ code files were found for the provided inputs.")
    return files


def _column_sort_key(prefix: str, column: str):
    suffix = column[len(prefix):]
    return int(suffix) if suffix.isdigit() else suffix


def _select_tabular_columns(column_names: Iterable[str],
                            explicit_columns: Sequence[str] | None,
                            prefix: str) -> List[str]:
    if explicit_columns:
        missing = set(explicit_columns) - set(column_names)
        if missing:
            raise ValueError(f"Column(s) {missing} were not found in the input file.")
        return list(explicit_columns)

    pq_columns = [col for col in column_names if col.startswith(prefix)]
    if not pq_columns:
        raise ValueError(
            f"Could not infer PQ code columns. Provide --pq-columns or use the "
            f"--pq-prefix that matches your files.")

    pq_columns.sort(key=lambda col: _column_sort_key(prefix, col))
    return pq_columns


def _load_parquet(path: str, explicit_columns: Sequence[str] | None, prefix: str) -> np.ndarray:
    df = pd.read_parquet(path)
    pq_columns = _select_tabular_columns(df.columns, explicit_columns, prefix)
    data = df[pq_columns].to_numpy()
    del df
    return data


def _load_csv(path: str, explicit_columns: Sequence[str] | None, prefix: str,
              delimiter: str) -> np.ndarray:
    if explicit_columns:
        usecols = list(explicit_columns)
    else:
        header = pd.read_csv(path, nrows=0, delimiter=delimiter)
        usecols = _select_tabular_columns(header.columns, None, prefix)
    df = pd.read_csv(path, usecols=usecols, delimiter=delimiter)
    data = df.to_numpy()
    del df
    return data


def load_pq_codes(paths: Sequence[str],
                  explicit_columns: Sequence[str] | None,
                  prefix: str,
                  npz_key: str | None,
                  csv_delimiter: str,
                  show_progress: bool) -> np.ndarray:
    files = resolve_input_files(paths)
    iterator = files
    if show_progress and len(files) > 1:
        iterator = tqdm(files, desc="Loading PQ codes")

    arrays = []
    for file_path in iterator:
        ext = Path(file_path).suffix.lower()
        if ext == ".npy":
            arr = np.load(file_path)
        elif ext == ".npz":
            with np.load(file_path) as data:
                key = npz_key
                if key is None:
                    if len(data.files) != 1:
                        raise ValueError(
                            f"NPZ file '{file_path}' has multiple arrays; use --npz-key to select one.")
                    key = data.files[0]
                arr = data[key]
        elif ext == ".parquet":
            arr = _load_parquet(file_path, explicit_columns, prefix)
        elif ext == ".csv":
            arr = _load_csv(file_path, explicit_columns, prefix, csv_delimiter)
        else:
            raise ValueError(f"Unsupported file extension for {file_path}")

        if arr.ndim != 2:
            raise ValueError(
                f"PQ codes loaded from '{file_path}' do not form a 2D matrix. Got shape {arr.shape}.")
        arrays.append(np.ascontiguousarray(arr))

    pq_codes = arrays[0] if len(arrays) == 1 else np.vstack(arrays)
    return pq_codes


def main():
    args = parse_args()

    try:
        from chelombus.clustering.PyQKmeans import PQKMeans
    except ModuleNotFoundError as exc:
        missing_dep = exc.name == "pqkmeans"
        msg = ("Missing optional dependency 'pqkmeans'. Install it with "
               "'pip install pqkmeans' to enable PQKMeans clustering.")
        if missing_dep:
            raise ModuleNotFoundError(msg) from exc
        raise

    start = time.time()
    pq_codes = load_pq_codes(
        paths=args.pq_codes,
        explicit_columns=args.pq_columns,
        prefix=args.pq_prefix,
        npz_key=args.npz_key,
        csv_delimiter=args.csv_delimiter,
        show_progress=args.progress)

    print(f"Loaded {pq_codes.shape[0]:,} PQ codes with dimension {pq_codes.shape[1]}.")

    pq_encoder = joblib.load(args.encoder)
    kmeans = PQKMeans(encoder=pq_encoder,
                      k=args.k,
                      iteration=args.iterations,
                      verbose=args.kmeans_verbose)

    train_start = time.time()
    kmeans.fit(pq_codes)
    elapsed = time.time() - train_start
    print(f"Training with {len(pq_codes):,} molecules took: {format_time(elapsed)}")

    if args.save_assignments:
        assignments = kmeans.predict(pq_codes)
        np.save(args.save_assignments, assignments)
        print(f"Saved cluster assignments to {args.save_assignments}.")

    if args.save_model:
        joblib.dump(kmeans, args.save_model)
        print(f"Saved fitted PQKMeans model to {args.save_model}.")

    total_elapsed = time.time() - start
    print(f"Total runtime: {format_time(total_elapsed)}")


if __name__ == "__main__":
    main()
