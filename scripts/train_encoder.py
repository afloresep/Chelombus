"""Train a PQEncoder on fingerprint training data.

Trains a Product Quantization encoder using KMeans clustering on
subvectors of the input fingerprints.

Example
-------
python scripts/train_encoder.py \
    --input training_data.npy \
    --output encoder.joblib \
    --k 256 --m 6 --iterations 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelombus.encoder.encoder import PQEncoder
from chelombus.utils.helper_functions import format_time


# Default m values for common fingerprint types
DEFAULT_M = {
    "mqn": 6,      # 42 dimensions / 6 = 7 dims per subvector
    "morgan": 8,   # 1024 or 2048 typically divisible by 8
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PQEncoder on fingerprint training data."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to training data file (.npy or .joblib)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save trained encoder (.joblib)."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=256,
        help="Number of centroids per subquantizer (default: 256)."
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Number of subvectors. Auto-detected if not specified: "
             "6 for MQN (42-dim), 8 for Morgan."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="KMeans iterations (default: 300)."
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=1,
        help="Number of KMeans initializations (default: 1)."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level: 0=silent, 1=progress (default: 1)."
    )
    return parser.parse_args()


def load_training_data(input_path: str) -> np.ndarray:
    """Load training data from .npy or .joblib file."""
    path = Path(input_path)
    ext = path.suffix.lower()

    if ext == ".npy":
        data = np.load(input_path)
    elif ext == ".joblib":
        data = joblib.load(input_path)
    else:
        # Try numpy first, then joblib
        try:
            data = np.load(input_path)
        except Exception:
            data = joblib.load(input_path)

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    return data


def auto_detect_m(fp_dim: int) -> int:
    """Auto-detect number of subvectors based on fingerprint dimension."""
    if fp_dim == 42:
        # MQN fingerprints
        return 6
    elif fp_dim == 1024:
        # Morgan 1024-bit
        return 8
    elif fp_dim == 2048:
        # Morgan 2048-bit
        return 8
    elif fp_dim % 8 == 0:
        # Default: divide into 8 subvectors
        return 8
    elif fp_dim % 6 == 0:
        return 6
    elif fp_dim % 4 == 0:
        return 4
    else:
        raise ValueError(
            f"Cannot auto-detect m for dimension {fp_dim}. "
            f"Please specify --m explicitly."
        )


def train_encoder(
    input_path: str,
    output_path: str,
    k: int,
    m: int | None,
    iterations: int,
    n_init: int,
    verbose: int,
) -> None:
    """Train PQEncoder and save to file.

    Args:
        input_path: Path to training data.
        output_path: Path to save trained encoder.
        k: Number of centroids per subquantizer.
        m: Number of subvectors (auto-detected if None).
        iterations: KMeans iterations.
        n_init: Number of KMeans initializations.
        verbose: Verbosity level.
    """
    # Load training data
    if verbose > 0:
        print(f"Loading training data from: {input_path}")

    data = load_training_data(input_path)

    if verbose > 0:
        print(f"Training data shape: {data.shape}")
        print(f"Data type: {data.dtype}")

    n_samples, fp_dim = data.shape

    # Auto-detect m if not specified
    if m is None:
        m = auto_detect_m(fp_dim)
        if verbose > 0:
            print(f"Auto-detected m={m} for dimension {fp_dim}")

    # Validate m divides dimension evenly
    if fp_dim % m != 0:
        raise ValueError(
            f"Fingerprint dimension ({fp_dim}) must be divisible by m ({m}). "
            f"Dimension / m = {fp_dim / m}"
        )

    subvector_dim = fp_dim // m

    if verbose > 0:
        print(f"\nEncoder parameters:")
        print(f"  k (centroids): {k}")
        print(f"  m (subvectors): {m}")
        print(f"  Subvector dimension: {subvector_dim}")
        print(f"  KMeans iterations: {iterations}")
        print(f"  KMeans n_init: {n_init}")
        print(f"  Training samples: {n_samples:,}")
        print()

    # Create and train encoder
    encoder = PQEncoder(k=k, m=m, iterations=iterations)

    start = time.time()
    encoder.fit(data, verbose=verbose, n_init=n_init)
    elapsed = time.time() - start

    if verbose > 0:
        print(f"\nTraining completed in {format_time(elapsed)}")

    # Save encoder
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder.save(output_path)

    if verbose > 0:
        print(f"Saved encoder to: {output_path}")

        # Print compression info
        original_bytes = n_samples * fp_dim * 4  # float32
        compressed_bytes = n_samples * m * 1  # uint8 codes
        compression_ratio = original_bytes / compressed_bytes
        print(f"\nCompression info:")
        print(f"  Original size per vector: {fp_dim * 4} bytes (float32)")
        print(f"  Compressed size per vector: {m} bytes (uint8 codes)")
        print(f"  Compression ratio: {compression_ratio:.1f}x")


def main():
    args = parse_arguments()

    train_encoder(
        input_path=args.input,
        output_path=args.output,
        k=args.k,
        m=args.m,
        iterations=args.iterations,
        n_init=args.n_init,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
