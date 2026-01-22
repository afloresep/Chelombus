"""Run the complete Chelombus clustering pipeline.

End-to-end pipeline: SMILES → MQN fingerprints → PQ codes → Clusters

This script runs all stages with checkpointing support, allowing you to
resume from where you left off if interrupted.

Stages:
    1. Sample SMILES and compute fingerprints for encoder training
    2. Train PQEncoder on sampled fingerprints
    3. Generate PQ codes from training fingerprints for clusterer training
    4. Train PQKMeans clusterer
    5. Process all input data: SMILES → fingerprints → PQ codes → cluster assignments

Example
-------
# Full pipeline
python scripts/run_pipeline.py \
    --input data/*.smi \
    --output results/ \
    --n-clusters 100000 \
    --encoder-training-size 50000000 \
    --clusterer-training-size 1000000000

# Resume interrupted pipeline
python scripts/run_pipeline.py \
    --input data/*.smi \
    --output results/ \
    --n-clusters 100000 \
    --resume
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelombus.encoder.encoder import PQEncoder
from chelombus.streamer.data_streamer import DataStreamer
from chelombus.utils.fingerprints import FingerprintCalculator
from chelombus.utils.helper_functions import format_time


# Constants
MQN_SIZE = 42
DEFAULT_M = 6  # For MQN fingerprints


@dataclass
class PipelineConfig:
    """Configuration for the pipeline run."""
    input_path: str
    output_dir: str
    n_clusters: int
    encoder_training_size: int
    clusterer_training_size: int
    chunksize: int
    smiles_col: int
    encoder_k: int
    encoder_m: int
    encoder_iterations: int
    clusterer_iterations: int
    verbose: int

    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'PipelineConfig':
        with open(path, 'r') as f:
            return cls(**json.load(f))


@dataclass
class PipelineState:
    """Tracks which stages have been completed."""
    stage1_complete: bool = False
    stage2_complete: bool = False
    stage3_complete: bool = False
    stage4_complete: bool = False
    stage5_complete: bool = False
    stage5_chunks_processed: int = 0

    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'PipelineState':
        if not path.exists():
            return cls()
        with open(path, 'r') as f:
            return cls(**json.load(f))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete Chelombus clustering pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1. Sample fingerprints for encoder training
  2. Train PQEncoder
  3. Generate PQ codes for clusterer training
  4. Train PQKMeans clusterer
  5. Process all data and assign clusters

Output structure:
  output_dir/
  ├── config.json           # Pipeline configuration
  ├── state.json            # Checkpoint state
  ├── models/
  │   ├── encoder.joblib    # Trained PQEncoder
  │   └── clusterer.joblib  # Trained PQKMeans
  ├── training/
  │   ├── fp_training.npy   # Fingerprints for encoder training
  │   └── pq_training.npy   # PQ codes for clusterer training
  └── results/
      └── chunk_*.parquet   # Final clustered output (smiles, cluster_id)
"""
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to SMILES file(s) or directory (glob patterns supported)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for all pipeline outputs."
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        required=True,
        help="Number of clusters for PQKMeans."
    )

    # Training sizes
    parser.add_argument(
        "--encoder-training-size",
        type=int,
        default=50_000_000,
        help="Number of fingerprints for encoder training (default: 50M)."
    )
    parser.add_argument(
        "--clusterer-training-size",
        type=int,
        default=None,
        help="Number of PQ codes for clusterer training. "
             "Default: min(1B, all available data)."
    )

    # Processing parameters
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="SMILES per chunk during processing (default: 100000)."
    )
    parser.add_argument(
        "--smiles-col",
        type=int,
        default=0,
        help="Column index for SMILES in input files (default: 0)."
    )

    # Encoder parameters
    parser.add_argument(
        "--encoder-k",
        type=int,
        default=256,
        help="Number of centroids per subquantizer (default: 256)."
    )
    parser.add_argument(
        "--encoder-m",
        type=int,
        default=DEFAULT_M,
        help=f"Number of subvectors for PQ (default: {DEFAULT_M} for MQN)."
    )
    parser.add_argument(
        "--encoder-iterations",
        type=int,
        default=20,
        help="KMeans iterations for encoder training (default: 20)."
    )

    # Clusterer parameters
    parser.add_argument(
        "--clusterer-iterations",
        type=int,
        default=20,
        help="Iterations for PQKMeans training (default: 20)."
    )

    # Control flags
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skip completed stages)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force restart from beginning, ignoring checkpoints."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level: 0=silent, 1=progress, 2=detailed (default: 1)."
    )

    return parser.parse_args()


def print_stage(stage_num: int, description: str, status: str = "RUNNING"):
    """Print stage header."""
    symbols = {"RUNNING": "→", "DONE": "✓", "SKIP": "○"}
    symbol = symbols.get(status, "→")
    print(f"\n{'='*60}")
    print(f"  {symbol} Stage {stage_num}: {description}")
    print(f"{'='*60}\n")


def count_input_molecules(input_path: str, smiles_col: int) -> int:
    """Estimate total molecules in input files."""
    ds = DataStreamer()
    count = 0
    for chunk in ds.parse_input(input_path, chunksize=100_000, verbose=0, smiles_col=smiles_col):
        count += len(chunk)
    return count


def stage1_sample_fingerprints(
    config: PipelineConfig,
    state: PipelineState,
    paths: dict,
    verbose: int,
) -> None:
    """Stage 1: Sample SMILES and compute fingerprints for encoder training."""

    print_stage(1, "Sample fingerprints for encoder training")

    training_size = config.encoder_training_size
    print(f"Target training size: {training_size:,} fingerprints")

    ds = DataStreamer()
    fp_calc = FingerprintCalculator()

    # Pre-allocate array for training fingerprints
    training_fps = np.empty((training_size, MQN_SIZE), dtype=np.uint32)
    collected = 0

    # Calculate sampling rate
    start_time = time.time()

    for chunk in ds.parse_input(
        config.input_path,
        chunksize=config.chunksize,
        verbose=verbose,
        smiles_col=config.smiles_col
    ):
        # Calculate fingerprints
        fps = fp_calc.FingerprintFromSmiles(chunk, fp="mqn", verbose=verbose)

        if len(fps) == 0:
            continue

        # Store in pre-allocated array
        n_valid = min(len(fps), training_size - collected)
        training_fps[collected:collected + n_valid] = fps[:n_valid]
        collected += n_valid

        print(f"\rCollected: {collected:,}/{training_size:,} fingerprints", end="", flush=True)

        del chunk, fps
        gc.collect()

        if collected >= config.encoder_training_size:
            break
        

    # Trim to actual size
    training_fps = training_fps[:collected]

    elapsed = time.time() - start_time
    print(f"\n\nCollected {collected:,} fingerprints in {format_time(elapsed)}")

    # Save training fingerprints
    np.save(paths['fp_training'], training_fps)
    print(f"Saved to: {paths['fp_training']}")

    state.stage1_complete = True
    state.save(paths['state'])


def stage2_train_encoder(
    config: PipelineConfig,
    state: PipelineState,
    paths: dict,
    verbose: int,
) -> PQEncoder:
    """Stage 2: Train PQEncoder on sampled fingerprints."""

    print_stage(2, "Train PQEncoder")

    # Load training data
    print(f"Loading training data from: {paths['fp_training']}")
    training_fps = np.load(paths['fp_training'])
    print(f"Training data shape: {training_fps.shape}")

    # Create and train encoder
    encoder = PQEncoder(
        k=config.encoder_k,
        m=config.encoder_m,
        iterations=config.encoder_iterations
    )

    print(f"\nEncoder parameters:")
    print(f"  k (centroids): {config.encoder_k}")
    print(f"  m (subvectors): {config.encoder_m}")
    print(f"  iterations: {config.encoder_iterations}")
    print(f"  subvector dimension: {MQN_SIZE // config.encoder_m}")
    print()

    start_time = time.time()
    encoder.fit(training_fps, verbose=verbose)
    elapsed = time.time() - start_time

    print(f"\nEncoder training completed in {format_time(elapsed)}")

    # Save encoder
    encoder.save(paths['encoder'])
    print(f"Saved encoder to: {paths['encoder']}")

    # Print compression info
    original_bytes = MQN_SIZE * 4  
    compressed_bytes = config.encoder_m  # uint8 codes
    print(f"\nCompression: {original_bytes} bytes → {compressed_bytes} bytes ({original_bytes/compressed_bytes:.0f}x)")

    state.stage2_complete = True
    state.save(paths['state'])

    return encoder


def stage3_generate_pq_training(
    config: PipelineConfig,
    state: PipelineState,
    paths: dict,
    encoder: PQEncoder,
    verbose: int,
) -> None:
    """Stage 3: Generate PQ codes for clusterer training."""

    print_stage(3, "Generate PQ codes for clusterer training")

    # Determine training size for clusterer
    if config.clusterer_training_size is None:
        # Use all fingerprints we have (from stage 1) to generate PQ codes
        # Then we'll sample more if needed
        clusterer_training_size = 1_000_000_000  # Default 1B max
    else:
        clusterer_training_size = config.clusterer_training_size

    print(f"Target: {clusterer_training_size:,} PQ codes for clusterer training")

    ds = DataStreamer()
    fp_calc = FingerprintCalculator()

    # Pre-allocate array for PQ codes
    pq_codes = np.empty((clusterer_training_size, config.encoder_m), dtype=np.uint8)
    collected = 0

    start_time = time.time()

    for chunk in ds.parse_input(
        config.input_path,
        chunksize=config.chunksize,
        verbose=0,
        smiles_col=config.smiles_col
    ):
        if collected >= clusterer_training_size:
            break

        # Calculate fingerprints
        fps = fp_calc.FingerprintFromSmiles(chunk, fp="mqn")

        if len(fps) == 0:
            continue

        # Transform to PQ codes
        chunk_pq_codes = encoder.transform(fps, verbose=0)

        # Store in pre-allocated array
        n_to_store = min(len(chunk_pq_codes), clusterer_training_size - collected)
        pq_codes[collected:collected + n_to_store] = chunk_pq_codes[:n_to_store]
        collected += n_to_store

        print(f"\rGenerated: {collected:,}/{clusterer_training_size:,} PQ codes", end="", flush=True)

        del chunk, fps, chunk_pq_codes
        gc.collect()

    # Trim to actual size
    pq_codes = pq_codes[:collected]

    elapsed = time.time() - start_time
    print(f"\n\nGenerated {collected:,} PQ codes in {format_time(elapsed)}")

    # Save PQ codes for clusterer training
    np.save(paths['pq_training'], pq_codes)
    print(f"Saved to: {paths['pq_training']}")

    state.stage3_complete = True
    state.save(paths['state'])


def stage4_train_clusterer(
    config: PipelineConfig,
    state: PipelineState,
    paths: dict,
    encoder: PQEncoder,
    verbose: int,
):
    """Stage 4: Train PQKMeans clusterer."""

    print_stage(4, "Train PQKMeans clusterer")

    # Import PQKMeans (requires pqkmeans)
    try:
        from chelombus.clustering.PyQKmeans import PQKMeans
    except ImportError as e:
        print("Error: pqkmeans is required for clustering.")
        print("Install with: pip install pqkmeans")
        raise e

    # Load PQ training codes
    print(f"Loading PQ codes from: {paths['pq_training']}")
    pq_codes = np.load(paths['pq_training'])
    print(f"Training data shape: {pq_codes.shape}")

    # Create clusterer
    clusterer = PQKMeans(
        encoder=encoder,
        k=config.n_clusters,
        iteration=config.clusterer_iterations,
        verbose=verbose > 0
    )

    print(f"\nClusterer parameters:")
    print(f"  k (clusters): {config.n_clusters:,}")
    print(f"  iterations: {config.clusterer_iterations}")
    print()

    start_time = time.time()
    clusterer.fit(pq_codes)
    elapsed = time.time() - start_time

    print(f"\nClusterer training completed in {format_time(elapsed)}")

    # Save clusterer
    clusterer.save(paths['clusterer'])
    print(f"Saved clusterer to: {paths['clusterer']}")

    state.stage4_complete = True
    state.save(paths['state'])

    return clusterer


def stage5_cluster_all_data(
    config: PipelineConfig,
    state: PipelineState,
    paths: dict,
    encoder: PQEncoder,
    clusterer,
    verbose: int,
) -> None:
    """Stage 5: Process all data and assign clusters."""

    print_stage(5, "Cluster all molecules")

    ds = DataStreamer()
    fp_calc = FingerprintCalculator()

    # Create results directory
    results_dir = paths['results']
    os.makedirs(results_dir, exist_ok=True)

    # Resume from last processed chunk if applicable
    start_chunk = state.stage5_chunks_processed
    if start_chunk > 0:
        print(f"Resuming from chunk {start_chunk}")

    total_molecules = 0
    total_valid = 0
    chunk_idx = 0

    start_time = time.time()

    for smiles_chunk in ds.parse_input(
        config.input_path,
        chunksize=config.chunksize,
        verbose=0,
        smiles_col=config.smiles_col
    ):
        # Skip already processed chunks
        if chunk_idx < start_chunk:
            chunk_idx += 1
            continue

        # Calculate fingerprints
        fps = fp_calc.FingerprintFromSmiles(smiles_chunk, fp="mqn")

        if len(fps) == 0:
            chunk_idx += 1
            continue

        # Transform to PQ codes
        pq_codes = encoder.transform(fps, verbose=0)

        # Predict clusters
        cluster_ids = clusterer.predict(pq_codes)

        # Create output DataFrame
        if len(fps) == len(smiles_chunk):
            output_df = pd.DataFrame({
                "smiles": smiles_chunk,
                "cluster_id": cluster_ids,
            })
        else:
            # Some SMILES failed - can't reliably associate
            output_df = pd.DataFrame({
                "cluster_id": cluster_ids,
            })

        # Save chunk
        output_file = os.path.join(results_dir, f"chunk_{chunk_idx:05d}.parquet")
        output_df.to_parquet(output_file, index=False)

        total_molecules += len(smiles_chunk)
        total_valid += len(fps)
        chunk_idx += 1

        # Update checkpoint
        state.stage5_chunks_processed = chunk_idx
        state.save(paths['state'])

        if verbose > 0:
            elapsed_so_far = time.time() - start_time
            rate = total_valid / elapsed_so_far if elapsed_so_far > 0 else 0
            print(
                f"\rProcessed: {total_valid:,} molecules "
                f"({chunk_idx} chunks, {rate:,.0f} mol/s)",
                end="",
                flush=True,
            )

        del smiles_chunk, fps, pq_codes, cluster_ids, output_df
        gc.collect()

    elapsed = time.time() - start_time

    print(f"\n\nClustering completed!")
    print(f"Total molecules: {total_valid:,}")
    print(f"Total chunks: {chunk_idx}")
    print(f"Time elapsed: {format_time(elapsed)}")
    print(f"Results directory: {results_dir}")

    state.stage5_complete = True
    state.save(paths['state'])


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the complete clustering pipeline."""

    # Setup output directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    training_dir = output_dir / "training"
    training_dir.mkdir(exist_ok=True)

    results_dir = output_dir / "results"

    # Define all paths
    paths = {
        'config': output_dir / "config.json",
        'state': output_dir / "state.json",
        'encoder': models_dir / "encoder.joblib",
        'clusterer': models_dir / "clusterer.joblib",
        'fp_training': training_dir / "fp_training.npy",
        'pq_training': training_dir / "pq_training.npy",
        'results': results_dir,
    }

    # Create or load configuration
    config = PipelineConfig(
        input_path=args.input,
        output_dir=str(output_dir),
        n_clusters=args.n_clusters,
        encoder_training_size=args.encoder_training_size,
        clusterer_training_size=args.clusterer_training_size,
        chunksize=args.chunksize,
        smiles_col=args.smiles_col,
        encoder_k=args.encoder_k,
        encoder_m=args.encoder_m,
        encoder_iterations=args.encoder_iterations,
        clusterer_iterations=args.clusterer_iterations,
        verbose=args.verbose,
    )
    config.save(paths['config'])

    # Load or create state
    if args.force:
        state = PipelineState()
    else:
        state = PipelineState.load(paths['state'])

    verbose = args.verbose

    # Print pipeline overview
    print("\n" + "="*60)
    print("  CHELOMBUS CLUSTERING PIPELINE")
    print("="*60)
    print(f"\nInput: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Clusters: {args.n_clusters:,}")
    print(f"Encoder training size: {args.encoder_training_size:,}")
    print(f"Clusterer training size: {args.clusterer_training_size or 'auto':,}" if args.clusterer_training_size else f"Clusterer training size: auto (up to 1B)")

    if args.resume:
        print("\nResume mode: Will skip completed stages")
        print(f"  Stage 1 (sample FPs): {'Done' if state.stage1_complete else 'Pending'}")
        print(f"  Stage 2 (train encoder): {'Done' if state.stage2_complete else 'Pending'}")
        print(f"  Stage 3 (generate PQ codes): {'Done' if state.stage3_complete else 'Pending'}")
        print(f"  Stage 4 (train clusterer): {'Done' if state.stage4_complete else 'Pending'}")
        print(f"  Stage 5 (cluster all): {'Done' if state.stage5_complete else f'Pending (chunks: {state.stage5_chunks_processed})'}")

    pipeline_start = time.time()

    # Stage 1: Sample fingerprints for encoder training
    if state.stage1_complete and args.resume:
        print_stage(1, "Sample fingerprints for encoder training", "SKIP")
        print("Already completed, skipping...")
    else:
        stage1_sample_fingerprints(config, state, paths, verbose)

    # Stage 2: Train encoder
    if state.stage2_complete and args.resume:
        print_stage(2, "Train PQEncoder", "SKIP")
        print("Already completed, loading existing encoder...")
        encoder = PQEncoder.load(paths['encoder'])
    else:
        encoder = stage2_train_encoder(config, state, paths, verbose)

    # Stage 3: Generate PQ codes for clusterer training
    if state.stage3_complete and args.resume:
        print_stage(3, "Generate PQ codes for clusterer training", "SKIP")
        print("Already completed, skipping...")
    else:
        stage3_generate_pq_training(config, state, paths, encoder, verbose)

    # Stage 4: Train clusterer
    if state.stage4_complete and args.resume:
        print_stage(4, "Train PQKMeans clusterer", "SKIP")
        print("Already completed, loading existing clusterer...")
        from chelombus.clustering.PyQKmeans import PQKMeans
        clusterer = PQKMeans.load(paths['clusterer'])
    else:
        clusterer = stage4_train_clusterer(config, state, paths, encoder, verbose)

    # Stage 5: Cluster all data
    if state.stage5_complete and args.resume:
        print_stage(5, "Cluster all molecules", "SKIP")
        print("Already completed!")
    else:
        stage5_cluster_all_data(config, state, paths, encoder, clusterer, verbose)

    # Final summary
    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"\nTotal time: {format_time(pipeline_elapsed)}")
    print(f"\nOutputs:")
    print(f"  Encoder:   {paths['encoder']}")
    print(f"  Clusterer: {paths['clusterer']}")
    print(f"  Results:   {paths['results']}/")
    print(f"\nTo query clusters, use:")
    print(f"  import chelombus")
    print(f"  df = chelombus.query_cluster('{paths['results']}', cluster_id=42)")


def main():
    args = parse_arguments()

    # Check for pqkmeans dependency
    try:
        import pqkmeans  # noqa: F401
    except ImportError:
        print(
            "Error: pqkmeans is required for clustering.\n"
            "Install with: pip install pqkmeans"
        )
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
