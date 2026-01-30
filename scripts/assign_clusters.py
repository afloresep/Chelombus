"""
Assign cluster IDs to molecules using a trained PQKMeans model.

Takes parquet files with SMILES and PQ codes (from generate_pqcodes.py) and
assigns cluster IDs using a trained PQKMeans clusterer.

Example
-------
python scripts/assign_clusters.py \
    --input /path/to/pqcodes/ \
    --output /path/to/clustered/ \
    --clusterer models/clusterer.joblib
"""
import os
import time
import argparse
import glob
import gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from chelombus.clustering.PyQKmeans import PQKMeans
from chelombus.utils.helper_functions import format_time


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Assign cluster IDs to molecules using trained PQKMeans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input structure (from generate_pqcodes.py):
  input_dir/
  └── chunk_*.parquet   # Each file: smiles, pq_0, pq_1, ..., pq_{m-1}

Output structure:
  output_dir/
  └── chunk_*.parquet   # Each file: smiles, cluster_id
"""
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="Input directory containing parquet files with PQ codes"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help="Output directory for parquet files with cluster assignments"
    )
    parser.add_argument(
        '--clusterer',
        type=str,
        required=True,
        help="Path to trained PQKMeans model"
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help="Verbosity level: 0=silent, 1=progress (default: 1)"
    )
    return parser.parse_args()


def assign_clusters(
    input_dir: str,
    output_dir: str,
    clusterer: PQKMeans,
    verbose: int = 1
) -> int:
    """Assign cluster IDs to molecules from PQ code parquet files.

    Args:
        input_dir: Directory with parquet files containing PQ codes
        output_dir: Output directory for clustered parquet files
        clusterer: Trained PQKMeans instance
        verbose: Verbosity level

    Returns:
        Total number of molecules processed
    """
    # Find all parquet files
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {input_dir}")

    total_count = 0
    start_time = time.time()

    pbar = tqdm(parquet_files, desc="Clustering", unit="file") if verbose > 0 else parquet_files

    for file_idx, input_file in enumerate(pbar):
        # Read parquet file
        table = pq.read_table(input_file)
        df_columns = table.column_names

        # Extract PQ code columns
        pq_cols = sorted([c for c in df_columns if c.startswith('pq_')])
        if not pq_cols:
            raise ValueError(f"No PQ code columns found in {input_file}")

        # Build PQ codes array
        pq_codes = np.column_stack([
            table.column(col).to_numpy() for col in pq_cols
        ]).astype(np.uint8)

        # Predict cluster IDs
        s = time.time()
        cluster_ids = clusterer.predict(pq_codes)
        print(format_time(time.time() - s))

        # Build output table
        data = {'cluster_id': cluster_ids}
        if 'smiles' in df_columns:
            data['smiles'] = table.column('smiles').to_pylist()

        output_table = pa.Table.from_pydict(data)

        # Write output file
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        pq.write_table(output_table, output_file)

        total_count += len(cluster_ids)

        if verbose > 0:
            elapsed = time.time() - start_time
            rate = total_count / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                'molecules': f'{total_count:,}',
                'rate': f'{rate:,.0f}/s'
            })

        # Free memory
        del table, pq_codes, cluster_ids, output_table
        gc.collect()

    print(f'Done. Total molecules: {total_count:,}')
    return total_count


def main():
    args = parse_arguments()

    print("\n" + "="*60)
    print("  ASSIGN CLUSTERS")
    print("="*60)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")
    print(f"Clusterer: {args.clusterer}")

    # Load trained clusterer
    print(f"\nLoading clusterer from {args.clusterer}...")
    try:
        clusterer = PQKMeans.load(args.clusterer)
    except Exception as e:
        raise ValueError(f'Could not load PQKMeans from {args.clusterer}: {e}')

    if not clusterer.is_trained:
        raise ValueError('The loaded model is not a trained PQKMeans clusterer')

    print(f"Clusterer loaded: k={clusterer.k:,} clusters")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("\nStarting clustering...\n")
    start = time.time()
    total = assign_clusters(
        input_dir=args.input,
        output_dir=args.output,
        clusterer=clusterer,
        verbose=args.verbose
    )
    elapsed = time.time() - start

    print("\n" + "="*60)
    print("  COMPLETE")
    print("="*60)
    print(f"\nTotal molecules: {total:,}")
    print(f"Total time: {format_time(elapsed)}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
