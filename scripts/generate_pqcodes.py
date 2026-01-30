"""
Generate PQ codes from SMILES input.

Takes SMILES as input, computes MQN fingerprints, transforms them to PQ codes,
and saves both SMILES and PQ codes to parquet files (needed for clustering step).

Example
-------
python scripts/generate_pqcodes.py \
    --input /path/to/smiles/ \
    --output /path/to/output/ \
    --pq-model models/encoder.joblib \
    --chunksize 1000000
"""
import os
import time
import argparse
import gc
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from chelombus import PQEncoder
from chelombus import DataStreamer
from chelombus.utils import FingerprintCalculator
from chelombus.utils.helper_functions import format_time


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate PQ codes from SMILES input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output structure:
  output_dir/
  └── chunk_*.parquet   # Each file contains: smiles, pq_0, pq_1, ..., pq_{m-1}
"""
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="Path to SMILES file or directory containing SMILES files"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help="Output directory for parquet files"
    )
    parser.add_argument(
        '--pq-model',
        type=str,
        required=True,
        help="Path to trained PQEncoder model"
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=1_000_000,
        help="Number of SMILES to process per chunk (default: 1000000)"
    )
    parser.add_argument(
        '--smiles-col',
        type=int,
        default=0,
        help="Column index containing SMILES in input file (default: 0)"
    )
    parser.add_argument(
        '--nprocesses',
        type=int,
        default=None,
        help="Number of processes for fingerprint calculation (default: all CPUs)"
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help="Verbosity level: 0=silent, 1=progress (default: 1)"
    )
    return parser.parse_args()


def process_smiles(
    input_path: str,
    output_dir: str,
    pq_encoder: PQEncoder,
    chunksize: int = 1_000_000,
    smiles_col: int = 0,
    nprocesses: int | None = None,
    verbose: int = 1
):
    """Process SMILES to PQ codes and save with SMILES to parquet.

    Args:
        input_path: Path to SMILES file or directory
        output_dir: Output directory for parquet chunk files
        pq_encoder: Trained PQEncoder instance
        chunksize: Number of SMILES per chunk
        smiles_col: Column index for SMILES in input
        nprocesses: Number of processes for fingerprint calculation
        verbose: Verbosity level
    """
    streamer = DataStreamer()
    calc = FingerprintCalculator()

    total_count = 0
    start_time = time.time()

    # Wrap iterator with tqdm if verbose
    chunks_iter = streamer.parse_input(
        input_path=input_path,
        chunksize=chunksize,
        smiles_col=smiles_col
    )

    pbar = tqdm(desc="Processing chunks", unit="chunk") if verbose > 0 else None

    for chunk_idx, smiles_chunk in enumerate(chunks_iter):

        # Compute MQN fingerprints
        fingerprints = calc.FingerprintFromSmiles(
            smiles=smiles_chunk,
            fp='mqn',
            nprocesses=nprocesses
        )

        # Filter out failed SMILES (where fingerprint calculation failed)
        # FingerprintCalculator returns array without failed entries, so we need
        # to track which SMILES succeeded
        if len(fingerprints) != len(smiles_chunk):
            # Some SMILES failed - fingerprints array is smaller
            # We can't reliably match, so skip SMILES column for this chunk
            valid_smiles = None
            valid_fps = fingerprints
        else:
            valid_smiles = smiles_chunk
            valid_fps = fingerprints

        if len(valid_fps) == 0:
            continue

        
        # Transform to PQ codes
        pq_codes = pq_encoder.transform(valid_fps, verbose=0)

        # Build table with SMILES and PQ codes
        # Using pyarrow directly instead of pandas for memory efficiency:
        data = {}
        if valid_smiles is not None:
            data['smiles'] = valid_smiles
        for i in range(pq_codes.shape[1]):
            data[f'pq_{i}'] = pq_codes[:, i]

        table = pa.Table.from_pydict(data)

        # Write chunk to parquet file
        output_file = os.path.join(output_dir, f"chunk_{chunk_idx:06d}.parquet")
        pq.write_table(table, output_file)

        total_count += len(valid_fps)

        if pbar is not None:
            elapsed = time.time() - start_time
            rate = total_count / elapsed if elapsed > 0 else 0
            pbar.update(1)
            pbar.set_postfix({
                'molecules': f'{total_count:,}',
                'rate': f'{rate:,.0f}/s'
            })

        # Free memory
        del smiles_chunk, fingerprints, valid_fps, pq_codes, table
        gc.collect()

    if pbar is not None:
        pbar.close()

    print(f'Done. Total molecules: {total_count:,}')
    return total_count


def main():
    args = parse_arguments()

    print("\n" + "="*60)
    print("  GENERATE PQ CODES")
    print("="*60)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")
    print(f"PQ model: {args.pq_model}")
    print(f"Chunk size: {args.chunksize:,}")

    # Load trained encoder
    print(f"\nLoading encoder from {args.pq_model}...")
    try:
        pq_encoder = PQEncoder.load(args.pq_model)
    except Exception as e:
        raise ValueError(f'Could not load PQEncoder from {args.pq_model}: {e}')

    if not getattr(pq_encoder, 'encoder_is_trained', False):
        raise ValueError('The loaded model is not a trained PQEncoder')

    print(f"Encoder loaded: m={pq_encoder.m}, k={pq_encoder.k}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("\nStarting processing...\n")
    start = time.time()
    total = process_smiles(
        input_path=args.input,
        output_dir=args.output,
        pq_encoder=pq_encoder,
        chunksize=args.chunksize,
        smiles_col=args.smiles_col,
        nprocesses=args.nprocesses,
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
