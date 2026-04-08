"""TMAP visualization utilities for molecular datasets.

Provides functions to create TMAPs from SMILES strings with customizable
molecular properties and fingerprint options. Built on top of tmap2.
"""
import argparse
import numpy as np
import pandas as pd
from typing import Sequence

from tmap import TMAP
from tmap.utils import fingerprints_from_smiles, molecular_properties, AVAILABLE_PROPERTIES

DEFAULT_PROPERTIES = list(AVAILABLE_PROPERTIES)


def create_tmap(
    smiles: list[str],
    fingerprint: str = 'morgan',
    fingerprint_bits: int = 2048,
    fingerprint_radius: int = 2,
    properties: list[str] | None = None,
    tmap_name: str = 'my_tmap',
    k_neighbors: int = 20,
) -> None:
    """Create a TMAP visualization from SMILES strings.

    Args:
        smiles: List of SMILES strings to visualize.
        fingerprint: Fingerprint type ('morgan' or 'mqn').
        fingerprint_bits: Number of bits for Morgan fingerprint (default: 2048).
        fingerprint_radius: Radius for Morgan fingerprint (default: 2).
        properties: List of molecular properties to display as color channels.
                   If None, uses all available properties.
        tmap_name: Name for the output TMAP HTML file.
        k_neighbors: Number of neighbors for layout (default: 20).
    """
    if properties is None:
        properties = DEFAULT_PROPERTIES

    _validate_properties(properties)

    # Compute fingerprints via tmap2
    fp_kwargs = {}
    if fingerprint == 'morgan':
        fp_kwargs = {'n_bits': fingerprint_bits, 'radius': fingerprint_radius}

    fps = fingerprints_from_smiles(smiles, fp_type=fingerprint, **fp_kwargs)

    # Fit TMAP
    model = TMAP(metric='jaccard', n_neighbors=k_neighbors).fit(fps)

    # Build visualization
    viz = model.to_tmapviz(include_edges=True)
    viz.title = tmap_name
    viz.background_color = "#FFFFFF"

    # Add molecular properties as color layers
    props = molecular_properties(smiles, properties=properties)
    for prop_name, values in props.items():
        viz.add_color_layout(prop_name, values, categorical=False, color='rainbow')

    # Add SMILES for structure rendering in tooltips
    viz.add_smiles(smiles)

    viz.write_html(f"{tmap_name}.html")


def representative_tmap(
    smiles: list[str],
    cluster_ids: Sequence[int | str] | None = None,
    fingerprint: str = 'mqn',
    fingerprint_bits: int = 2048,
    fingerprint_radius: int = 2,
    properties: list[str] | None = None,
    tmap_name: str = 'representative_tmap',
    k_neighbors: int = 21,
) -> None:
    """Create a TMAP for representative molecules.

    Recommended for primary TMAPs showing cluster representatives.

    Args:
        smiles: List of SMILES strings (typically cluster representatives).
        cluster_ids: Optional list of cluster IDs corresponding to each SMILES.
                    If provided, added as a categorical color channel.
        fingerprint: Fingerprint type ('mqn' recommended, or 'morgan').
        fingerprint_bits: Number of bits for Morgan fingerprint (default: 2048).
        fingerprint_radius: Radius for Morgan fingerprint (default: 2).
        properties: List of molecular properties to display as color channels.
                   If None, uses all available properties.
        tmap_name: Name for the output TMAP HTML file.
        k_neighbors: Number of neighbors for KNN graph (default: 21).
    """
    if properties is None:
        properties = DEFAULT_PROPERTIES

    _validate_properties(properties)

    # Compute fingerprints via tmap2
    fp_kwargs = {}
    if fingerprint == 'morgan':
        fp_kwargs = {'n_bits': fingerprint_bits, 'radius': fingerprint_radius}

    fps = fingerprints_from_smiles(smiles, fp_type=fingerprint, **fp_kwargs)

    # Fit TMAP — use euclidean for dense descriptors like MQN
    metric = 'euclidean' if fingerprint == 'mqn' else 'jaccard'
    model = TMAP(metric=metric, n_neighbors=k_neighbors).fit(fps)

    # Build visualization
    viz = model.to_tmapviz(include_edges=True)
    viz.title = tmap_name
    viz.background_color = "#FFFFFF"

    # Add molecular properties as color layers
    props = molecular_properties(smiles, properties=properties)
    for prop_name, values in props.items():
        viz.add_color_layout(prop_name, values, categorical=False, color='rainbow')

    # Add cluster ID as categorical color channel
    if cluster_ids is not None:
        viz.add_color_layout(
            'Cluster ID',
            [str(cid) for cid in cluster_ids],
            categorical=True,
            color='tab20',
        )

    # Add SMILES for structure rendering in tooltips
    viz.add_smiles(smiles)

    viz.write_html(f"{tmap_name}.html")


def _validate_properties(properties: list[str]) -> None:
    """Raise ValueError if any property name is not recognized by tmap2."""
    invalid = set(properties) - set(AVAILABLE_PROPERTIES)
    if invalid:
        raise ValueError(
            f"Unknown properties: {invalid}. Available: {AVAILABLE_PROPERTIES}"
        )


def main():
    """CLI entry point for TMAP generation."""
    parser = argparse.ArgumentParser(
        description="Generate TMAP visualizations from SMILES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available properties: {', '.join(AVAILABLE_PROPERTIES)}

Examples:
  chelombus-tmap --smiles molecules.smi
  chelombus-tmap --smiles molecules.smi --fingerprint mqn --properties mw,logp,qed
  chelombus-tmap --smiles molecules.smi --cluster-file clusters.csv
        """
    )
    parser.add_argument('--smiles', type=str, required=True,
                       help="Path to file containing SMILES (one per line)")
    parser.add_argument('--fingerprint', type=str, default='morgan',
                       choices=['morgan', 'mqn'],
                       help="Fingerprint type (default: morgan)")
    parser.add_argument('--bits', type=int, default=2048,
                       help="Morgan fingerprint bits (default: 2048)")
    parser.add_argument('--radius', type=int, default=2,
                       help="Morgan fingerprint radius (default: 2)")
    parser.add_argument('--properties', type=str, default=None,
                       help=f"Comma-separated properties (default: all). Available: {', '.join(AVAILABLE_PROPERTIES)}")
    parser.add_argument('--output', type=str, default='my_tmap',
                       help="Output TMAP name (default: my_tmap)")
    parser.add_argument('--cluster-file', type=str, default=None,
                       help="CSV/parquet file with 'smiles' and 'cluster_id' columns for representative TMAP")
    parser.add_argument('--representative', action='store_true',
                       help="Use representative_tmap layout (edge list based)")

    args = parser.parse_args()

    # Parse properties
    properties = None
    if args.properties:
        properties = [p.strip() for p in args.properties.split(',')]

    # Load data
    if args.cluster_file:
        if args.cluster_file.endswith('.parquet'):
            df = pd.read_parquet(args.cluster_file)
        elif args.cluster_file.endswith('.csv'):
            df = pd.read_csv(args.cluster_file)
        else:
            raise ValueError("Cluster file must be .csv or .parquet")

        smiles = df['smiles'].tolist()
        cluster_ids = df['cluster_id'].tolist() if 'cluster_id' in df.columns else None

        representative_tmap(
            smiles=smiles,
            cluster_ids=cluster_ids,
            fingerprint=args.fingerprint,
            fingerprint_bits=args.bits,
            fingerprint_radius=args.radius,
            properties=properties,
            tmap_name=args.output,
        )
    else:
        with open(args.smiles, 'r') as file:
            smiles = [line.strip() for line in file.readlines() if line.strip()]

        if args.representative:
            representative_tmap(
                smiles=smiles,
                fingerprint=args.fingerprint,
                fingerprint_bits=args.bits,
                fingerprint_radius=args.radius,
                properties=properties,
                tmap_name=args.output,
            )
        else:
            create_tmap(
                smiles=smiles,
                fingerprint=args.fingerprint,
                fingerprint_bits=args.bits,
                fingerprint_radius=args.radius,
                properties=properties,
                tmap_name=args.output,
            )


if __name__ == "__main__":
    main()
