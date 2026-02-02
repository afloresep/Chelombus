"""TMAP visualization utilities for molecular datasets.

Provides functions to create TMAPs from SMILES strings with customizable
molecular properties and fingerprint options.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from faerun import Faerun
import tmap as tm
import argparse
import pandas as pd
from pandarallel import pandarallel
from chelombus.utils.fingerprints import FingerprintCalculator
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Callable

# Available molecular properties with display names and calculation functions
AVAILABLE_PROPERTIES: dict[str, tuple[str, Callable]] = {
    'hac': ('HAC', lambda mol: mol.GetNumHeavyAtoms()),
    'num_aromatic_atoms': ('Number Aromatic Atoms', lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())),
    'num_hba': ('NumHBA', rdMolDescriptors.CalcNumHBA),
    'num_hbd': ('NumHBD', rdMolDescriptors.CalcNumHBD),
    'num_rings': ('Number of Rings', rdMolDescriptors.CalcNumRings),
    'mw': ('MW', Descriptors.ExactMolWt),
    'clogp': ('cLogP', Descriptors.MolLogP),
    'fsp3': ('Fraction Csp3', Descriptors.FractionCSP3),
}

DEFAULT_PROPERTIES = ['hac', 'num_aromatic_atoms', 'num_hba', 'num_hbd', 'num_rings', 'mw', 'clogp', 'fsp3']


def _mol_properties_from_smiles(smiles: str, properties: list[str]) -> tuple | None:
    """Get molecular properties from a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    values = []
    for prop in properties:
        display_name, calc_func = AVAILABLE_PROPERTIES[prop]
        values.append(calc_func(mol))

    return tuple(values)


def calc_properties(smiles: list[str], properties: list[str] | None = None) -> pd.DataFrame:
    """Calculate selected molecular properties for a list of SMILES.

    Args:
        smiles: List of SMILES strings.
        properties: List of property identifiers to calculate.
                   If None, uses DEFAULT_PROPERTIES.
                   Available: hac, num_aromatic_atoms, num_hba, num_hbd,
                             num_rings, mw, clogp, fsp3

    Returns:
        DataFrame with 'smiles' column and selected property columns.

    Raises:
        ValueError: If unknown property identifiers are provided.
    """
    if properties is None:
        properties = DEFAULT_PROPERTIES

    # Validate properties
    invalid = set(properties) - set(AVAILABLE_PROPERTIES.keys())
    if invalid:
        raise ValueError(f"Unknown properties: {invalid}. Available: {list(AVAILABLE_PROPERTIES.keys())}")

    pandarallel.initialize(progress_bar=False, verbose=0)

    dataframe = pd.DataFrame({'smiles': smiles})

    # Get display names for columns
    column_names = [AVAILABLE_PROPERTIES[prop][0] for prop in properties]

    dataframe[column_names] = dataframe['smiles'].apply(
        lambda s: _mol_properties_from_smiles(s, properties)
    ).parallel_apply(pd.Series)

    return dataframe


def _calculate_fingerprint(smiles: list[str], fp: str, bits: int = 1024, radius: int = 2) -> np.ndarray:
    """Calculate fingerprints for a list of SMILES."""
    fp_calc = FingerprintCalculator()
    fp_arr = fp_calc.FingerprintFromSmiles(smiles, fp=fp, fpSize=bits, radius=radius)
    return fp_arr


def create_tmap(
    smiles: list[str],
    fingerprint: str = 'morgan',
    fingerprint_bits: int = 1024,
    fingerprint_radius: int = 2,
    properties: list[str] | None = None,
    tmap_name: str = 'my_tmap',
    k_neighbors: int = 20,
) -> None:
    """Create a TMAP visualization from SMILES strings.

    Uses LSH Forest for layout generation.

    Args:
        smiles: List of SMILES strings to visualize.
        fingerprint: Fingerprint type ('morgan' or 'mqn').
        fingerprint_bits: Number of bits for fingerprint (default: 1024).
        fingerprint_radius: Radius for Morgan fingerprint (default: 2).
        properties: List of molecular properties to display as color channels.
                   If None, uses all default properties.
        tmap_name: Name for the output TMAP file.
        k_neighbors: Number of neighbors for layout (default: 20).
    """
    if properties is None:
        properties = DEFAULT_PROPERTIES

    # Validate properties
    invalid = set(properties) - set(AVAILABLE_PROPERTIES.keys())
    if invalid:
        raise ValueError(f"Unknown properties: {invalid}. Available: {list(AVAILABLE_PROPERTIES.keys())}")

    # Calculate fingerprints
    fp_arr = _calculate_fingerprint(smiles, fingerprint, fingerprint_bits, fingerprint_radius)

    # Calculate molecular properties
    descriptors_df = calc_properties(smiles, properties)

    # Create TMAP fingerprints
    tm_fingerprint = [tm.VectorUint(fp) for fp in fp_arr]

    # Build LSH Forest
    lf = tm.LSHForest(fingerprint_bits)
    lf.batch_add(tm_fingerprint)
    lf.index()

    # Layout configuration
    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1/30
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 10
    cfg.k = k_neighbors
    cfg.sl_scaling_type = tm.RelativeToAvgLength
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)

    # Create labels
    labels = descriptors_df['smiles'].tolist()

    # Get color data for selected properties
    column_names = [AVAILABLE_PROPERTIES[prop][0] for prop in properties]
    c = [descriptors_df[col].to_numpy() for col in column_names]

    # Plotting
    f = Faerun(
        view="front",
        coords=False,
        title="",
        clear_color="#FFFFFF",
    )

    f.add_scatter(
        tmap_name + "_TMAP",
        {
            "x": x,
            "y": y,
            "c": np.array(c),
            "labels": labels,
        },
        shader="smoothCircle",
        point_scale=2.5,
        max_point_size=20,
        interactive=True,
        series_title=column_names,
        has_legend=True,
        colormap=['rainbow'] * len(properties),
        categorical=[False] * len(properties),
    )

    f.add_tree(tmap_name + "_TMAP_tree", {"from": s, "to": t}, point_helper=tmap_name + "_TMAP")
    f.plot(tmap_name + "_TMAP", template='smiles')


def representative_tmap(
    smiles: list[str],
    cluster_ids: list[int | str] | None = None,
    fingerprint: str = 'mqn',
    fingerprint_bits: int = 1024,
    properties: list[str] | None = None,
    tmap_name: str = 'representative_tmap',
    k_neighbors: int = 21,
) -> None:
    """Create a TMAP for representative molecules using edge list layout.

    Uses scikit-learn's NearestNeighbors for edge list construction and
    TMAP's layout_from_edge_list for positioning. Recommended for primary
    TMAPs showing cluster representatives.

    Args:
        smiles: List of SMILES strings (typically cluster representatives).
        cluster_ids: Optional list of cluster IDs corresponding to each SMILES.
                    If provided, cluster ID will be shown in labels.
        fingerprint: Fingerprint type ('mqn' recommended, or 'morgan').
        fingerprint_bits: Number of bits for fingerprint (default: 1024).
        properties: List of molecular properties to display as color channels.
                   If None, uses all default properties.
        tmap_name: Name for the output TMAP file.
        k_neighbors: Number of neighbors for KNN graph (default: 21).
    """
    if properties is None:
        properties = DEFAULT_PROPERTIES

    # Validate properties
    invalid = set(properties) - set(AVAILABLE_PROPERTIES.keys())
    if invalid:
        raise ValueError(f"Unknown properties: {invalid}. Available: {list(AVAILABLE_PROPERTIES.keys())}")

    # Calculate fingerprints (default MQN for representatives)
    fp_arr = _calculate_fingerprint(smiles, fingerprint, fingerprint_bits)

    # Build edge list using scikit-learn
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean').fit(fp_arr)
    distances, indices = nbrs.kneighbors(fp_arr)

    edge_list = []
    for i in range(len(fp_arr)):
        for neighbor_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
            edge_list.append((i, neighbor_idx, float(dist)))

    # Calculate molecular properties
    descriptors_df = calc_properties(smiles, properties)

    # Layout configuration
    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1/30
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 10
    cfg.k = k_neighbors
    cfg.sl_scaling_type = tm.RelativeToAvgLength
    x, y, s, t, _ = tm.layout_from_edge_list(len(fp_arr), edge_list, cfg)

    # Create labels with optional cluster_id
    labels = []
    for idx, row in descriptors_df.iterrows():
        smi = row['smiles']
        if cluster_ids is not None:
            labels.append(f"{smi}" + "__" + f"Cluster ID: {cluster_ids[idx]}")
        else:
            labels.append(smi)

    # Get color data for selected properties
    column_names = [AVAILABLE_PROPERTIES[prop][0] for prop in properties]
    c = [descriptors_df[col].to_numpy() for col in column_names]

    # Add cluster_id as a color channel if provided
    if cluster_ids is not None:
        # Convert cluster_ids to numeric for coloring
        unique_clusters = list(set(cluster_ids))
        cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
        cluster_colors = np.array([cluster_to_idx[cid] for cid in cluster_ids])
        c.append(cluster_colors)
        column_names.append('Cluster ID')
        properties = list(properties) + ['cluster_id']  # For categorical list

    # Plotting
    f = Faerun(
        view="front",
        coords=False,
        title="",
        clear_color="#FFFFFF",
    )

    # Set categorical for cluster_id if present
    categorical = [False] * (len(properties) - 1) + [True] if cluster_ids is not None else [False] * len(properties)
    colormaps = ['rainbow'] * (len(properties) - 1) + ['tab20'] if cluster_ids is not None else ['rainbow'] * len(properties)

    f.add_scatter(
        tmap_name + "_TMAP",
        {
            "x": x,
            "y": y,
            "c": np.array(c),
            "labels": labels,
        },
        shader="smoothCircle",
        point_scale=2.5,
        max_point_size=20,
        interactive=True,
        series_title=column_names,
        has_legend=True,
        colormap=colormaps,
        categorical=categorical,
    )

    f.add_tree(tmap_name + "_TMAP_tree", {"from": s, "to": t}, point_helper=tmap_name + "_TMAP")
    f.plot(tmap_name + "_TMAP", template='smiles')


def main():
    """CLI entry point for TMAP generation."""
    parser = argparse.ArgumentParser(
        description="Generate TMAP visualizations from SMILES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available properties: {', '.join(AVAILABLE_PROPERTIES.keys())}

Examples:
  chelombus-tmap --smiles molecules.smi
  chelombus-tmap --smiles molecules.smi --fingerprint mqn --properties hac,mw,clogp
  chelombus-tmap --smiles molecules.smi --cluster-file clusters.csv
        """
    )
    parser.add_argument('--smiles', type=str, required=True,
                       help="Path to file containing SMILES (one per line)")
    parser.add_argument('--fingerprint', type=str, default='morgan',
                       choices=['morgan', 'mqn'],
                       help="Fingerprint type (default: morgan)")
    parser.add_argument('--bits', type=int, default=1024,
                       help="Fingerprint bits (default: 1024)")
    parser.add_argument('--radius', type=int, default=2,
                       help="Morgan fingerprint radius (default: 2)")
    parser.add_argument('--properties', type=str, default=None,
                       help=f"Comma-separated properties (default: all). Available: {', '.join(AVAILABLE_PROPERTIES.keys())}")
    parser.add_argument('--output', type=str, default='my_tmap',
                       help="Output TMAP name (default: my_tmap)")
    parser.add_argument('--cluster-file', type=str, default=None,
                       help="Optional CSV/parquet file with 'smiles' and 'cluster_id' columns for representative TMAP")
    parser.add_argument('--representative', action='store_true',
                       help="Use representative_tmap layout (edge list based)")

    args = parser.parse_args()

    # Parse properties
    properties = None
    if args.properties:
        properties = [p.strip() for p in args.properties.split(',')]

    # Load data
    if args.cluster_file:
        # Load from CSV/parquet with cluster_ids
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
            properties=properties,
            tmap_name=args.output,
        )
    else:
        # Load from simple SMILES file
        with open(args.smiles, 'r') as file:
            smiles = [line.strip() for line in file.readlines() if line.strip()]

        if args.representative:
            representative_tmap(
                smiles=smiles,
                fingerprint=args.fingerprint,
                fingerprint_bits=args.bits,
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
