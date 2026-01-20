"""Cluster I/O utilities using DuckDB for efficient querying.

This module provides functions for querying and exporting clustered molecular data
stored in chunked parquet files. It uses DuckDB for efficient SQL-based queries
across multiple files without loading everything into memory.

Architecture:
    During pipeline.transform(), data is saved as chunked parquet files:
        results/chunk_00001.parquet  (smiles, cluster_id)
        results/chunk_00002.parquet  (smiles, cluster_id)
        ...

    These functions query across all chunks using DuckDB's glob pattern support,
    enabling efficient filtering and export without loading all data into RAM.

Example:
    >>> from chelombus.utils.cluster_io import query_cluster, export_all_clusters
    >>>
    >>> # Get molecules from cluster 42
    >>> df = query_cluster('results/', cluster_id=42)
    >>>
    >>> # Export all clusters (for HPC/SLURM processing)
    >>> stats = export_all_clusters('results/', 'clusters/')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _check_duckdb_available() -> None:
    """Check if duckdb is installed, raise ImportError with helpful message if not."""
    try:
        import duckdb  # noqa: F401
    except ImportError:
        raise ImportError(
            "duckdb is required for cluster I/O operations. "
            "Install it with: pip install chelombus[io] or pip install duckdb"
        )


def _validate_results_dir(results_dir: Union[str, Path]) -> Path:
    """Validate that results directory exists and contains parquet files.

    Args:
        results_dir: Path to directory containing chunked parquet files.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If directory doesn't exist.
        ValueError: If no parquet files found in directory.
    """
    path = Path(results_dir)
    if not path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not path.is_dir():
        raise ValueError(f"Expected directory, got file: {results_dir}")

    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in: {results_dir}")

    return path


def _quote_identifier(identifier: str) -> str:
    """Quote SQL identifiers for DuckDB, escaping internal quotes."""
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier
    return f'"{identifier.replace("\"", "\"\"")}"'


def query_cluster(
    results_dir: Union[str, Path],
    cluster_id: int,
    cluster_column: str = "cluster_id",
    columns: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """Query all molecules from a specific cluster.

    Scans all chunked parquet files in results_dir and returns molecules
    belonging to the specified cluster as a pandas DataFrame.

    Args:
        results_dir: Path to directory containing chunked parquet files
            (output from pipeline.transform()).
        cluster_id: The cluster ID to query.
        cluster_column: Name of the column containing cluster IDs.
        columns: Optional list of columns to return. Default returns all columns.
            Common columns: ['smiles', 'cluster_id'].

    Returns:
        DataFrame containing all molecules from the specified cluster.

    Raises:
        ImportError: If duckdb is not installed.
        FileNotFoundError: If results_dir doesn't exist.
        ValueError: If no parquet files found or cluster_id doesn't exist.

    Example:
        >>> df = query_cluster('results/', cluster_id=42)
        >>> print(f"Cluster 42 has {len(df)} molecules")
        >>> print(df['smiles'].head())
    """
    _check_duckdb_available()
    import duckdb

    path = _validate_results_dir(results_dir)
    glob_pattern = str(path / "*.parquet")
    quoted_cluster_column = _quote_identifier(cluster_column)

    # Build column selection
    if columns:
        col_str = ", ".join(_quote_identifier(col) for col in columns)
    else:
        col_str = "*"

    query = f"""
        SELECT {col_str}
        FROM '{glob_pattern}'
        WHERE {quoted_cluster_column} = {cluster_id}
    """

    result = duckdb.query(query).df()

    if result.empty:
        logger.warning(f"No molecules found for cluster_id={cluster_id}")

    return result


def export_cluster(
    results_dir: Union[str, Path],
    cluster_id: int,
    output_path: Union[str, Path],
    cluster_column: str = "cluster_id",
    columns: Optional[List[str]] = None,
) -> int:
    """Export a single cluster to a file.

    Queries all molecules from a specific cluster and writes them to a file.
    Supports parquet and CSV output formats.

    Args:
        results_dir: Path to directory containing chunked parquet files.
        cluster_id: The cluster ID to export.
        output_path: Output file path. Must end with .parquet or .csv.
        cluster_column: Name of the column containing cluster IDs.
        columns: Optional list of columns to export. Default exports all columns.

    Returns:
        Number of molecules exported.

    Raises:
        ImportError: If duckdb is not installed.
        ValueError: If output format is not supported.

    Example:
        >>> count = export_cluster('results/', 42, 'cluster_42.parquet')
        >>> print(f"Exported {count} molecules")
    """
    _check_duckdb_available()
    import duckdb

    path = _validate_results_dir(results_dir)
    output_path = Path(output_path)
    glob_pattern = str(path / "*.parquet")
    quoted_cluster_column = _quote_identifier(cluster_column)

    # Build column selection
    col_str = ", ".join(columns) if columns else "*"

    # First get the count
    count_query = f"""
        SELECT COUNT(*) as cnt
        FROM '{glob_pattern}'
        WHERE {quoted_cluster_column} = {cluster_id}
    """
    count = duckdb.query(count_query).fetchone()[0] # type: ignore

    if count == 0:
        logger.warning(f"No molecules found for cluster_id={cluster_id}")
        return 0

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export based on format
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        export_query = f"""
            COPY (
                SELECT {col_str}
                FROM '{glob_pattern}'
                WHERE {quoted_cluster_column} = {cluster_id}
            ) TO '{output_path}' (FORMAT PARQUET)
        """
    elif suffix == ".csv":
        export_query = f"""
            COPY (
                SELECT {col_str}
                FROM '{glob_pattern}'
                WHERE {quoted_cluster_column} = {cluster_id}
            ) TO '{output_path}' (FORMAT CSV, HEADER)
        """
    elif suffix in (".smi", ".txt"):
        # For SMILES files, export only the smiles column without header
        export_query = f"""
            COPY (
                SELECT smiles
                FROM '{glob_pattern}'
                WHERE {quoted_cluster_column} = {cluster_id}
            ) TO '{output_path}' (FORMAT CSV, HEADER FALSE)
        """
    else:
        raise ValueError(
            f"Unsupported output format: {suffix}. "
            "Use .parquet, .csv, .smi, or .txt"
        )

    duckdb.query(export_query)
    logger.info(f"Exported {count} molecules to {output_path}")

    return count


def export_all_clusters(
    results_dir: Union[str, Path],
    output_dir: Union[str, Path],
    format: str = "parquet",
    batch_size: int = 1000,
) -> dict[int, int]:
    """Export all clusters to individual files using partitioned writes.

    This function exports all clusters to separate files in a single streaming
    pass through the data, using DuckDB's PARTITION_BY functionality for
    memory efficiency.

    Output structure (Hive-style partitioning):
        output_dir/
        ├── cluster_id=0/data_0.parquet
        ├── cluster_id=1/data_0.parquet
        ├── cluster_id=2/data_0.parquet
        ...
        └── cluster_id=99999/data_0.parquet

    This structure is compatible with:
    - DuckDB queries: SELECT * FROM 'clusters/*/*.parquet' WHERE cluster_id = 42
    - Spark/pandas: Reads partition column automatically
    - SLURM arrays: Process cluster_id=$SLURM_ARRAY_TASK_ID

    Args:
        results_dir: Path to directory containing chunked parquet files.
        output_dir: Output directory for partitioned cluster files.
        format: Output format, either 'parquet' or 'csv'. Default 'parquet'.
        batch_size: Number of clusters to process per batch for progress reporting.
            Does not affect memory usage (DuckDB handles streaming internally).

    Returns:
        Dictionary mapping cluster_id to molecule count.

    Raises:
        ImportError: If duckdb is not installed.
        ValueError: If format is not supported.

    Example:
        >>> stats = export_all_clusters('results/', 'clusters/')
        >>> print(f"Exported {len(stats)} clusters")
        >>> print(f"Largest cluster: {max(stats.values())} molecules")

    Note:
        For HPC/SLURM usage, you can process clusters in parallel:

        ```bash
        # SLURM array job
        #SBATCH --array=0-99999

        chelombus visualize \\
            --input clusters/cluster_id=$SLURM_ARRAY_TASK_ID/ \\
            --output tmaps/cluster_$SLURM_ARRAY_TASK_ID.html
        ```
    """
    _check_duckdb_available()
    import duckdb
    from tqdm import tqdm

    path = _validate_results_dir(results_dir)
    output_path = Path(output_dir)
    glob_pattern = str(path / "*.parquet")

    if format not in ("parquet", "csv"):
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting all clusters from {results_dir} to {output_dir}")

    # Use DuckDB's partitioned write - single pass through all data
    # This is memory-efficient as DuckDB streams the data
    format_str = "PARQUET" if format == "parquet" else "CSV"

    export_query = f"""
        COPY (
            SELECT smiles, cluster_id
            FROM '{glob_pattern}'
        ) TO '{output_path}' (
            FORMAT {format_str},
            PARTITION_BY (cluster_id),
            OVERWRITE_OR_IGNORE
        )
    """

    logger.info("Starting partitioned export (streaming, memory-efficient)...")
    duckdb.query(export_query)

    # Get cluster statistics after export
    logger.info("Calculating cluster statistics...")
    stats = get_cluster_stats(results_dir)

    # Convert to dictionary
    result = dict(zip(stats["cluster_id"], stats["molecule_count"]))

    logger.info(
        f"Exported {len(result)} clusters, "
        f"total {sum(result.values()):,} molecules"
    )

    return result


def get_cluster_stats(
    results_dir: Union[str, Path],
    column: str = 'cluster_id',
) -> "pd.DataFrame":
    """Get statistics for all clusters.

    Returns a DataFrame with cluster IDs and molecule counts, sorted by cluster_id.

    Args:
        results_dir: Path to directory containing chunked parquet files.
        column: Name for the column containing cluster ID defaults to 'cluster_id'

    Returns:
        DataFrame with columns ['cluster_id', 'molecule_count'].

    Example:
        >>> stats = get_cluster_stats('results/')
        >>> print(f"Number of clusters: {len(stats)}")
        >>> print(f"Average cluster size: {stats['molecule_count'].mean():.1f}")
        >>> print(f"Largest cluster: {stats['molecule_count'].max()}")
    """
    _check_duckdb_available()
    import duckdb

    path = _validate_results_dir(results_dir)
    glob_pattern = str(path / "*.parquet")
    quoted_column = _quote_identifier(column)

    query = f"""
        SELECT
            {quoted_column} as cluster_id,
            COUNT(*) as molecule_count
        FROM '{glob_pattern}'
        GROUP BY {quoted_column}
        ORDER BY {quoted_column}
    """

    return duckdb.query(query).df()


def get_cluster_ids(
    results_dir: Union[str, Path],
) -> List[int]:
    """Get list of all cluster IDs in the results.

    Args:
        results_dir: Path to directory containing chunked parquet files.

    Returns:
        Sorted list of cluster IDs.

    Example:
        >>> cluster_ids = get_cluster_ids('results/')
        >>> print(f"Clusters range from {min(cluster_ids)} to {max(cluster_ids)}")
    """
    _check_duckdb_available()
    import duckdb

    path = _validate_results_dir(results_dir)
    glob_pattern = str(path / "*.parquet")

    query = f"""
        SELECT DISTINCT cluster_id
        FROM '{glob_pattern}'
        ORDER BY cluster_id
    """

    result = duckdb.query(query).fetchall()
    return [row[0] for row in result]


def get_total_molecules(
    results_dir: Union[str, Path],
) -> int:
    """Get total number of molecules across all clusters.

    Args:
        results_dir: Path to directory containing chunked parquet files.

    Returns:
        Total molecule count.
    """
    _check_duckdb_available()
    import duckdb

    path = _validate_results_dir(results_dir)
    glob_pattern = str(path / "*.parquet")

    query = f"SELECT COUNT(*) FROM '{glob_pattern}'"
    return duckdb.query(query).fetchone()[0] # type: ignore


def query_clusters_batch(
    results_dir: Union[str, Path],
    cluster_ids: List[int],
    columns: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """Query molecules from multiple clusters at once.

    More efficient than calling query_cluster() multiple times when you need
    data from several clusters.

    Args:
        results_dir: Path to directory containing chunked parquet files.
        cluster_ids: List of cluster IDs to query.
        columns: Optional list of columns to return.

    Returns:
        DataFrame containing molecules from all specified clusters.

    Example:
        >>> df = query_clusters_batch('results/', [1, 2, 3, 4, 5])
        >>> cluster_counts = df.groupby('cluster_id').size()
    """
    _check_duckdb_available()
    import duckdb

    path = _validate_results_dir(results_dir)
    glob_pattern = str(path / "*.parquet")

    col_str = ", ".join(columns) if columns else "*"
    ids_str = ", ".join(str(cid) for cid in cluster_ids)

    query = f"""
        SELECT {col_str}
        FROM '{glob_pattern}'
        WHERE cluster_id IN ({ids_str})
    """

    return duckdb.query(query).df()


def sample_from_cluster(
    results_dir: Union[str, Path],
    cluster_id: int,
    n: int = 100,
    random_state: Optional[int] = None,
) -> "pd.DataFrame":
    """Get a random sample of molecules from a cluster.

    Useful for previewing cluster contents without loading all molecules.

    Args:
        results_dir: Path to directory containing chunked parquet files.
        cluster_id: The cluster ID to sample from.
        n: Number of molecules to sample.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with sampled molecules.

    Example:
        >>> sample = sample_from_cluster('results/', cluster_id=42, n=10)
        >>> print(sample['smiles'].tolist())
    """
    _check_duckdb_available()
    import duckdb

    path = _validate_results_dir(results_dir)
    glob_pattern = str(path / "*.parquet")

    # Use setseed() for reproducibility when random_state is provided
    if random_state is not None:
        duckdb.query(f"SELECT setseed({random_state / 2**31})")

    # Use ORDER BY RANDOM() for sampling, which is more portable
    query = f"""
        SELECT *
        FROM '{glob_pattern}'
        WHERE cluster_id = {cluster_id}
        ORDER BY RANDOM()
        LIMIT {n}
    """

    return duckdb.query(query).df()
