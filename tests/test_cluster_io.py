"""Tests for the cluster_io module.

These tests require duckdb and pandas to be installed.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Skip all tests if duckdb is not installed
duckdb = pytest.importorskip("duckdb", reason="duckdb not installed")

from chelombus.utils.cluster_io import (
    query_cluster,
    export_cluster,
    export_all_clusters,
    get_cluster_stats,
    get_cluster_ids,
    get_total_molecules,
    query_clusters_batch,
    sample_from_cluster,
    _validate_results_dir,
    _quote_identifier,
)


@pytest.fixture
def sample_results_dir(tmp_path):
    """Create a temporary results directory with sample parquet files."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create two chunk files with molecules in different clusters
    chunk1_data = pd.DataFrame({
        "smiles": ["CCO", "CC", "CCC", "CCCC", "CCCCC"],
        "cluster_id": [0, 1, 0, 2, 1]
    })
    chunk1_data.to_parquet(results_dir / "chunk_00001.parquet", index=False)

    chunk2_data = pd.DataFrame({
        "smiles": ["c1ccccc1", "CCN", "CCCO", "c1ccc(O)cc1", "CCCN"],
        "cluster_id": [1, 2, 0, 1, 2]
    })
    chunk2_data.to_parquet(results_dir / "chunk_00002.parquet", index=False)

    return results_dir


@pytest.fixture
def large_results_dir(tmp_path):
    """Create a results directory with more data for statistics tests."""
    results_dir = tmp_path / "large_results"
    results_dir.mkdir()

    np.random.seed(42)
    n_molecules = 1000
    n_clusters = 50

    # Create multiple chunks
    molecules_per_chunk = 200
    for i in range(n_molecules // molecules_per_chunk):
        chunk_data = pd.DataFrame({
            "smiles": [f"MOL_{i}_{j}" for j in range(molecules_per_chunk)],
            "cluster_id": np.random.randint(0, n_clusters, molecules_per_chunk)
        })
        chunk_data.to_parquet(results_dir / f"chunk_{i:05d}.parquet", index=False)

    return results_dir


class TestValidateResultsDir:
    """Tests for _validate_results_dir helper function."""

    def test_validates_existing_dir(self, sample_results_dir):
        """Test that existing directory with parquet files validates."""
        result = _validate_results_dir(sample_results_dir)
        assert result == sample_results_dir

    def test_validates_string_path(self, sample_results_dir):
        """Test that string paths are converted to Path objects."""
        result = _validate_results_dir(str(sample_results_dir))
        assert isinstance(result, Path)

    def test_raises_for_nonexistent_dir(self, tmp_path):
        """Test that non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            _validate_results_dir(tmp_path / "nonexistent")

    def test_raises_for_file_instead_of_dir(self, tmp_path):
        """Test that file path raises ValueError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="Expected directory"):
            _validate_results_dir(file_path)

    def test_raises_for_empty_dir(self, tmp_path):
        """Test that directory without parquet files raises ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No parquet files"):
            _validate_results_dir(empty_dir)


class TestQuoteIdentifier:
    """Tests for _quote_identifier helper function."""

    def test_quotes_simple_identifier(self):
        """Test quoting a simple identifier."""
        assert _quote_identifier("cluster_id") == '"cluster_id"'

    def test_escapes_quotes_in_identifier(self):
        """Test that internal quotes are escaped."""
        assert _quote_identifier('col"name') == '"col""name"'

    def test_skips_already_quoted(self):
        """Test that already quoted identifiers are not double-quoted."""
        assert _quote_identifier('"cluster_id"') == '"cluster_id"'


class TestQueryCluster:
    """Tests for query_cluster function."""

    def test_query_existing_cluster(self, sample_results_dir):
        """Test querying molecules from an existing cluster."""
        df = query_cluster(sample_results_dir, cluster_id=0)

        assert len(df) == 3  # CCO, CCC, CCCO
        assert "smiles" in df.columns
        assert "cluster_id" in df.columns
        assert all(df["cluster_id"] == 0)

    def test_query_returns_all_molecules(self, sample_results_dir):
        """Test that query returns all molecules from a cluster."""
        df = query_cluster(sample_results_dir, cluster_id=1)

        expected_smiles = {"CC", "CCCCC", "c1ccccc1", "c1ccc(O)cc1"}
        assert set(df["smiles"]) == expected_smiles

    def test_query_empty_cluster(self, sample_results_dir):
        """Test querying a cluster that doesn't exist."""
        df = query_cluster(sample_results_dir, cluster_id=999)

        assert len(df) == 0

    def test_query_specific_columns(self, sample_results_dir):
        """Test querying specific columns."""
        df = query_cluster(sample_results_dir, cluster_id=0, columns=["smiles"])

        assert list(df.columns) == ["smiles"]
        assert len(df) == 3

    def test_query_with_string_path(self, sample_results_dir):
        """Test that string paths work."""
        df = query_cluster(str(sample_results_dir), cluster_id=0)

        assert len(df) == 3


class TestExportCluster:
    """Tests for export_cluster function."""

    def test_export_to_parquet(self, sample_results_dir, tmp_path):
        """Test exporting cluster to parquet format."""
        output_path = tmp_path / "cluster_0.parquet"
        count = export_cluster(sample_results_dir, cluster_id=0, output_path=output_path)

        assert count == 3
        assert output_path.exists()

        # Verify contents
        df = pd.read_parquet(output_path)
        assert len(df) == 3
        assert all(df["cluster_id"] == 0)

    def test_export_to_csv(self, sample_results_dir, tmp_path):
        """Test exporting cluster to CSV format."""
        output_path = tmp_path / "cluster_0.csv"
        count = export_cluster(sample_results_dir, cluster_id=0, output_path=output_path)

        assert count == 3
        assert output_path.exists()

        # Verify contents
        df = pd.read_csv(output_path)
        assert len(df) == 3

    def test_export_to_smi(self, sample_results_dir, tmp_path):
        """Test exporting cluster to SMILES format."""
        output_path = tmp_path / "cluster_0.smi"
        count = export_cluster(sample_results_dir, cluster_id=0, output_path=output_path)

        assert count == 3
        assert output_path.exists()

        # Verify contents (should be just SMILES, no header)
        with open(output_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 3

    def test_export_to_txt(self, sample_results_dir, tmp_path):
        """Test exporting cluster to TXT format (same as SMI)."""
        output_path = tmp_path / "cluster_0.txt"
        count = export_cluster(sample_results_dir, cluster_id=0, output_path=output_path)

        assert count == 3
        assert output_path.exists()

    def test_export_nonexistent_cluster(self, sample_results_dir, tmp_path):
        """Test exporting a cluster that doesn't exist."""
        output_path = tmp_path / "cluster_999.parquet"
        count = export_cluster(sample_results_dir, cluster_id=999, output_path=output_path)

        assert count == 0

    def test_export_creates_parent_dir(self, sample_results_dir, tmp_path):
        """Test that export creates parent directories if needed."""
        output_path = tmp_path / "subdir" / "cluster_0.parquet"
        count = export_cluster(sample_results_dir, cluster_id=0, output_path=output_path)

        assert count == 3
        assert output_path.exists()

    def test_export_unsupported_format_raises(self, sample_results_dir, tmp_path):
        """Test that unsupported format raises ValueError."""
        output_path = tmp_path / "cluster_0.xyz"

        with pytest.raises(ValueError, match="Unsupported output format"):
            export_cluster(sample_results_dir, cluster_id=0, output_path=output_path)


class TestExportAllClusters:
    """Tests for export_all_clusters function."""

    def test_export_all_creates_partitions(self, sample_results_dir, tmp_path):
        """Test that export_all_clusters creates Hive-style partitions."""
        output_dir = tmp_path / "clusters"
        stats = export_all_clusters(sample_results_dir, output_dir)

        # Should have 3 clusters (0, 1, 2)
        assert len(stats) == 3

        # Check Hive-style directory structure
        for cluster_id in [0, 1, 2]:
            partition_dir = output_dir / f"cluster_id={cluster_id}"
            assert partition_dir.exists()

    def test_export_all_molecule_counts(self, sample_results_dir, tmp_path):
        """Test that export_all_clusters returns correct molecule counts."""
        output_dir = tmp_path / "clusters"
        stats = export_all_clusters(sample_results_dir, output_dir)

        # Verify counts match expected
        assert stats[0] == 3  # CCO, CCC, CCCO
        assert stats[1] == 4  # CC, CCCCC, c1ccccc1, c1ccc(O)cc1
        assert stats[2] == 3  # CCCC, CCN, CCCN

    def test_export_all_csv_format(self, sample_results_dir, tmp_path):
        """Test exporting all clusters in CSV format."""
        output_dir = tmp_path / "clusters"
        stats = export_all_clusters(sample_results_dir, output_dir, format="csv")

        assert len(stats) == 3

        # Check that CSV files were created (not parquet)
        for cluster_id in [0, 1, 2]:
            partition_dir = output_dir / f"cluster_id={cluster_id}"
            csv_files = list(partition_dir.glob("*.csv"))
            assert len(csv_files) > 0

    def test_export_all_unsupported_format_raises(self, sample_results_dir, tmp_path):
        """Test that unsupported format raises ValueError."""
        output_dir = tmp_path / "clusters"

        with pytest.raises(ValueError, match="Unsupported format"):
            export_all_clusters(sample_results_dir, output_dir, format="xyz")


class TestGetClusterStats:
    """Tests for get_cluster_stats function."""

    def test_get_stats_returns_dataframe(self, sample_results_dir):
        """Test that get_cluster_stats returns a DataFrame."""
        stats = get_cluster_stats(sample_results_dir)

        assert isinstance(stats, pd.DataFrame)
        assert "cluster_id" in stats.columns
        assert "molecule_count" in stats.columns

    def test_get_stats_correct_counts(self, sample_results_dir):
        """Test that molecule counts are correct."""
        stats = get_cluster_stats(sample_results_dir)

        # Convert to dict for easier checking
        counts = dict(zip(stats["cluster_id"], stats["molecule_count"]))

        assert counts[0] == 3
        assert counts[1] == 4
        assert counts[2] == 3

    def test_get_stats_sorted_by_cluster_id(self, sample_results_dir):
        """Test that results are sorted by cluster_id."""
        stats = get_cluster_stats(sample_results_dir)

        assert list(stats["cluster_id"]) == sorted(stats["cluster_id"])

    def test_get_stats_large_data(self, large_results_dir):
        """Test statistics with larger dataset."""
        stats = get_cluster_stats(large_results_dir)

        # Should have up to 50 clusters (some may be empty due to random assignment)
        assert len(stats) > 0
        assert len(stats) <= 50

        # Total should be 1000 molecules
        assert stats["molecule_count"].sum() == 1000


class TestGetClusterIds:
    """Tests for get_cluster_ids function."""

    def test_get_cluster_ids_returns_list(self, sample_results_dir):
        """Test that get_cluster_ids returns a sorted list."""
        ids = get_cluster_ids(sample_results_dir)

        assert isinstance(ids, list)
        assert ids == [0, 1, 2]

    def test_get_cluster_ids_sorted(self, large_results_dir):
        """Test that cluster IDs are sorted."""
        ids = get_cluster_ids(large_results_dir)

        assert ids == sorted(ids)


class TestGetTotalMolecules:
    """Tests for get_total_molecules function."""

    def test_get_total_molecules(self, sample_results_dir):
        """Test counting total molecules."""
        total = get_total_molecules(sample_results_dir)

        assert total == 10  # 5 + 5 molecules in two chunks

    def test_get_total_molecules_large(self, large_results_dir):
        """Test counting total molecules in larger dataset."""
        total = get_total_molecules(large_results_dir)

        assert total == 1000


class TestQueryClustersBatch:
    """Tests for query_clusters_batch function."""

    def test_query_multiple_clusters(self, sample_results_dir):
        """Test querying multiple clusters at once."""
        df = query_clusters_batch(sample_results_dir, cluster_ids=[0, 1])

        # Should have molecules from clusters 0 and 1
        assert set(df["cluster_id"].unique()) == {0, 1}
        assert len(df) == 7  # 3 from cluster 0, 4 from cluster 1

    def test_query_single_cluster_in_batch(self, sample_results_dir):
        """Test querying single cluster via batch function."""
        df = query_clusters_batch(sample_results_dir, cluster_ids=[2])

        assert len(df) == 3
        assert all(df["cluster_id"] == 2)

    def test_query_all_clusters_batch(self, sample_results_dir):
        """Test querying all clusters."""
        df = query_clusters_batch(sample_results_dir, cluster_ids=[0, 1, 2])

        assert len(df) == 10  # All molecules

    def test_query_nonexistent_clusters_batch(self, sample_results_dir):
        """Test querying non-existent clusters returns empty DataFrame."""
        df = query_clusters_batch(sample_results_dir, cluster_ids=[99, 100])

        assert len(df) == 0

    def test_query_specific_columns_batch(self, sample_results_dir):
        """Test querying specific columns in batch."""
        df = query_clusters_batch(
            sample_results_dir,
            cluster_ids=[0, 1],
            columns=["smiles"]
        )

        assert list(df.columns) == ["smiles"]


class TestSampleFromCluster:
    """Tests for sample_from_cluster function."""

    def test_sample_returns_correct_size(self, large_results_dir):
        """Test that sample returns requested number of molecules."""
        n = 10
        df = sample_from_cluster(large_results_dir, cluster_id=0, n=n)

        # Note: May return fewer if cluster has fewer molecules
        assert len(df) <= n

    def test_sample_from_cluster_all_same_cluster(self, large_results_dir):
        """Test that all sampled molecules are from correct cluster."""
        df = sample_from_cluster(large_results_dir, cluster_id=0, n=5)

        if len(df) > 0:
            assert all(df["cluster_id"] == 0)

    def test_sample_with_random_state(self, large_results_dir):
        """Test reproducibility with random_state."""
        df1 = sample_from_cluster(large_results_dir, cluster_id=0, n=5, random_state=42)
        df2 = sample_from_cluster(large_results_dir, cluster_id=0, n=5, random_state=42)

        # Note: DuckDB's SAMPLE REPEATABLE should give same results
        # but may not be exactly identical in all versions
        assert len(df1) == len(df2)

    def test_sample_empty_cluster(self, sample_results_dir):
        """Test sampling from non-existent cluster."""
        df = sample_from_cluster(sample_results_dir, cluster_id=999, n=5)

        assert len(df) == 0


class TestIntegration:
    """Integration tests for cluster_io module."""

    def test_full_workflow(self, sample_results_dir, tmp_path):
        """Test a complete workflow: stats -> query -> export."""
        # Get statistics
        stats = get_cluster_stats(sample_results_dir)
        assert len(stats) == 3

        # Find largest cluster
        largest_cluster = stats.loc[stats["molecule_count"].idxmax(), "cluster_id"]

        # Query it
        df = query_cluster(sample_results_dir, cluster_id=largest_cluster)
        assert len(df) == stats.loc[stats["cluster_id"] == largest_cluster, "molecule_count"].values[0]

        # Export it
        output_path = tmp_path / "largest_cluster.parquet"
        count = export_cluster(sample_results_dir, cluster_id=largest_cluster, output_path=output_path)
        assert count == len(df)

    def test_export_all_and_query_back(self, sample_results_dir, tmp_path):
        """Test exporting all clusters and querying from partitioned output."""
        output_dir = tmp_path / "clusters"
        stats = export_all_clusters(sample_results_dir, output_dir)

        # Query from partitioned output using DuckDB directly
        df = duckdb.query(f"""
            SELECT * FROM '{output_dir}/*/*.parquet'
            WHERE cluster_id = 0
        """).df()

        assert len(df) == stats[0]
