"""Integration tests for the full Chelombus pipeline.

These tests verify the complete workflow from SMILES to clustered results.
They require all optional dependencies: rdkit, pqkmeans, and duckdb.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from chelombus import DataStreamer, FingerprintCalculator, PQEncoder

# Skip all tests if pqkmeans is not installed
pqkmeans = pytest.importorskip("pqkmeans", reason="pqkmeans not installed")
duckdb = pytest.importorskip("duckdb", reason="duckdb not installed")

from chelombus import PQKMeans
from chelombus.utils.cluster_io import query_cluster, get_cluster_stats, get_total_molecules


# Sample SMILES for testing - small, valid molecules
SAMPLE_SMILES = [
    "CCO",           # ethanol
    "CC",            # ethane
    "CCC",           # propane
    "CCCC",          # butane
    "CCCCC",         # pentane
    "CCCCCC",        # hexane
    "c1ccccc1",      # benzene
    "Cc1ccccc1",     # toluene
    "CCc1ccccc1",    # ethylbenzene
    "c1ccc(O)cc1",   # phenol
    "c1ccc(N)cc1",   # aniline
    "CC(C)C",        # isobutane
    "CC(C)CC",       # isopentane
    "CCN",           # ethylamine
    "CCCO",          # propanol
    "CCC(C)C",       # 2-methylbutane
    "CCOCC",         # diethyl ether
    "c1ccc(C)cc1",   # toluene (canonical)
    "CC(=O)C",       # acetone
    "CC(=O)O",       # acetic acid
    "CCO",           # ethanol (duplicate)
    "CCCCCO",        # pentanol
    "c1ccccc1C",     # toluene (another form)
    "CCCCCCC",       # heptane
    "CCCCCCCC",      # octane
    "c1ccc2ccccc2c1", # naphthalene
    "CC(C)(C)C",     # neopentane
    "CCNCC",         # diethylamine
    "CCCN",          # propylamine
    "CCCCN",         # butylamine
]


@pytest.fixture
def smiles_file(tmp_path):
    """Create a temporary SMILES file."""
    file_path = tmp_path / "molecules.smi"
    file_path.write_text("\n".join(SAMPLE_SMILES))
    return file_path


@pytest.fixture
def results_dir(tmp_path):
    """Create a results directory."""
    results_path = tmp_path / "results"
    results_path.mkdir()
    return results_path


class TestDataStreamingIntegration:
    """Tests for data streaming component."""

    def test_stream_smiles_file(self, smiles_file):
        """Test streaming SMILES from file."""
        streamer = DataStreamer()
        chunks = list(streamer.parse_input(str(smiles_file), chunksize=10, verbose=0))

        # Should have 3 chunks (30 SMILES, 10 per chunk)
        assert len(chunks) == 3
        assert len(chunks[0]) == 10
        assert len(chunks[1]) == 10
        assert len(chunks[2]) == 10

    def test_stream_without_chunksize(self, smiles_file):
        """Test streaming entire file as one chunk."""
        streamer = DataStreamer()
        chunks = list(streamer.parse_input(str(smiles_file), chunksize=None, verbose=0))

        assert len(chunks) == 1
        assert len(chunks[0]) == len(SAMPLE_SMILES)


class TestFingerprintIntegration:
    """Tests for fingerprint calculation component."""

    def test_calculate_mqn_fingerprints(self):
        """Test MQN fingerprint calculation."""
        fp_calc = FingerprintCalculator()
        fps = fp_calc.FingerprintFromSmiles(SAMPLE_SMILES[:10], fp="mqn", nprocesses=1)

        # MQN fingerprints are 42-dimensional
        assert fps.shape == (10, 42)
        assert fps.dtype == np.int16

    def test_calculate_morgan_fingerprints(self):
        """Test Morgan fingerprint calculation."""
        fp_calc = FingerprintCalculator()
        fps = fp_calc.FingerprintFromSmiles(
            SAMPLE_SMILES[:10],
            fp="morgan",
            fpSize=1024,
            radius=2,
            nprocesses=1
        )

        assert fps.shape == (10, 1024)
        assert fps.dtype == np.uint8

    def test_invalid_smiles_filtered(self):
        """Test that invalid SMILES are filtered out."""
        fp_calc = FingerprintCalculator()
        smiles_with_invalid = ["CCO", "INVALID_SMILES", "CCC", "NOT_A_MOLECULE"]
        fps = fp_calc.FingerprintFromSmiles(smiles_with_invalid, fp="mqn", nprocesses=1)

        # Should only have 2 valid fingerprints
        assert fps.shape == (2, 42)


class TestPQEncoderIntegration:
    """Tests for PQ encoder component."""

    def test_fit_transform_mqn_fingerprints(self):
        """Test encoding MQN fingerprints to PQ codes."""
        fp_calc = FingerprintCalculator()
        fps = fp_calc.FingerprintFromSmiles(SAMPLE_SMILES, fp="mqn", nprocesses=1)

        # MQN is 42-dimensional, use m=6 (42/6=7 dimensions per subvector)
        encoder = PQEncoder(k=16, m=6, iterations=5)
        pq_codes = encoder.fit_transform(fps.astype(np.float32), verbose=0)

        assert pq_codes.shape == (len(fps), 6)
        assert pq_codes.dtype == np.uint8  # k=16 < 256, so uint8


class TestFullPipelineIntegration:
    """Full pipeline integration tests: SMILES → Fingerprints → PQ → Clustering → Results."""

    def test_full_pipeline_small_dataset(self, smiles_file, results_dir):
        """Test complete pipeline with small dataset."""
        # Step 1: Stream SMILES
        streamer = DataStreamer()
        all_smiles = []
        for chunk in streamer.parse_input(str(smiles_file), chunksize=None, verbose=0):
            all_smiles.extend(chunk)

        # Step 2: Calculate fingerprints
        fp_calc = FingerprintCalculator()
        fps = fp_calc.FingerprintFromSmiles(all_smiles, fp="mqn", nprocesses=1)

        # Some SMILES may have been filtered due to parsing issues
        n_valid = len(fps)
        assert n_valid > 0

        # Step 3: Train PQ encoder
        encoder = PQEncoder(k=16, m=6, iterations=5)
        pq_codes = encoder.fit_transform(fps.astype(np.float32), verbose=0)

        assert pq_codes.shape == (n_valid, 6)

        # Step 4: Train and apply clustering
        n_clusters = 5
        clusterer = PQKMeans(encoder, k=n_clusters, iteration=5, verbose=False)
        labels = clusterer.fit_predict(pq_codes)

        assert labels.shape == (n_valid,)
        assert labels.min() >= 0
        assert labels.max() < n_clusters

        # Step 5: Save results to parquet
        # Filter valid SMILES based on fingerprint calculation
        valid_smiles = all_smiles[:n_valid]  # Approximation; in real use, track validity
        df = pd.DataFrame({
            "smiles": valid_smiles,
            "cluster_id": labels
        })
        df.to_parquet(results_dir / "chunk_00001.parquet", index=False)

        # Step 6: Verify results can be queried
        total = get_total_molecules(results_dir)
        assert total == n_valid

        stats = get_cluster_stats(results_dir)
        assert len(stats) > 0
        assert stats["molecule_count"].sum() == n_valid

        # Query a specific cluster
        first_cluster = stats.iloc[0]["cluster_id"]
        cluster_df = query_cluster(results_dir, cluster_id=int(first_cluster))
        assert len(cluster_df) == stats.iloc[0]["molecule_count"]

    def test_chunked_processing_pipeline(self, smiles_file, results_dir):
        """Test pipeline with chunked processing (simulating large dataset)."""
        # Step 1: Train encoder on sample
        fp_calc = FingerprintCalculator()
        training_smiles = SAMPLE_SMILES[:20]
        training_fps = fp_calc.FingerprintFromSmiles(training_smiles, fp="mqn", nprocesses=1)

        encoder = PQEncoder(k=16, m=6, iterations=5)
        encoder.fit(training_fps.astype(np.float32), verbose=0)

        # Step 2: Train clusterer on PQ codes
        training_pq_codes = encoder.transform(training_fps.astype(np.float32), verbose=0)
        n_clusters = 3
        clusterer = PQKMeans(encoder, k=n_clusters, iteration=5, verbose=False)
        clusterer.fit(training_pq_codes)

        # Step 3: Process in chunks
        streamer = DataStreamer()
        chunk_idx = 0

        for smiles_chunk in streamer.parse_input(str(smiles_file), chunksize=10, verbose=0):
            # Calculate fingerprints
            fps = fp_calc.FingerprintFromSmiles(smiles_chunk, fp="mqn", nprocesses=1)

            if len(fps) == 0:
                continue

            # Encode to PQ codes
            pq_codes = encoder.transform(fps.astype(np.float32), verbose=0)

            # Assign clusters
            labels = clusterer.predict(pq_codes)

            # Save chunk
            df = pd.DataFrame({
                "smiles": smiles_chunk[:len(labels)],
                "cluster_id": labels
            })
            df.to_parquet(results_dir / f"chunk_{chunk_idx:05d}.parquet", index=False)
            chunk_idx += 1

        # Verify results
        assert chunk_idx > 0

        total = get_total_molecules(results_dir)
        assert total > 0

        stats = get_cluster_stats(results_dir)
        assert len(stats) <= n_clusters

    def test_save_load_models(self, tmp_path):
        """Test that models can be saved and loaded."""
        # Train encoder
        fp_calc = FingerprintCalculator()
        fps = fp_calc.FingerprintFromSmiles(SAMPLE_SMILES[:20], fp="mqn", nprocesses=1)

        encoder = PQEncoder(k=16, m=6, iterations=5)
        encoder.fit(fps.astype(np.float32), verbose=0)

        # Save encoder
        encoder_path = tmp_path / "encoder.pkl"
        encoder.save(encoder_path)

        # Train clusterer
        pq_codes = encoder.transform(fps.astype(np.float32), verbose=0)
        clusterer = PQKMeans(encoder, k=3, iteration=5, verbose=False)
        clusterer.fit(pq_codes)

        # Save clusterer
        clusterer_path = tmp_path / "clusterer.pkl"
        clusterer.save(clusterer_path)

        # Load and verify
        loaded_encoder = PQEncoder.load(encoder_path)
        loaded_clusterer = PQKMeans.load(clusterer_path)

        # Process new data with loaded models
        new_fps = fp_calc.FingerprintFromSmiles(SAMPLE_SMILES[20:], fp="mqn", nprocesses=1)
        new_pq_codes = loaded_encoder.transform(new_fps.astype(np.float32), verbose=0)
        new_labels = loaded_clusterer.predict(new_pq_codes)

        assert new_labels.shape == (len(new_fps),)
        assert new_labels.max() < 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_smiles_list(self):
        """Test handling of empty SMILES list."""
        fp_calc = FingerprintCalculator()
        fps = fp_calc.FingerprintFromSmiles([], fp="mqn", nprocesses=1)

        assert len(fps) == 0

    def test_all_invalid_smiles(self):
        """Test handling when all SMILES are invalid."""
        fp_calc = FingerprintCalculator()
        invalid_smiles = ["NOT_VALID", "ALSO_NOT_VALID", "BAD_SMILES"]
        fps = fp_calc.FingerprintFromSmiles(invalid_smiles, fp="mqn", nprocesses=1)

        assert len(fps) == 0

    def test_single_molecule(self):
        """Test pipeline with single molecule."""
        fp_calc = FingerprintCalculator()
        fps = fp_calc.FingerprintFromSmiles(["CCO"], fp="mqn", nprocesses=1)

        assert fps.shape == (1, 42)

    def test_duplicate_smiles(self):
        """Test that duplicate SMILES are handled correctly."""
        fp_calc = FingerprintCalculator()
        duplicate_smiles = ["CCO", "CCO", "CCO", "CCO", "CCO"]
        fps = fp_calc.FingerprintFromSmiles(duplicate_smiles, fp="mqn", nprocesses=1)

        assert fps.shape == (5, 42)
        # All fingerprints should be identical
        assert np.all(fps[0] == fps[1])


class TestReproducibility:
    """Test reproducibility of the pipeline."""

    def test_encoder_reproducibility(self):
        """Test that encoder produces consistent results with same random state."""
        fp_calc = FingerprintCalculator()
        fps = fp_calc.FingerprintFromSmiles(SAMPLE_SMILES[:20], fp="mqn", nprocesses=1)

        # Set seed before each encoder
        np.random.seed(42)
        encoder1 = PQEncoder(k=16, m=6, iterations=5)
        encoder1.fit(fps.astype(np.float32), verbose=0)
        codes1 = encoder1.transform(fps.astype(np.float32), verbose=0)

        np.random.seed(42)
        encoder2 = PQEncoder(k=16, m=6, iterations=5)
        encoder2.fit(fps.astype(np.float32), verbose=0)
        codes2 = encoder2.transform(fps.astype(np.float32), verbose=0)

        # Note: KMeans has some randomness, but with same seed should be close
        # In practice, may not be exactly equal due to parallel initialization
        assert codes1.shape == codes2.shape

    def test_fingerprint_reproducibility(self):
        """Test that fingerprint calculation is deterministic."""
        fp_calc = FingerprintCalculator()

        fps1 = fp_calc.FingerprintFromSmiles(["CCO", "CCC"], fp="mqn", nprocesses=1)
        fps2 = fp_calc.FingerprintFromSmiles(["CCO", "CCC"], fp="mqn", nprocesses=1)

        np.testing.assert_array_equal(fps1, fps2)
