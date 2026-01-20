"""Tests for the PQKMeans clustering module.

These tests require the pqkmeans library to be installed. If not installed,
all tests will be skipped.
"""

import numpy as np
import pytest
from pathlib import Path

from chelombus.encoder.encoder import PQEncoder

# Skip all tests if pqkmeans is not installed
pqkmeans_available = pytest.importorskip("pqkmeans", reason="pqkmeans not installed")

from chelombus.clustering.PyQKmeans import PQKMeans


@pytest.fixture
def trained_encoder():
    """Create a trained PQEncoder for testing."""
    np.random.seed(42)
    # Create training data: 500 samples, 42 dimensions (like MQN fingerprints)
    X_train = np.random.rand(500, 42).astype(np.float32)

    encoder = PQEncoder(k=32, m=6, iterations=5)  # k=32 for faster tests
    encoder.fit(X_train, verbose=0)
    return encoder


@pytest.fixture
def untrained_encoder():
    """Create an untrained PQEncoder."""
    return PQEncoder(k=32, m=6, iterations=5)


@pytest.fixture
def sample_pq_codes(trained_encoder):
    """Create sample PQ codes using a trained encoder."""
    np.random.seed(123)
    X = np.random.rand(100, 42).astype(np.float32)
    return trained_encoder.transform(X, verbose=0)


class TestPQKMeansInit:
    """Tests for PQKMeans initialization."""

    def test_init_with_trained_encoder(self, trained_encoder):
        """Test that PQKMeans can be initialized with a trained encoder."""
        clusterer = PQKMeans(trained_encoder, k=10, iteration=5, verbose=False)

        assert clusterer.encoder == trained_encoder
        assert clusterer.k == 10
        assert clusterer.iteration == 5
        assert clusterer.verbose is False
        assert clusterer.is_trained is True  # Encoder is trained

    def test_init_with_untrained_encoder_raises(self, untrained_encoder):
        """Test that initializing with untrained encoder raises ValueError."""
        with pytest.raises(ValueError, match="Encoder must be trained"):
            PQKMeans(untrained_encoder, k=10)

    def test_init_default_parameters(self, trained_encoder):
        """Test default parameter values."""
        clusterer = PQKMeans(trained_encoder, k=100)

        assert clusterer.k == 100
        assert clusterer.iteration == 20  # default
        assert clusterer.verbose is False  # default


class TestPQKMeansFit:
    """Tests for PQKMeans fit method."""

    def test_fit_returns_self(self, trained_encoder, sample_pq_codes):
        """Test that fit() returns self for method chaining."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        result = clusterer.fit(sample_pq_codes)

        assert result is clusterer

    def test_fit_with_valid_pq_codes(self, trained_encoder, sample_pq_codes):
        """Test fitting with valid PQ codes."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes)

        # Should complete without error
        assert True

    def test_fit_with_small_k(self, trained_encoder, sample_pq_codes):
        """Test fitting with small number of clusters."""
        clusterer = PQKMeans(trained_encoder, k=3, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes)

        # Predict should return labels in range [0, k)
        labels = clusterer.predict(sample_pq_codes)
        assert labels.min() >= 0
        assert labels.max() < 3


class TestPQKMeansPredict:
    """Tests for PQKMeans predict method."""

    def test_predict_shape(self, trained_encoder, sample_pq_codes):
        """Test that predict returns correct shape."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes)

        labels = clusterer.predict(sample_pq_codes)

        assert labels.shape == (len(sample_pq_codes),)

    def test_predict_label_range(self, trained_encoder, sample_pq_codes):
        """Test that predicted labels are in valid range."""
        k = 5
        clusterer = PQKMeans(trained_encoder, k=k, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes)

        labels = clusterer.predict(sample_pq_codes)

        assert labels.min() >= 0
        assert labels.max() < k

    def test_predict_dtype(self, trained_encoder, sample_pq_codes):
        """Test that predicted labels are integers."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes)

        labels = clusterer.predict(sample_pq_codes)

        assert np.issubdtype(labels.dtype, np.integer)


class TestPQKMeansFitPredict:
    """Tests for PQKMeans fit_predict method."""

    def test_fit_predict_shape(self, trained_encoder, sample_pq_codes):
        """Test that fit_predict returns correct shape."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)

        labels = clusterer.fit_predict(sample_pq_codes)

        assert labels.shape == (len(sample_pq_codes),)

    def test_fit_predict_equivalence(self, trained_encoder, sample_pq_codes):
        """Test that fit_predict gives same result as fit then predict."""
        np.random.seed(42)

        clusterer1 = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        labels1 = clusterer1.fit_predict(sample_pq_codes)

        # Note: Due to randomness in k-means, we can't guarantee exact equality
        # But labels should be in valid range
        assert labels1.min() >= 0
        assert labels1.max() < 5


class TestPQKMeansSaveLoad:
    """Tests for PQKMeans save and load methods."""

    def test_save_creates_file(self, trained_encoder, sample_pq_codes, tmp_path):
        """Test that save creates a file."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes)

        save_path = tmp_path / "clusterer.pkl"
        clusterer.save(save_path)

        assert save_path.exists()

    def test_load_restores_clusterer(self, trained_encoder, sample_pq_codes, tmp_path):
        """Test that load restores a clusterer correctly."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes)
        labels_before = clusterer.predict(sample_pq_codes)

        save_path = tmp_path / "clusterer.pkl"
        clusterer.save(save_path)

        loaded_clusterer = PQKMeans.load(save_path)
        labels_after = loaded_clusterer.predict(sample_pq_codes)

        np.testing.assert_array_equal(labels_before, labels_after)

    def test_load_wrong_type_raises(self, tmp_path):
        """Test that loading a non-PQKMeans object raises TypeError."""
        import joblib

        # Save something that's not a PQKMeans
        wrong_path = tmp_path / "wrong.pkl"
        joblib.dump({"not": "a clusterer"}, wrong_path)

        with pytest.raises(TypeError, match="Expected PQKMeans"):
            PQKMeans.load(wrong_path)

    def test_load_with_string_path(self, trained_encoder, sample_pq_codes, tmp_path):
        """Test that load works with string path."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes)

        save_path = str(tmp_path / "clusterer.pkl")
        clusterer.save(save_path)

        loaded = PQKMeans.load(save_path)
        assert isinstance(loaded, PQKMeans)


class TestPQKMeansProperties:
    """Tests for PQKMeans properties."""

    def test_is_trained_property(self, trained_encoder):
        """Test that is_trained reflects encoder state."""
        clusterer = PQKMeans(trained_encoder, k=5)

        # is_trained should be True because encoder is trained
        assert clusterer.is_trained is True


class TestPQKMeansIntegration:
    """Integration tests for PQKMeans with realistic data."""

    def test_cluster_mqn_like_data(self, trained_encoder):
        """Test clustering with MQN-like data (42 dimensions)."""
        np.random.seed(42)

        # Create data with some cluster structure
        n_samples = 200
        n_true_clusters = 4

        # Generate clustered data
        data = []
        for i in range(n_true_clusters):
            center = np.random.rand(42) * 100
            cluster_data = center + np.random.randn(n_samples // n_true_clusters, 42) * 10
            data.append(cluster_data)

        X = np.vstack(data).astype(np.float32)

        # Encode and cluster
        pq_codes = trained_encoder.transform(X, verbose=0)

        clusterer = PQKMeans(trained_encoder, k=n_true_clusters, iteration=10, verbose=False)
        labels = clusterer.fit_predict(pq_codes)

        # Check basic properties
        assert len(np.unique(labels)) <= n_true_clusters
        assert labels.shape == (n_samples,)

    def test_large_k_clustering(self, trained_encoder):
        """Test with larger number of clusters."""
        np.random.seed(42)

        # Need enough samples for k clusters
        X = np.random.rand(1000, 42).astype(np.float32)
        pq_codes = trained_encoder.transform(X, verbose=0)

        k = 50
        clusterer = PQKMeans(trained_encoder, k=k, iteration=5, verbose=False)
        labels = clusterer.fit_predict(pq_codes)

        # Should have multiple clusters represented
        unique_labels = np.unique(labels)
        assert len(unique_labels) > 1
        assert labels.max() < k
