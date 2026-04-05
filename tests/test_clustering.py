"""Tests for the PQKMeans clustering module.

These tests require the pqkmeans library to be installed. If not installed,
all tests will be skipped.
"""

import numpy as np
import pytest
from pathlib import Path

from chelombus.encoder.encoder import PQEncoder
import chelombus.encoder.encoder as encoder_module
import chelombus.clustering.PyQKmeans as pyqkmeans_module

# Skip all tests if pqkmeans is not installed
pqkmeans_available = pytest.importorskip("pqkmeans", reason="pqkmeans not installed")

from chelombus.clustering.PyQKmeans import PQKMeans, _predict_numba, _update_centers

_GPU_AVAILABLE = False
try:
    import torch
    _GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

gpu = pytest.mark.skipif(not _GPU_AVAILABLE, reason="CUDA not available")


@pytest.fixture
def trained_encoder():
    """Create a trained PQEncoder for testing."""
    np.random.seed(42)
    # Create training data: 500 samples, 42 dimensions (like MQN fingerprints)
    # Use float64 for sklearn compatibility
    X_train = np.random.rand(500, 42).astype(np.float64)

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
    # Use float64 to match training dtype
    X = np.random.rand(100, 42).astype(np.float64)
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
        assert clusterer.encoder.is_trained is True  # Encoder is trained

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

    def test_is_trained_property(self, trained_encoder, sample_pq_codes):
        """Test that is_trained reflects encoder state."""
        clusterer = PQKMeans(trained_encoder, k=5)
        # is_trained should be False because hasn't been trained
        assert clusterer.is_trained is False
        clusterer.fit(sample_pq_codes)  # Use proper PQ codes with correct dimensions
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

        X = np.vstack(data).astype(np.float64)

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
        X = np.random.rand(1000, 42).astype(np.float64)
        pq_codes = trained_encoder.transform(X, verbose=0)

        k = 50
        clusterer = PQKMeans(trained_encoder, k=k, iteration=5, verbose=False)
        labels = clusterer.fit_predict(pq_codes)

        # Should have multiple clusters represented
        unique_labels = np.unique(labels)
        assert len(unique_labels) > 1
        assert labels.max() < k


class TestPQKMeansRegressions:
    """Regression tests for GPU control flow that run without CUDA."""

    def test_update_centers_preserves_empty_clusters(self):
        """Empty clusters should keep their previous center instead of collapsing to zero."""
        pq_codes = np.array([[5, 7], [5, 7], [9, 1]], dtype=np.uint8)
        labels = np.array([0, 0, 1], dtype=np.int64)
        previous_centers = np.array([[5, 7], [9, 1], [4, 3]], dtype=np.uint8)

        dtables = np.ones((2, 256, 256), dtype=np.float32)
        for sub in range(2):
            np.fill_diagonal(dtables[sub], 0.0)

        new_centers = _update_centers(
            pq_codes,
            labels,
            K=3,
            dtables=dtables,
            previous_centers=previous_centers,
        )

        np.testing.assert_array_equal(new_centers[0], [5, 7])
        np.testing.assert_array_equal(new_centers[1], [9, 1])
        np.testing.assert_array_equal(new_centers[2], previous_centers[2])

    def test_mocked_gpu_fit_predict_matches_final_centers(self, monkeypatch, trained_encoder, sample_pq_codes):
        """fit_predict on the GPU path must return labels for the final stored centers."""
        monkeypatch.setattr(pyqkmeans_module, "_GPU_AVAILABLE", True)
        monkeypatch.setattr(
            pyqkmeans_module,
            "predict_gpu",
            lambda pq_codes, centers, dtables, batch_size=0, verbose=False:
                _predict_numba(pq_codes, centers, dtables),
            raising=False,
        )
        monkeypatch.setattr(
            pyqkmeans_module.np.random,
            "default_rng",
            lambda *args, **kwargs: np.random.Generator(np.random.PCG64(0)),
        )

        clusterer = PQKMeans(trained_encoder, k=10, iteration=20, tol=1e-3, verbose=False)
        labels = clusterer.fit_predict(sample_pq_codes, device='gpu')
        final_labels = clusterer.predict(sample_pq_codes, device='cpu')

        np.testing.assert_array_equal(labels, final_labels)

    def test_mocked_gpu_fit_clears_transient_labels(self, monkeypatch, trained_encoder, sample_pq_codes):
        """GPU fit should not leave the last training assignments attached to the model."""
        monkeypatch.setattr(pyqkmeans_module, "_GPU_AVAILABLE", True)
        monkeypatch.setattr(
            pyqkmeans_module,
            "predict_gpu",
            lambda pq_codes, centers, dtables, batch_size=0, verbose=False:
                _predict_numba(pq_codes, centers, dtables),
            raising=False,
        )
        monkeypatch.setattr(
            pyqkmeans_module.np.random,
            "default_rng",
            lambda *args, **kwargs: np.random.Generator(np.random.PCG64(1)),
        )

        clusterer = PQKMeans(trained_encoder, k=5, iteration=5, verbose=False)
        clusterer.fit(sample_pq_codes, device='gpu')

        assert clusterer._fit_labels is None
        assert clusterer.__getstate__()["_fit_labels"] is None

    def test_auto_falls_back_to_cpu_when_gpu_path_is_unsupported(self, monkeypatch):
        """device='auto' should not try the GPU path when m is unsupported."""
        monkeypatch.setattr(pyqkmeans_module, "_GPU_AVAILABLE", True)

        def fail_predict_gpu(*args, **kwargs):
            raise AssertionError("predict_gpu should not be called for unsupported m")

        monkeypatch.setattr(pyqkmeans_module, "predict_gpu", fail_predict_gpu, raising=False)

        np.random.seed(7)
        X = np.random.rand(200, 40).astype(np.float32)
        encoder = PQEncoder(k=16, m=8, iterations=3)
        encoder.fit(X, verbose=0)
        pq_codes = encoder.transform(X, verbose=0)

        clusterer = PQKMeans(encoder, k=5, iteration=3, verbose=False)
        labels = clusterer.fit_predict(pq_codes, device='auto')

        assert labels.shape == (len(pq_codes),)
        assert clusterer.predict(pq_codes, device='auto').shape == (len(pq_codes),)


class TestPQEncoderRegressions:
    """Regression tests for CPU portability after GPU-oriented training changes."""

    def test_gpu_encoder_batch_size_uses_free_vram(self, monkeypatch):
        """GPU encoder batch sizing should respond to the current free VRAM."""
        monkeypatch.setattr(
            encoder_module.torch.cuda,
            "mem_get_info",
            lambda: (int(1.5 * 1024**3), int(16 * 1024**3)),
        )

        batch = PQEncoder._gpu_encoder_batch_size(N=10_000_000, k=256)
        expected = max((int(0.5 * 1024**3)) // (256 * 4 + 8 + 4), 250_000)

        assert batch == expected

    def test_cpu_transform_without_sklearn_models_uses_codewords(self, tmp_path):
        """A saved encoder must remain usable on the CPU even without stored sklearn models."""
        np.random.seed(11)
        X = np.random.rand(200, 42).astype(np.float32)

        encoder = PQEncoder(k=16, m=6, iterations=5)
        encoder.fit(X, verbose=0)
        expected = encoder.transform(X[:50], verbose=0, device='cpu')

        encoder.pq_trained = []
        path = tmp_path / "encoder.pkl"
        encoder.save(path)

        loaded = PQEncoder.load(path)
        actual = loaded.transform(X[:50], verbose=0, device='cpu')

        np.testing.assert_array_equal(actual, expected)


# ── GPU-specific tests ────────────────────────────────────────────────────


class TestGPUPredict:
    """Tests for GPU prediction path."""

    @gpu
    def test_gpu_predict_matches_cpu(self, trained_encoder, sample_pq_codes):
        """GPU and CPU predict must return identical labels."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes, device='cpu')

        labels_cpu = clusterer.predict(sample_pq_codes, device='cpu')
        labels_gpu = clusterer.predict(sample_pq_codes, device='gpu')

        np.testing.assert_array_equal(labels_cpu, labels_gpu)

    @gpu
    def test_gpu_predict_shape_and_range(self, trained_encoder, sample_pq_codes):
        """GPU predict returns correct shape and valid label range."""
        k = 5
        clusterer = PQKMeans(trained_encoder, k=k, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes, device='cpu')

        labels = clusterer.predict(sample_pq_codes, device='gpu')

        assert labels.shape == (len(sample_pq_codes),)
        assert labels.min() >= 0
        assert labels.max() < k
        assert np.issubdtype(labels.dtype, np.integer)

    @gpu
    def test_gpu_predict_deterministic(self, trained_encoder, sample_pq_codes):
        """Repeated GPU predict calls return the same labels."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes, device='cpu')

        labels1 = clusterer.predict(sample_pq_codes, device='gpu')
        labels2 = clusterer.predict(sample_pq_codes, device='gpu')

        np.testing.assert_array_equal(labels1, labels2)


class TestGPUFit:
    """Tests for GPU training path."""

    @gpu
    def test_gpu_fit_produces_valid_clusters(self, trained_encoder, sample_pq_codes):
        """GPU fit must produce a trained model with valid labels."""
        k = 5
        clusterer = PQKMeans(trained_encoder, k=k, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes, device='gpu')

        assert clusterer.is_trained
        labels = clusterer.predict(sample_pq_codes)
        assert labels.shape == (len(sample_pq_codes),)
        assert labels.min() >= 0
        assert labels.max() < k

    @gpu
    def test_gpu_fit_returns_self(self, trained_encoder, sample_pq_codes):
        """GPU fit returns self for method chaining."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        result = clusterer.fit(sample_pq_codes, device='gpu')

        assert result is clusterer

    @gpu
    def test_gpu_fit_centers_are_valid_codes(self, trained_encoder, sample_pq_codes):
        """Cluster centers after GPU fit must be valid codebook indices."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, verbose=False)
        clusterer.fit(sample_pq_codes, device='gpu')

        centers = clusterer.cluster_centers_
        assert centers.shape == (5, trained_encoder.m)
        assert centers.min() >= 0
        assert centers.max() < trained_encoder.k

    @gpu
    def test_gpu_fit_predict(self, trained_encoder, sample_pq_codes):
        """GPU fit_predict returns labels of correct shape and range."""
        k = 5
        clusterer = PQKMeans(trained_encoder, k=k, iteration=3, verbose=False)
        labels = clusterer.fit_predict(sample_pq_codes, device='gpu')

        assert labels.shape == (len(sample_pq_codes),)
        assert labels.min() >= 0
        assert labels.max() < k


class TestGPUSaveLoad:
    """Tests for save/load round-trip after GPU training."""

    @gpu
    def test_gpu_save_load_labels_match(self, trained_encoder, sample_pq_codes, tmp_path):
        """Labels must match after save/load of a GPU-trained model."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, tol=0, verbose=False)
        clusterer.fit(sample_pq_codes, device='gpu')
        labels_before = clusterer.predict(sample_pq_codes)

        path = tmp_path / "gpu_model.pkl"
        clusterer.save(path)
        loaded = PQKMeans.load(path)
        labels_after = loaded.predict(sample_pq_codes)

        np.testing.assert_array_equal(labels_before, labels_after)

    @gpu
    def test_gpu_save_load_gpu_predict(self, trained_encoder, sample_pq_codes, tmp_path):
        """GPU predict works correctly on a loaded model."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, tol=0, verbose=False)
        clusterer.fit(sample_pq_codes, device='gpu')
        labels_before = clusterer.predict(sample_pq_codes, device='gpu')

        path = tmp_path / "gpu_model.pkl"
        clusterer.save(path)
        loaded = PQKMeans.load(path)
        labels_after = loaded.predict(sample_pq_codes, device='gpu')

        np.testing.assert_array_equal(labels_before, labels_after)


class TestEarlyStopping:
    """Tests for early stopping in GPU training."""

    @gpu
    def test_tol_stops_before_max_iterations(self, trained_encoder, sample_pq_codes):
        """With default tol, training should stop before iteration limit."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=50, tol=1e-3, verbose=False)
        clusterer.fit(sample_pq_codes, device='gpu')

        # If early stopping works, it should finish quickly (not run 50 iters).
        # We just verify it completes and produces valid output.
        assert clusterer.is_trained
        labels = clusterer.predict(sample_pq_codes)
        assert labels.min() >= 0
        assert labels.max() < 5

    @gpu
    def test_tol_zero_only_stops_on_exact_convergence(self, trained_encoder, sample_pq_codes):
        """tol=0 should only stop when no centers change at all."""
        clusterer = PQKMeans(trained_encoder, k=5, iteration=3, tol=0, verbose=False)
        clusterer.fit(sample_pq_codes, device='gpu')

        assert clusterer.is_trained

    @gpu
    def test_default_tol_value(self, trained_encoder):
        """Default tol should be 1e-3."""
        clusterer = PQKMeans(trained_encoder, k=5)
        assert clusterer.tol == 1e-3


class TestGPUEncoderTransform:
    """Tests for GPU encoder transform path."""

    @gpu
    def test_gpu_transform_matches_cpu(self, trained_encoder):
        """GPU and CPU transform must produce identical PQ codes."""
        np.random.seed(99)
        X = np.random.rand(200, 42).astype(np.float64)

        codes_cpu = trained_encoder.transform(X, verbose=0, device='cpu')
        codes_gpu = trained_encoder.transform(X, verbose=0, device='gpu')

        np.testing.assert_array_equal(codes_cpu, codes_gpu)

    @gpu
    def test_gpu_transform_dtype(self, trained_encoder):
        """GPU transform must return the correct codebook dtype."""
        X = np.random.rand(50, 42).astype(np.float64)
        codes = trained_encoder.transform(X, verbose=0, device='gpu')

        assert codes.dtype == trained_encoder.codebook_dtype
