"""PQKMeans clustering using the C++ implementation from Matsui et al.

Reference:
    @inproceedings{pqkmeans,
        author = {Yusuke Matsui and Keisuke Ogaki and Toshihiko Yamasaki and Kiyoharu Aizawa},
        title = {PQk-means: Billion-scale Clustering for Product-quantized Codes},
        booktitle = {ACM International Conference on Multimedia (ACMMM)},
        year = {2017},
    }
"""
import joblib
from pathlib import Path
import numpy as np
import pqkmeans
from numba import njit, prange
from chelombus import PQEncoder

_GPU_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        from chelombus.clustering._gpu_predict import predict_gpu
        _GPU_AVAILABLE = True
except ImportError:
    pass


def _build_distance_tables(codewords: np.ndarray) -> np.ndarray:
    """Precompute symmetric squared-distance tables per subvector.

    Args:
        codewords: (m, k_codebook, D_sub) codebook arrays from the encoder.

    Returns:
        (m, k_codebook, k_codebook) float32 distance lookup tables.
    """
    m, k_codebook, _ = codewords.shape
    dtables = np.zeros((m, k_codebook, k_codebook), dtype=np.float32)
    for sub in range(m):
        cb = codewords[sub]
        diff = cb[:, None, :] - cb[None, :, :]
        dtables[sub] = np.sum(diff ** 2, axis=2)
    return dtables


@njit(parallel=True, cache=True)
def _predict_numba(pq_codes, centers, dtables):
    """Assign each PQ code to its nearest cluster center using symmetric distance."""
    n = pq_codes.shape[0]
    m = pq_codes.shape[1]
    n_centers = centers.shape[0]
    labels = np.empty(n, dtype=np.int64)
    for i in prange(n):
        best_dist = np.inf
        best_label = 0
        for c in range(n_centers):
            dist = np.float32(0.0)
            for sub in range(m):
                dist += dtables[sub, pq_codes[i, sub], centers[c, sub]]
            if dist < best_dist:
                best_dist = dist
                best_label = c
        labels[i] = best_label
    return labels


class PQKMeans:
    """
    This class provides a scikit-learn-like interface to the PQk-means algorithm,
    which operates directly on PQ codes using symmetric distance.

    Args:
        encoder: A trained PQEncoder instance
        k: Number of clusters
        iteration: Number of k-means iterations (default: 20)
        verbose: Whether to print progress information (default: False)

    Example:
        >>> encoder = PQEncoder(k=256, m=6)
        >>> encoder.fit(training_data)
        >>> pq_codes = encoder.transform(data)
        >>> clusterer = PQKMeans(encoder, k=100000)
        >>> labels = clusterer.fit_predict(pq_codes)
    """

    def __init__(
        self,
        encoder: PQEncoder,
        k: int,
        iteration: int = 20,
        verbose: bool = False
    ):
        if not encoder.encoder_is_trained: # type: ignore
            raise ValueError("Encoder must be trained before clustering")

        self.encoder = encoder
        self.k = k
        self.iteration = iteration
        self.verbose = verbose
        self.trained = False
        self._dtables = None
        self._centers_u8 = None
        self._cluster = pqkmeans.clustering.PQKMeans(
            encoder=encoder,
            k=k,
            verbose=verbose
        )

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Get cluster centers (PQ codes of shape (k, m))."""
        return np.array(self._cluster.cluster_centers_)

    @cluster_centers_.setter
    def cluster_centers_(self, centers: np.ndarray) -> None:
        """Set cluster centers and mark as trained."""
        centers_uint8 = np.array(centers).astype(np.uint8)
        self._cluster._impl.set_cluster_centers(centers_uint8.tolist())
        self.trained = True

    def fit(self, X_train: np.ndarray) -> 'PQKMeans':
        """Fit the PQk-means model to training PQ codes.

        Args:
            X_train: PQ codes of shape (n_samples, n_subvectors)

        Returns:
            self
        """
        self._cluster.fit(X_train)
        self.trained = True
        self._dtables = None
        self._centers_u8 = None
        return self

    def predict(self, X: np.ndarray, device: str = 'auto') -> np.ndarray:
        """Predict cluster labels for PQ codes.

        Args:
            X: PQ codes of shape (n_samples, n_subvectors), dtype uint8
            device: 'cpu' for Numba, 'gpu' for Triton/CUDA, 'auto' to pick GPU if available.

        Returns:
            Cluster labels of shape (n_samples,)
        """
        if not self.trained: # type: ignore
            raise ValueError("Must be trained before clustering. Use `.fit()` first")
        if self._dtables is None:
            self._dtables = _build_distance_tables(self.encoder.codewords)
            self._centers_u8 = self.cluster_centers_.astype(np.uint8)

        use_gpu = (device == 'gpu') or (device == 'auto' and _GPU_AVAILABLE)
        codes = np.asarray(X, dtype=np.uint8)

        if use_gpu:
            if not _GPU_AVAILABLE:
                raise RuntimeError("GPU requested but CUDA/Triton not available")
            return predict_gpu(codes, self._centers_u8, self._dtables)

        return _predict_numba(codes, self._centers_u8, self._dtables)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and predict cluster labels in one step.

        Args:
            X: PQ codes of shape (n_samples, n_subvectors)

        Returns:
            Cluster labels of shape (n_samples,)
        """
        self.trained = True
        return np.array(self._cluster.fit_predict(X))

    @property
    def is_trained(self) -> bool:
        return bool(self.trained) # type: ignore

    def save(self, path: str | Path) -> None:
        path = Path(path)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PQKMeans":
        path = Path(path)
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        if not hasattr(obj, '_dtables'):
            obj._dtables = None
        return obj
