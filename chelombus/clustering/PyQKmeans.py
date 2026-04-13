"""PQKMeans clustering using the C++ implementation from Matsui et al.

Reference:
    @inproceedings{pqkmeans,
        author = {Yusuke Matsui and Keisuke Ogaki and Toshihiko Yamasaki and Kiyoharu Aizawa},
        title = {PQk-means: Billion-scale Clustering for Product-quantized Codes},
        booktitle = {ACM International Conference on Multimedia (ACMMM)},
        year = {2017},
    }
"""
from typing import Literal, overload

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


def _update_centers(
    pq_codes,
    labels,
    K,
    dtables,
    previous_centers=None,
    chunk_size=50_000_000,
):
    """Recompute PQ-code cluster centers from point assignments.

    For each cluster *j* and subspace *s*, pick the codebook entry that
    minimises total symmetric distance to every point in the cluster:

        center[j, s] = argmin_{c'} Σ_c hist[j, c] · dtable[s, c, c']

    where hist[j, c] counts points in cluster j whose subvector-s code is c.

    Histograms are accumulated in chunks to keep peak memory bounded.
    Empty clusters keep their previous center when *previous_centers* is given.
    """
    N, m = pq_codes.shape
    k_cb = dtables.shape[1]
    new_centers = np.zeros((K, m), dtype=np.uint8)
    old_centers = None
    if previous_centers is not None:
        old_centers = np.asarray(previous_centers, dtype=np.uint8)
        if old_centers.shape != (K, m):
            raise ValueError(
                f"previous_centers must have shape {(K, m)}, got {old_centers.shape}"
            )

    for s in range(m):
        hist = np.zeros(K * k_cb, dtype=np.int64)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            flat = (labels[start:end] * k_cb
                    + pq_codes[start:end, s].astype(np.int64))
            hist += np.bincount(flat, minlength=K * k_cb)

        hist_2d = hist.reshape(K, k_cb)
        cost = hist_2d.astype(np.float32) @ dtables[s]
        best_codes = np.argmin(cost, axis=1).astype(np.uint8)

        if old_centers is not None:
            empty = hist_2d.sum(axis=1) == 0
            best_codes[empty] = old_centers[empty, s]

        new_centers[:, s] = best_codes

    return new_centers


class PQKMeans:
    """
    This class provides a scikit-learn-like interface to the PQk-means algorithm,
    which operates directly on PQ codes using symmetric distance.

    Args:
        encoder: A trained PQEncoder instance
        k: Number of clusters
        iteration: Maximum number of k-means iterations (default: 20)
        tol: Early-stopping tolerance for the GPU path. Training stops when
             the fraction of changed center coordinates drops below *tol*.
             0 means stop only on exact convergence. (default: 1e-3)
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
        tol: float = 1e-3,
        verbose: bool = False
    ):
        if not encoder.encoder_is_trained: # type: ignore
            raise ValueError("Encoder must be trained before clustering")

        self.encoder = encoder
        self.k = k
        self.iteration = iteration
        self.tol = tol
        self.verbose = verbose
        self.trained = False
        self._dtables = None
        self._centers_u8 = None
        self._fit_labels = None
        self._cluster = pqkmeans.clustering.PQKMeans(
            encoder=encoder,
            k=k,
            iteration=iteration,
            verbose=verbose,
        )

    def _gpu_support_reason(self) -> str | None:
        if not _GPU_AVAILABLE:
            return "CUDA/Triton not available"
        if self.encoder.k > 256:
            return (
                "GPU path currently supports only 8-bit PQ codes "
                f"(encoder.k <= 256), got encoder.k={self.encoder.k}"
            )
        return None

    def _should_use_gpu(self, device: str) -> bool:
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError(f"device must be 'auto', 'cpu', or 'gpu', got {device!r}")

        gpu_reason = self._gpu_support_reason()
        if device == "gpu":
            if gpu_reason is not None:
                raise RuntimeError(f"GPU requested but unavailable: {gpu_reason}")
            return True

        return device == "auto" and gpu_reason is None

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Get cluster centers (PQ codes of shape (k, m))."""
        return np.array(self._cluster.cluster_centers_)

    @cluster_centers_.setter
    def cluster_centers_(self, centers: np.ndarray) -> None:
        """Set cluster centers and mark as trained."""
        centers_codes = np.asarray(centers, dtype=self.encoder.codebook_dtype)
        self._cluster._impl.set_cluster_centers(centers_codes.tolist())
        self.trained = True
        self._centers_u8 = None
        self._fit_labels = None

    def fit(self, X_train: np.ndarray, device: str = 'auto') -> 'PQKMeans':
        """Fit the PQk-means model to training PQ codes.

        Args:
            X_train: PQ codes of shape (n_samples, n_subvectors)
            device: 'cpu' uses the C++ backend,
                    'gpu' uses Triton assignment + CPU centroid update,
                    'auto' picks GPU when available (default).

        Returns:
            self
        """
        use_gpu = self._should_use_gpu(device)

        if use_gpu:
            self._fit_gpu(X_train)
        else:
            self._cluster.fit(X_train)

        self.trained = True
        self._dtables = None
        self._centers_u8 = None
        self._fit_labels = None
        return self

    @overload
    def _fit_gpu(self, X_train: np.ndarray, return_labels: Literal[False] = False) -> None: ...
    @overload
    def _fit_gpu(self, X_train: np.ndarray, return_labels: Literal[True]) -> np.ndarray: ...
    def _fit_gpu(self, X_train: np.ndarray, return_labels: bool = False) -> np.ndarray | None:
        """GPU-accelerated training: Triton assignment + CPU centroid update."""
        import time

        pq_codes = np.ascontiguousarray(X_train, dtype=np.uint8)
        N, m = pq_codes.shape
        dtables = _build_distance_tables(self.encoder.codewords)
        fit_batch = max(N // 20, 1_000_000)

        # Initialise centres by sampling K data points at random
        rng = np.random.default_rng()
        indices = rng.choice(N, size=self.k, replace=(N < self.k))
        centers = pq_codes[indices].copy()

        labels = None
        prev_centers = None
        final_labels_match_centers = False
        for it in range(self.iteration):
            t0 = time.time()

            # assignment (GPU)
            # Force ~20 batches for progress reporting on large N
            labels = predict_gpu(pq_codes, centers, dtables,
                                 batch_size=fit_batch, verbose=self.verbose)
            t1 = time.time()

            # centroid update (CPU)
            new_centers = _update_centers(
                pq_codes,
                labels,
                self.k,
                dtables,
                previous_centers=centers,
            )
            t2 = time.time()

            changed = int(np.sum(new_centers != centers))
            total_coords = self.k * m
            frac = changed / total_coords

            if self.verbose:
                print(
                    f"  iter {it + 1}/{self.iteration}  "
                    f"assign={t1 - t0:.1f}s  update={t2 - t1:.1f}s  "
                    f"changed={changed}/{total_coords} ({frac:.4%})"
                )

            # Early stopping: below tolerance or an exact 2-cycle oscillation.
            converged = frac <= self.tol
            oscillating = prev_centers is not None and np.array_equal(new_centers, prev_centers)
            final_labels_match_centers = changed == 0
            old_centers = centers
            centers = new_centers

            if converged or oscillating:
                if self.verbose:
                    reason = "converged" if converged else "oscillation"
                    print(f"  Stopped at iteration {it + 1} ({reason})")
                break
            prev_centers = old_centers.copy()

        # Store centres in the pqkmeans backend for compatibility
        self._cluster._impl.set_cluster_centers(centers.tolist())
        self._fit_labels = None

        if not return_labels:
            return None

        if labels is None:
            raise RuntimeError("GPU fit did not produce assignments")

        if final_labels_match_centers:
            return labels

        return predict_gpu(
            pq_codes,
            centers,
            dtables,
            batch_size=fit_batch,
            verbose=self.verbose,
        )

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
            self._centers_u8 = self.cluster_centers_.astype(self.encoder.codebook_dtype)

        use_gpu = self._should_use_gpu(device)

        if use_gpu:
            codes = np.asarray(X, dtype=np.uint8)
            centers = np.asarray(self._centers_u8, dtype=np.uint8)
            return predict_gpu(codes, centers, self._dtables)

        codes = np.asarray(X, dtype=self.encoder.codebook_dtype)
        return _predict_numba(codes, self._centers_u8, self._dtables)

    def fit_predict(self, X: np.ndarray, device: str = 'auto') -> np.ndarray:
        """Fit the model and predict cluster labels in one step.

        Args:
            X: PQ codes of shape (n_samples, n_subvectors)
            device: 'cpu', 'gpu', or 'auto' (default).

        Returns:
            Cluster labels of shape (n_samples,)
        """
        use_gpu = self._should_use_gpu(device)

        if use_gpu:
            labels = self._fit_gpu(X, return_labels=True)
            self.trained = True
            self._dtables = None
            self._centers_u8 = None
            self._fit_labels = None
            return labels

        labels = np.array(self._cluster.fit_predict(X))
        self.trained = True
        self._dtables = None
        self._centers_u8 = None
        self._fit_labels = None
        return labels

    @property
    def is_trained(self) -> bool:
        return bool(self.trained) # type: ignore

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dtables"] = None
        state["_centers_u8"] = None
        state["_fit_labels"] = None
        return state

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
        if not hasattr(obj, '_centers_u8'):
            obj._centers_u8 = None
        if not hasattr(obj, '_fit_labels'):
            obj._fit_labels = None
        return obj
