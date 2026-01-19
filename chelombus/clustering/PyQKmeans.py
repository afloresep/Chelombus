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
from typing import Type
import numpy as np
import pqkmeans
from chelombus import PQEncoder


class PQKMeans:
    """Wrapper around pqkmeans C++ library for billion-scale clustering.

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
        self.cluster = pqkmeans.clustering.PQKMeans(
            encoder=encoder,
            k=k,
            iteration=iteration,
            verbose=verbose
        )

    def fit(self, X_train: np.ndarray) -> 'PQKMeans':
        """Fit the PQk-means model to training PQ codes.

        Args:
            X_train: PQ codes of shape (n_samples, n_subvectors)

        Returns:
            self
        """
        self.cluster.fit(X_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for PQ codes.

        Args:
            X: PQ codes of shape (n_samples, n_subvectors)

        Returns:
            Cluster labels of shape (n_samples,)
        """
        return self.cluster.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and predict cluster labels in one step.

        Args:
            X: PQ codes of shape (n_samples, n_subvectors)

        Returns:
            Cluster labels of shape (n_samples,)
        """
        return self.cluster.fit_predict(X)

    @property
    def is_trained(self) -> bool:
        return bool(self.encoder.encoder_is_trained) # type: ignore

    def save(self, path: str | Path) -> None:
        path = Path(path)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PQKMeans":
        path = Path(path)
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
