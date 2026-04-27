import numpy as np
from .encoder_base import PQEncoderBase
from tqdm import tqdm
from sklearn.cluster import KMeans
from pathlib import Path
import joblib
from numpy.typing import NDArray

_GPU_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        _GPU_AVAILABLE = True
except ImportError:
    pass


class PQEncoder(PQEncoderBase):
    """
    Class to encode high-dimensional vectors into PQ-codes.
    """
    def __init__(self, k:int=256, m:int=8, iterations=20):
        """ Initializes the encoder with trained sub-block centroids.

        Args:
            k (int): Number of centroids. Default is 256. We assume that all subquantizers 
            have the same finit number (k') of reproduction values. So k = (k')^m 
            m (int): Number of distinct subvectors of the input X vector. 
            m is a subvector of dimension D' = D/m where D is the 
            dimension of the input vector X. D is therefore a multiple of m
            iterations (int): Number of iterations for the k-means

        High values of k increase the computational cost of the quantizer as well as memory 
        usage of strogin the centroids (k' x D floating points). Using k=256 and m=8 is often
        a reasonable choice. 

        Reference: DOI: 10.1109/TPAMI.2010.57
        """

        self.m = m 
        self.k = k 
        self.iterations = iterations
        self.encoder_is_trained = False
        self.codebook_dtype =  np.uint8 if self.k <= 2**8 else (np.uint16 if self.k<= 2**16 else np.uint32)
        self.pq_trained = []

        """ The codebook is defined as the Cartesian product of all centroids. 
        Storing the codebook C explicitly is not efficient, the full codebook would be (k')^m  centroids
        Instead we store the K' centroids of all the subquantizers (k' · m). We can later simulate the full 
        codebook by combining the centroids from each subquantizer. 
        """
    @property
    def is_trained(self) -> bool: return self.encoder_is_trained

    def fit(self, X_train:NDArray, verbose:int=1, device:str='auto', **kwargs)->None:
        """ KMeans fitting of every subvector matrix from the X_train matrix. Populates
        the codebook by storing the cluster centers of every subvector

        X_train is the input matrix. For a vector that has dimension D
        then X_train is a matrix of size (N, D)
        where N is the number of rows (vectors) and D the number of columns (dimension of
        every vector i.e. fingerprint in the case of molecular data)

        Args:
           X_train(np.array): Input matrix to train the encoder.
           verbose(int): Level of verbosity. Default is 1
           device: 'cpu' for sklearn KMeans, 'gpu' for torch-based KMeans on CUDA,
                   'auto' picks GPU when available. Default is 'auto'.
           **kwargs: Optional keyword arguments passed to the underlying KMeans `fit()` function
                     (only used on the CPU path).
        """

        assert X_train.ndim == 2, "The input can only be a matrix (X.ndim == 2)"
        N, D = X_train.shape # N number of input vectors, D dimension of the vectors
        assert self.k < N, "the number of training vectors (N for N,D = X_train.shape) should be more than the number of centroids (K)"
        assert D % self.m == 0, f"Vector (fingerprint) dimension should be divisible by the number of subvectors (m). Got {D} / {self.m}"
        self.D_subvector = int(D / self.m) # Dimension of the subvector.
        self.og_D = D # We save the original dimensions of the input vector (fingerprint) for later use
        assert not self.encoder_is_trained, "Encoder can only be fitted once"

        self.codewords= np.zeros((self.m, self.k, self.D_subvector), dtype=np.float32)

        use_gpu = (device == 'gpu') or (device == 'auto' and _GPU_AVAILABLE)
        if use_gpu:
            if not _GPU_AVAILABLE:
                raise RuntimeError("GPU requested but CUDA not available")
            self._fit_gpu(X_train, verbose)
        else:
            self._fit_cpu(X_train, verbose, **kwargs)

        self.encoder_is_trained = True
        del X_train # remove initial training data from memory

    def _fit_cpu(self, X_train: NDArray, verbose: int = 1, **kwargs) -> None:
        subvector_dim = self.D_subvector

        iterable = range(self.m)
        if verbose > 0:
            iterable = tqdm(iterable, desc='Training PQ-encoder', total=self.m)

        for subvector_idx in iterable:
            X_train_subvector = X_train[:, subvector_dim * subvector_idx : subvector_dim * (subvector_idx + 1)]
            kmeans = KMeans(n_clusters=self.k, init='k-means++', max_iter=self.iterations, **kwargs).fit(X_train_subvector)
            self.pq_trained.append(kmeans)
            self.codewords[subvector_idx] = kmeans.cluster_centers_

    @staticmethod
    def _gpu_encoder_batch_size(
        N: int,
        k: int,
        reserve_mb: int = 1024,
        min_batch: int = 250_000,
        max_batch: int = 2_000_000,
    ) -> int:
        """Choose a GPU assignment batch size from the current free VRAM.

        This is called after the resident tensors for one subvector are already
        allocated on the GPU. The main scratch tensors per batch point are:

        - one float32 distance row of length ``k``
        - one int64 label from ``argmin``
        - one float32 ``ones`` entry for ``index_add_``

        A reserve margin is kept for CUDA context, allocator fragmentation,
        and GEMM workspace. If VRAM telemetry is unavailable, the method falls
        back to a conservative fixed 1 GiB scratch budget.
        """
        if N <= 0:
            return 0

        floor = min(min_batch, N)
        ceiling = min(max_batch, N)
        bytes_per_point = k * 4 + 8 + 4
        fallback_budget = 1 * 1024**3

        try:
            free, _ = torch.cuda.mem_get_info()
            usable = free - reserve_mb * 1024**2
            scratch_budget = usable if usable > 0 else max(free // 2, floor * bytes_per_point)
        except Exception:
            scratch_budget = fallback_budget

        batch = max(scratch_budget // bytes_per_point, floor)
        return int(min(max(batch, floor), ceiling))

    def _fit_gpu(self, X_train: NDArray, verbose: int = 1) -> None:
        """GPU-accelerated KMeans fitting with batched assignment.

        Each subvector's data (N x D_sub) is kept GPU-resident.  The
        distance matrix is computed in batches to cap VRAM at ~1 GB
        scratch regardless of N.  The GEMM form
        ``||x-c||² = ||x||² + ||c||² - 2·x·cᵀ`` avoids ``torch.cdist``
        workspace overhead and fuses well with cuBLAS.
        """
        N = X_train.shape[0]
        subvector_dim = self.D_subvector
        X_f32 = np.ascontiguousarray(X_train, dtype=np.float32)
        rng = np.random.default_rng()

        iterable = range(self.m)
        if verbose > 0:
            iterable = tqdm(iterable, desc='Training PQ-encoder (GPU)', total=self.m)

        for subvector_idx in iterable:
            sub_slice = X_f32[:, subvector_dim * subvector_idx : subvector_dim * (subvector_idx + 1)]
            X_gpu = torch.from_numpy(sub_slice).cuda()
            # Precompute ||x||^2 (stays constant across iterations)
            x_sq = (X_gpu * X_gpu).sum(dim=1)  # (N,)
            B = self._gpu_encoder_batch_size(N, self.k)

            # Random init
            idx = rng.choice(N, size=self.k, replace=(N < self.k))
            centroids = X_gpu[torch.from_numpy(idx.astype(np.int64)).cuda()].clone()

            for _ in range(self.iterations):
                c_sq = (centroids * centroids).sum(dim=1)  # (k,)
                counts = torch.zeros(self.k, dtype=torch.float32, device='cuda')
                sums = torch.zeros_like(centroids)

                for start in range(0, N, B):
                    end = min(start + B, N)
                    x_batch = X_gpu[start:end]           # view, no copy
                    # ||x-c||² = ||x||² + ||c||² - 2·x·cᵀ
                    dist = (x_sq[start:end, None]
                            + c_sq[None, :]
                            - 2.0 * (x_batch @ centroids.T))
                    dist.clamp_min_(0.0)
                    labels = dist.argmin(dim=1)
                    del dist

                    ones = torch.ones(end - start, dtype=torch.float32, device='cuda')
                    counts.index_add_(0, labels, ones)
                    sums.index_add_(0, labels, x_batch)
                    del labels, ones

                empty = counts == 0
                counts.clamp_min_(1.0)
                new_centroids = sums / counts.unsqueeze(1)
                new_centroids[empty] = centroids[empty]
                centroids = new_centroids

            self.codewords[subvector_idx] = centroids.cpu().numpy()
            del X_gpu, centroids, x_sq


    def transform(self, X:NDArray, verbose:int=1, device:str='auto', **kwargs) -> NDArray:
        """
        Transforms the input matrix X into its PQ-codes.

        For each sample in X, the input vector is split into `m` equal-sized subvectors.
        Each subvector is assigned to the nearest cluster centroid
        and the index of the closest centroid is stored.

        The result is a compact representation of X, where each sample is encoded as a sequence of centroid indices.

        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features),
                            where n_features must be divisible by the number of subvectors `m`.
            verbose(int): Level of verbosity. Default is 1
            device: 'cpu' for sklearn, 'gpu' for torch.cdist on CUDA, 'auto' to pick GPU if available.
            **kwargs: Optional keyword arguments passed to the underlying KMeans `predict()` function.

        Returns:
            np.ndarray: PQ codes of shape (n_samples, m), where each element is the index of the nearest centroid
                        for the corresponding subvector.
        """

        assert self.encoder_is_trained, "PQEncoder must be trained before calling transform"

        use_gpu = (device == 'gpu') or (device == 'auto' and _GPU_AVAILABLE)
        if use_gpu:
            if not _GPU_AVAILABLE:
                raise RuntimeError("GPU requested but CUDA not available")
            return self._transform_gpu(X)

        return self._transform_cpu(X, verbose, **kwargs)

    def _transform_cpu(self, X: NDArray, verbose: int = 1, **kwargs) -> NDArray:
        N, D = X.shape
        pq_codes = np.zeros((N, self.m), dtype=self.codebook_dtype)
        has_sklearn_models = len(self.pq_trained) == self.m

        iterable = range(self.m)
        if verbose > 0:
            iterable = tqdm(iterable, desc='Generating PQ-codes', total=self.m)

        subvector_dim = int(D / self.m)

        for subvector_idx in iterable:
            X_train_subvector = X[:, subvector_dim * subvector_idx : subvector_dim * (subvector_idx + 1)]
            if has_sklearn_models:
                pq_codes[:, subvector_idx] = self.pq_trained[subvector_idx].predict(X_train_subvector, **kwargs)
                continue

            codewords = np.ascontiguousarray(self.codewords[subvector_idx], dtype=np.float32)
            X_sub = np.ascontiguousarray(X_train_subvector, dtype=np.float32)
            bytes_budget = 64 * 1024**2
            bytes_per_row = max(codewords.shape[0] * codewords.shape[1] * 4, 1)
            chunk_size = max(1, min(N, bytes_budget // bytes_per_row))

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                chunk = X_sub[start:end]
                diff = chunk[:, None, :] - codewords[None, :, :]
                dists = np.sum(diff * diff, axis=2)
                pq_codes[start:end, subvector_idx] = np.argmin(dists, axis=1).astype(self.codebook_dtype)

        del X
        return pq_codes

    def _transform_gpu(self, X: NDArray) -> NDArray:
        """GPU-accelerated transform using torch.cdist + argmin.

        Batches points to avoid OOM.  Each batch is uploaded to the GPU once
        and then sliced per subvector on-device, avoiding m separate PCIe
        transfers and CPU-side contiguous copies.
        """
        N, D = X.shape
        subvector_dim = int(D / self.m)
        pq_codes = np.zeros((N, self.m), dtype=self.codebook_dtype)

        cw_gpu = [
            torch.from_numpy(np.ascontiguousarray(self.codewords[sub], dtype=np.float32)).cuda()
            for sub in range(self.m)
        ]

        # Budget: batch tensor (D floats) + distance matrix (k floats) per point
        bytes_per_point = (D + self.k) * 4
        max_batch = max((2 * 1024**3) // bytes_per_point, 1024)
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)

        for start in range(0, N, max_batch):
            end = min(start + max_batch, N)
            batch_gpu = torch.from_numpy(X_f32[start:end]).cuda()
            for sub in range(self.m):
                chunk = batch_gpu[:, subvector_dim * sub : subvector_dim * (sub + 1)]
                dists = torch.cdist(chunk, cw_gpu[sub])
                pq_codes[start:end, sub] = dists.argmin(dim=1).cpu().numpy()
                del dists
            del batch_gpu

        del cw_gpu
        return pq_codes

    def fit_transform(self, X:NDArray, verbose:int=1, device:str='auto', **kwargs) -> NDArray:
        """Fit and transforms the input matrix `X` into its PQ-codes

        The encoder is trained on the matrix and then for each sample in X,
          the input vector is split into `m` equal-sized vectors subvectors composed
          by the index of the closest centroid. Returns a compact representation of X,
          where each sample is encoded as a sequence of centroid indices (i.e PQcodes)

        Args:
            X (np.array): Input data matrix of shape (n_samples, n_features)
            verbose (int, optional): Level of verbosity. Defaults to 1.
            device: 'cpu', 'gpu', or 'auto' (picks GPU when available). Default is 'auto'.
            **kwargs: Optional keyword. These arguments will be passed to the underlying KMeans
            predict() function.

        Returns:
            np.array: PQ codes of shape (n_samples, m), where each element is the index of the nearest
            centroid for the corresponding subvector
        """

        self.fit(X, verbose, device=device, **kwargs)
        return self.transform(X, verbose, device=device)


    def inverse_transform(self, 
                          pq_codes:NDArray, 
                          binary=False, 
                          round=True):
        """ Inverse transform. From PQ-code to the original vector. 
        This process is lossy so we don't expect to get the exact same data.
        If binary=True then the vectors will be returned in binary. 
        This is useful for the case where our original vectors were binary. 
        With binary=True then the returned vectors are transformed from 
        [0.32134, 0.8232, 0.0132, ... 0.1432, 1.19234] to 
        [0, 1, 0, ..., 0, 1]
        If round=True then the vector will be approximated to integer values
        (useful in cases where we expeect to have a count-based fingerprint)

        Args:
            pq_code: (np.array): Input data of PQ codes to be transformed into the
            original vectors.
            binary: (bool): Wheter to return the vectors rounded to 0s and 1s. Default is False
            round: (bool): Round inversed vector to integers values. Default is True  
        """
        
        # Get shape of the input matrix of PQ codes
        N, D = pq_codes.shape

        # The dimension of the PQ vectors should be the same 
        # as the number of splits (subvectors) from the original data 

        assert D == (self.m), f"The dimension D of the PQ-codes (N,D) should be the same as the number of the subvectors or splits (m) . Got D = {D} for m = {self.m}"
        assert D == (self.og_D  / self.D_subvector), f"The dimension D of the PQ-codes (N,D) should be the same as the original vector dimension divided the subvector dimension"

        X_inversed = np.empty((N, D*self.D_subvector), dtype=float)
        for subvector_idx in range(self.m):
            X_inversed[:, subvector_idx*self.D_subvector:((subvector_idx+1)*self.D_subvector)] = self.codewords[subvector_idx][pq_codes[:, subvector_idx], :]

        if binary:
            reconstructed_binary = (X_inversed>= 0.6).astype('int8')
            return reconstructed_binary 

        if round: return X_inversed.astype(int)
        else: return X_inversed
        

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "version": 1,
            "init": {"k": self.k, "m": self.m, "iterations": self.iterations},
            "state": {
                "encoder_is_trained": self.encoder_is_trained,
                "og_D": getattr(self, "og_D", None),
                "D_subvector": getattr(self, "D_subvector", None),
                "codewords": getattr(self, "codewords", None),
                "pq_trained": self.pq_trained,  # optional
            },
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "PQEncoder":
        path = Path(path)
        payload = joblib.load(path)

        if isinstance(payload, cls): return payload

        obj = cls(**payload["init"])
        state = payload["state"]
        obj.encoder_is_trained = state["encoder_is_trained"]
        obj.og_D = state["og_D"]
        obj.D_subvector = state["D_subvector"]
        obj.codewords = state["codewords"]
        obj.pq_trained = state.get("pq_trained", [])
        return obj
