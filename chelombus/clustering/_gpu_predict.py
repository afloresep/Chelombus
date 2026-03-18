"""GPU-accelerated PQ assignment using Triton kernels.

Provides a drop-in replacement for _predict_numba that runs on CUDA GPUs.
The kernel computes symmetric PQ distance via lookup tables and maintains
an online argmin, never materializing the N x K distance matrix.

VRAM budget per call
--------------------
Resident (cached, allocated once):
    centers:  K × M bytes               (100K × 6 = 600 KB)
    dtables:  M × 256 × 256 × 4 bytes   (6 × 256 × 256 × 4 = 1.5 MB)

Per-batch (freed after each batch):
    codes:    batch_n × M bytes          (1 byte per subvector)
    labels:   batch_n × 4 bytes          (int32)
    → 10 bytes per point

So for a given free VRAM of F bytes, max batch ≈ F / 10.
"""

import numpy as np
import torch
import triton
import triton.language as tl

# Fixed VRAM overhead for PyTorch/Triton context (conservative)
_VRAM_OVERHEAD_MB = 256


@triton.jit
def _pq_assign_kernel(
    codes_ptr,      # (N, M) uint8 — PQ codes for data points
    centers_ptr,    # (K, M) uint8 — PQ codes for cluster centers
    dtables_ptr,    # (M, 256, 256) float32 — precomputed distance tables
    labels_ptr,     # (N,) int32 — output cluster assignments
    N,              # number of data points
    K,              # number of cluster centers
    M: tl.constexpr,          # number of subvectors
    BLOCK_N: tl.constexpr,    # number of points per program
    BLOCK_K: tl.constexpr,    # number of centers per tile
):
    pid = tl.program_id(0)
    point_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    point_mask = point_offs < N

    best_dist = tl.full((BLOCK_N,), float('inf'), dtype=tl.float32)
    best_label = tl.zeros((BLOCK_N,), dtype=tl.int32)

    # Pre-load point codes per subvector (M=6, unrolled)
    pc0 = tl.load(codes_ptr + point_offs * M + 0, mask=point_mask, other=0).to(tl.int32)
    pc1 = tl.load(codes_ptr + point_offs * M + 1, mask=point_mask, other=0).to(tl.int32)
    pc2 = tl.load(codes_ptr + point_offs * M + 2, mask=point_mask, other=0).to(tl.int32)
    pc3 = tl.load(codes_ptr + point_offs * M + 3, mask=point_mask, other=0).to(tl.int32)
    pc4 = tl.load(codes_ptr + point_offs * M + 4, mask=point_mask, other=0).to(tl.int32)
    pc5 = tl.load(codes_ptr + point_offs * M + 5, mask=point_mask, other=0).to(tl.int32)

    # Tile over centers
    for c_start in range(0, K, BLOCK_K):
        c_offs = c_start + tl.arange(0, BLOCK_K)
        c_mask = c_offs < K

        cc0 = tl.load(centers_ptr + c_offs * M + 0, mask=c_mask, other=0).to(tl.int32)
        cc1 = tl.load(centers_ptr + c_offs * M + 1, mask=c_mask, other=0).to(tl.int32)
        cc2 = tl.load(centers_ptr + c_offs * M + 2, mask=c_mask, other=0).to(tl.int32)
        cc3 = tl.load(centers_ptr + c_offs * M + 3, mask=c_mask, other=0).to(tl.int32)
        cc4 = tl.load(centers_ptr + c_offs * M + 4, mask=c_mask, other=0).to(tl.int32)
        cc5 = tl.load(centers_ptr + c_offs * M + 5, mask=c_mask, other=0).to(tl.int32)

        TABLE = 256 * 256

        idx0 = pc0[:, None] * 256 + cc0[None, :]
        dist = tl.load(dtables_ptr + 0 * TABLE + idx0)

        idx1 = pc1[:, None] * 256 + cc1[None, :]
        dist += tl.load(dtables_ptr + 1 * TABLE + idx1)

        idx2 = pc2[:, None] * 256 + cc2[None, :]
        dist += tl.load(dtables_ptr + 2 * TABLE + idx2)

        idx3 = pc3[:, None] * 256 + cc3[None, :]
        dist += tl.load(dtables_ptr + 3 * TABLE + idx3)

        idx4 = pc4[:, None] * 256 + cc4[None, :]
        dist += tl.load(dtables_ptr + 4 * TABLE + idx4)

        idx5 = pc5[:, None] * 256 + cc5[None, :]
        dist += tl.load(dtables_ptr + 5 * TABLE + idx5)

        dist = tl.where(c_mask[None, :], dist, float('inf'))

        tile_min_dist = tl.min(dist, axis=1)
        tile_min_idx = tl.argmin(dist, axis=1).to(tl.int32)
        tile_min_label = c_start + tile_min_idx

        update_mask = tile_min_dist < best_dist
        best_dist = tl.where(update_mask, tile_min_dist, best_dist)
        best_label = tl.where(update_mask, tile_min_label, best_label)

    tl.store(labels_ptr + point_offs, best_label, mask=point_mask)


# ---------------------------------------------------------------------------
# GPU tensor cache (centers + dtables persist across predict calls)
# ---------------------------------------------------------------------------
_gpu_cache: dict = {}


def _get_or_upload(key: str, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Upload numpy array to GPU, caching by (key, data_ptr, shape)."""
    cache_key = (key, array.ctypes.data, array.shape)
    cached = _gpu_cache.get(cache_key)
    if cached is not None:
        return cached
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=dtype, device='cuda')
    _gpu_cache[cache_key] = tensor
    return tensor


def _auto_batch_size(N: int, M: int) -> int:
    """Compute the max batch size that fits in available VRAM.

    Per-point VRAM:  M bytes (codes) + 4 bytes (labels) = M + 4
    We leave _VRAM_OVERHEAD_MB for PyTorch/Triton context and cached tensors.
    """
    free, total = torch.cuda.mem_get_info()
    usable = free - _VRAM_OVERHEAD_MB * 1024 * 1024
    if usable < 0:
        usable = free // 2
    bytes_per_point = M + 4  # uint8 codes + int32 label
    max_batch = max(usable // bytes_per_point, 1024)
    return min(max_batch, N)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_gpu(
    pq_codes: np.ndarray,
    centers: np.ndarray,
    dtables: np.ndarray,
    batch_size: int = 0,
) -> np.ndarray:
    """GPU-accelerated PQ assignment.

    Args:
        pq_codes: (N, M) uint8 PQ codes.
        centers: (K, M) uint8 cluster center codes.
        dtables: (M, 256, 256) float32 distance lookup tables.
        batch_size: Max points per GPU batch.
                    0 (default) = auto-detect from free VRAM.

    Returns:
        (N,) int64 cluster labels (same dtype as CPU path).
    """
    N, M = pq_codes.shape
    K = centers.shape[0]

    if M != 6:
        raise ValueError(f"Triton kernel is compiled for M=6, got M={M}")

    # Kernel assumes dtables are (M, 256, 256). Pad if k_codebook < 256.
    if dtables.shape[1] != 256 or dtables.shape[2] != 256:
        padded = np.zeros((M, 256, 256), dtype=np.float32)
        k_cb = dtables.shape[1]
        padded[:, :k_cb, :k_cb] = dtables
        dtables = padded

    # Cache centers and dtables on GPU (persist across calls)
    centers_gpu = _get_or_upload('centers', centers, torch.uint8)
    dtables_gpu = _get_or_upload('dtables', dtables, torch.float32)

    if batch_size <= 0:
        batch_size = _auto_batch_size(N, M)

    labels_out = np.empty(N, dtype=np.int64)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = pq_codes[start:end]
        n_chunk = end - start

        codes_gpu = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.uint8)).cuda()
        labels_gpu = torch.empty(n_chunk, dtype=torch.int32, device='cuda')

        _launch_kernel(codes_gpu, centers_gpu, dtables_gpu, labels_gpu, n_chunk, K, M)

        labels_out[start:end] = labels_gpu.cpu().numpy()
        del codes_gpu, labels_gpu

    return labels_out


def _launch_kernel(codes, centers, dtables, labels, N, K, M):
    BLOCK_N = 32
    BLOCK_K = 128
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)
    _pq_assign_kernel[grid](
        codes, centers, dtables, labels,
        N, K, M,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
