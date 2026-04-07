"""GPU-accelerated PQ assignment using Triton kernels.

Provides a drop-in replacement for _predict_numba that runs on CUDA GPUs.
The kernel computes symmetric PQ distance via lookup tables and maintains
an online argmin, never materializing the N x K distance matrix.

Supports any number of subvectors M via tl.static_range compile-time
unrolling — Triton JIT-compiles a specialized kernel per M value.

VRAM budget per call
--------------------
Resident (cached, allocated once):
    centers:  K x M bytes
    dtables:  M x 256 x 256 x 4 bytes

Per-batch (freed after each batch):
    codes:    batch_n x M bytes          (1 byte per subvector)
    labels:   batch_n x 4 bytes          (int32)
    -> (M + 4) bytes per point

So for a given free VRAM of F bytes, max batch ~ F / (M + 4).
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
    # int64 offsets: point_offs * M overflows int32 when N > ~357M
    point_offs = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    point_mask = point_offs < N

    best_dist = tl.full((BLOCK_N,), float('inf'), dtype=tl.float32)
    best_label = tl.zeros((BLOCK_N,), dtype=tl.int32)

    TABLE: tl.constexpr = 256 * 256

    # Tile over centers
    for c_start in range(0, K, BLOCK_K):
        c_offs = c_start + tl.arange(0, BLOCK_K)
        c_mask = c_offs < K

        dist = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for m in tl.static_range(M):
            pc = tl.load(codes_ptr + point_offs * M + m, mask=point_mask, other=0).to(tl.int32)
            cc = tl.load(centers_ptr + c_offs * M + m, mask=c_mask, other=0).to(tl.int32)
            idx = pc[:, None] * 256 + cc[None, :]
            dist += tl.load(dtables_ptr + m * TABLE + idx)

        dist = tl.where(c_mask[None, :], dist, float('inf'))

        tile_min_dist = tl.min(dist, axis=1)
        tile_min_idx = tl.argmin(dist, axis=1).to(tl.int32)
        tile_min_label = c_start + tile_min_idx

        update_mask = tile_min_dist < best_dist
        best_dist = tl.where(update_mask, tile_min_dist, best_dist)
        best_label = tl.where(update_mask, tile_min_label, best_label)

    tl.store(labels_ptr + point_offs, best_label, mask=point_mask)


# GPU tensor cache (centers + dtables persist across predict calls)
_gpu_cache: dict = {}


def _get_or_upload(key: str, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Upload numpy array to GPU, caching by key with content comparison.

    Stores one tensor per key.  On a cache hit the stored reference array is
    compared element-wise to *array*; a mismatch triggers a re-upload.
    This avoids the old scheme of caching by memory address, which silently
    returned stale tensors when Python reused a freed address.
    """
    arr = np.ascontiguousarray(array)
    entry = _gpu_cache.get(key)
    if entry is not None:
        ref, tensor = entry
        if ref.shape == arr.shape and np.array_equal(ref, arr):
            return tensor
    tensor = torch.from_numpy(arr).to(dtype=dtype, device='cuda')
    _gpu_cache[key] = (arr.copy(), tensor)
    return tensor


def _auto_batch_size(N: int, M: int) -> int:
    """Compute the max batch size that fits in available VRAM.

    Per-point VRAM:  M bytes (codes) + 4 bytes (labels) = M + 4
    We leave _VRAM_OVERHEAD_MB for PyTorch/Triton context and cached tensors.
    """
    free, _total = torch.cuda.mem_get_info()
    usable = free - _VRAM_OVERHEAD_MB * 1024 * 1024
    if usable < 0:
        usable = free // 2
    bytes_per_point = M + 4  # uint8 codes + int32 label
    max_batch = max(usable // bytes_per_point, 1024)
    return min(max_batch, N)


# Public API

def predict_gpu(
    pq_codes: np.ndarray,
    centers: np.ndarray,
    dtables: np.ndarray,
    batch_size: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """GPU-accelerated PQ assignment.

    Args:
        pq_codes: (N, M) uint8 PQ codes.
        centers: (K, M) uint8 cluster center codes.
        dtables: (M, 256, 256) float32 distance lookup tables.
        batch_size: Max points per GPU batch.
                    0 (default) = auto-detect from free VRAM.
        verbose: Print per-batch progress (useful for billion-scale runs).

    Returns:
        (N,) int64 cluster labels (same dtype as CPU path).
    """
    import time as _time

    N, M = pq_codes.shape
    K = centers.shape[0]

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

    # Adaptive BLOCK_K: larger M means more registers per subvector,
    # so reduce BLOCK_K to avoid register spill.
    if M <= 8:
        BLOCK_K = 128
    elif M <= 32:
        BLOCK_K = 64
    else:
        BLOCK_K = 32
    BLOCK_N = 32

    n_batches = (N + batch_size - 1) // batch_size
    t0 = _time.time()

    for batch_idx, start in enumerate(range(0, N, batch_size)):
        end = min(start + batch_size, N)
        chunk = pq_codes[start:end]
        n_chunk = end - start

        codes_gpu = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.uint8)).cuda()
        labels_gpu = torch.empty(n_chunk, dtype=torch.int32, device='cuda')

        grid = ((n_chunk + BLOCK_N - 1) // BLOCK_N,)
        _pq_assign_kernel[grid](
            codes_gpu, centers_gpu, dtables_gpu, labels_gpu,
            n_chunk, K, M,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        labels_out[start:end] = labels_gpu.cpu().numpy()
        del codes_gpu, labels_gpu

        if verbose and n_batches > 1:
            elapsed = _time.time() - t0
            rate = end / elapsed
            eta = (N - end) / rate if rate > 0 else 0
            print(f"    batch {batch_idx+1}/{n_batches}  "
                  f"{end:,}/{N:,} ({end/N*100:.0f}%)  "
                  f"rate={rate:,.0f} pts/s  "
                  f"ETA={int(eta//60)}m{int(eta%60)}s",
                  flush=True)

    return labels_out
