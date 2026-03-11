# Choosing k (Number of Clusters)

Selecting the right number of clusters is critical for balancing cluster granularity against computational cost and empty cluster waste. The `scripts/select_k.py` script automates this by sweeping over k values on a subsample of PQ codes.

## Running the sweep

```bash
python scripts/select_k.py \
    --pq-codes data/pq_codes.npy \
    --encoder models/encoder.joblib \
    --n-subsample 10000000 \
    --k-values 10000 25000 50000 100000 200000 \
    --iterations 10 \
    --output results/k_selection.csv \
    --plot results/k_selection.png
```

The script supports **checkpointing** — results are saved to the CSV after each k value. If interrupted, rerun the same command and it resumes from where it left off.

## Results on 100M Enamine REAL molecules

Benchmark configuration: AMD Ryzen 7, 64GB RAM, 10 PQk-means iterations per k.

| k | Avg Distance | Empty Clusters | Median Cluster Size | Fit Time |
|---:|---:|---:|---:|---:|
| 10,000 | 3.65 | 6.8% | 8,945 | 1.3 h |
| 25,000 | 2.74 | 13.3% | 3,673 | 3.1 h |
| 50,000 | 2.17 | 19.6% | 1,876 | 6.2 h |
| 100,000 | 1.69 | 26.6% | 956 | 12.6 h |
| 200,000 | 1.30 | 34.7% | 492 | 26.4 h |

## How to interpret these metrics

- **Avg Distance**: Mean PQ-space distance from each point to its assigned centroid. Lower is better, but diminishing returns set in quickly.
- **Empty Clusters**: Fraction of clusters with zero members. High values mean you're over-partitioning — the data doesn't have that many natural groupings.
- **Median Cluster Size**: Typical number of molecules per cluster. Determines how many molecules you see in each leaf TMAP.

## Guidelines

- **k = 50,000** is a good default — under 20% empty clusters, median size ~1,900, and the avg distance improvement starts plateauing beyond this point.
- **k = 100,000** if you need tighter clusters and can tolerate ~27% empty clusters.
- Beyond 200K, over a third of clusters are empty — diminishing returns.

## Scaling estimates

Fit time scales linearly with both n (number of molecules) and k:

| Scenario | Estimated Fit Time |
|---|---|
| 1B molecules, k=50K | ~2.6 days |
| 1B molecules, k=100K | ~5.2 days |
| 2B molecules, k=100K | ~10.5 days |

!!! note "Memory requirements"
    PQ codes are (n, 42) uint8 arrays. At 1B molecules this is ~42 GB. Ensure your system has sufficient RAM, or use the `--n-subsample` flag to fit on a representative subset and then assign in chunks.
