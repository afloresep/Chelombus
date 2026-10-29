# Make a TMAP website for a very large molecule library

This folder builds an interactive "map of molecules" website for a huge set of
molecules (it was tested on ~1 billion). You can zoom from a top map of all
clusters down into the molecules inside each cluster.

How it works, in one line:

```
SMILES  ->  fingerprint  ->  clusters  ->  top map (1 dot per cluster)
                                        ->  detail map (1 map per cluster)
```

You do not need to understand the math. You only need to run a few commands.

---

## 1. What your input file must look like

The simplest input is a plain text file with **one SMILES per line** and nothing
else. No header line. No extra columns. Example:

```
CCO
c1ccccc1
CC(=O)O
```

That is all you need. If your file is like this, you are done — skip to step 3.

**If your file has more columns** (for example an id and a SMILES, separated by
a space or a TAB), that is also fine. You just tell the scripts which column is
the SMILES. Columns are counted from 0. Example file:

```
ZINC0001    CCO
ZINC0002    c1ccccc1
```

Here the id is column 0 and the SMILES is column 1.

**About ids.** Every molecule gets an id, shown on the map. If your file has an
id column, the scripts keep it. If your file has **no** id, the scripts make one
up automatically (just the line number: 0, 1, 2, ...). You do not have to create
ids yourself.

The input can be **one file or a whole folder of files** — point the scripts at
the folder and they read every file inside.

---

## 2. Two software environments

The work has two halves, and each half needs different packages. On the test
machine they were kept in two separate conda environments:

| Half | What it does | Packages it needs |
|------|--------------|-------------------|
| compute | clustering on the GPU | `chelombus`, `pqkmeans`, `torch`+CUDA, `rdkit`, `pyarrow`, `duckdb` |
| viz | drawing the maps | `tmap2` (run `pip install tmap2`), `rdkit`, `duckdb`, `pandas` |

The maps need the **`tmap2`** package. Install it with:

```bash
pip install tmap2
```

(There is an old package called `tmap` — that is a different, older one. Use
`tmap2`.)

In the commands below we use two python paths, one per environment:

```bash
PQ=/home/.../envs/pqkmeans/bin/python    # compute (GPU) half
TM=/home/.../envs/tmap2/bin/python       # viz half
```

Change these two paths to match your machine.

---

## 3. The easy way: edit one file and run it

`run_all.sh` runs the whole thing for you. Open it, change the few settings at
the top (the two python paths, your input file, the output folder, and which
column is the SMILES), then run:

```bash
bash scripts/pipeline/run_all.sh
```

It is **safe to stop and restart** — it skips any step that is already finished.
That is the whole pipeline. The sections below explain each step, in case you
want to run them one at a time or change something.

---

## 4. The steps, one by one

There are 6 steps. The first two are the heavy ones (they use the GPU).

### Step 1 — Train (learn the clusters)

You do **not** need all 1 billion molecules to learn the clusters. A sample is
enough (the example uses 100 million). This one command learns everything and
saves two model files into `OUT/models/`:

```bash
$PQ scripts/pipeline/01_train_models.py \
    --input  sample.smi \
    --output OUT \
    --n-clusters 50000 \
    --smiles-col 0          # use 1 if SMILES is the 2nd column, etc.
```

Result: `OUT/models/encoder.joblib` and `OUT/models/clusterer.joblib`.

### Step 2 — Assign (give every molecule its cluster)

This is the one pass over **all** your molecules. It uses the GPU. It writes
parquet files with three columns: `id`, `smiles`, `cluster_id`.

If your file has no id column, leave out `--id-col 0` (molecules are then
numbered 0, 1, 2, ...). `--resume` skips chunks already done if you restart.

```bash
$PQ scripts/pipeline/02_assign_clusters.py \
    --input  all_molecules.smi \
    --output OUT/labels \
    --encoder   OUT/models/encoder.joblib \
    --clusterer OUT/models/clusterer.joblib \
    --smiles-col 0 \
    --id-col 0 \
    --device gpu \
    --resume
```

### Step 3 — Group (put each cluster together)

```bash
$PQ scripts/pipeline/03_group_by_cluster.py \
    --labels OUT/labels --output OUT/clustered --memory-limit 32GB
```

### Step 4 — Pick (one example molecule per cluster)

```bash
$PQ scripts/pipeline/04_pick_representatives.py \
    --clustered OUT/clustered --output OUT/representatives.csv
```

### Step 5 — Top map (one dot per cluster)

```bash
$TM scripts/pipeline/05_primary_map.py \
    --representatives OUT/representatives.csv \
    --output OUT/site/representatives_TMAP.html
```

### Step 6 — Detail maps (one map per cluster)

```bash
$TM scripts/pipeline/06_secondary_maps.py \
    --clustered OUT/clustered --output OUT/site --max-points 2000
```

When everything is done, put the folder `OUT/site/` on any web server (it is
just static HTML files, no backend needed). Open `representatives_TMAP.html`.
Clicking "Open cluster map" on a cluster opens its detail map.

---

## 5. Building the detail maps on many machines

Step 6 is the slow part: there can be tens of thousands of clusters, and each
one becomes its own small map. This part does **not** use the GPU — it uses CPU
cores — so the way to make it fast is to spread it over many cores or many
machines.

The unit of work is simply **a list of cluster ids**. Give each machine a
different list. Use `secondary_worker.py` for this:

```bash
# 1. Make a list of all cluster ids (run once):
duckdb -noheader -list \
  -c "SELECT DISTINCT cluster_id FROM read_parquet('OUT/clustered/**/*.parquet')" \
  > all_ids.txt

# 2. Split it into pieces (here, 8 pieces):
split -n l/8 all_ids.txt chunk_

# 3. Run one worker per piece (on different machines, or with SLURM / & / etc.):
$TM scripts/pipeline/secondary_worker.py \
    --clustered OUT/clustered --output OUT/site \
    --cluster-ids-file chunk_aa --max-points 2000
```

Workers do not talk to each other and skip any map that already exists, so it is
safe to run them at the same time and to restart them. (To build every cluster
on one machine, use `--all` instead of `--cluster-ids-file`.)

---

## 6. Common things you may want to change

**The molecule id shown on the maps.** The id is whatever you chose in step 2
(`--id-col`, or the auto line number). To change how it is *labelled* on the
card, edit the `configure_card(...)` call in `05_primary_map.py` (top map) and
the `add_label(...)` / `add_smiles(...)` calls in `06_secondary_maps.py` and
`secondary_worker.py` (detail maps).

**Add more colour layers or labels.** All maps share one helper,
`add_property_colors(...)` in `_common.py`. Add a line there and every map gets
the new colour layer. Continuous numbers (like weight, logP) use a colour
gradient; small whole-number counts (like number of rings) use a discrete legend.

**Number of clusters.** Change `--n-clusters` in step 1 (and `K` in
`run_all.sh`). For ~1 billion molecules, 50000 is a good starting value.

**Which column is the SMILES / id.** `--smiles-col` and `--id-col` everywhere.
Counting starts at 0.

---

## 7. Notes

- The id travels **with** each molecule through every step, so a molecule and
  its label can never get mixed up — even if some SMILES fail to read, those
  rows are simply dropped together with their ids (nothing shifts).
- The top map and the detail maps show the same colour layers, so they are easy
  to read together. The top map uses an MQN-based layout; the detail maps use a
  Morgan/Jaccard layout (better for seeing fine structure inside one cluster).
- Tested on one machine (RTX 4090 GPU, 16 cores, 62 GB RAM) on ~956 million
  molecules: it produced a top map of ~49,862 clusters whose buttons open the
  per-cluster detail maps.
