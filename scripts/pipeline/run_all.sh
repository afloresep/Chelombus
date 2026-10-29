#!/usr/bin/env bash
# ============================================================================
#  Build the full nested-TMAP website for a ~1-billion-molecule library.
# ============================================================================
#
#  There are only TWO heavy compute steps, then four map-building steps:
#
#     STEP 1  TRAIN    learn the encoder + the K cluster centres, on a SAMPLE
#     STEP 2  ASSIGN   give every one of the ~1B molecules its cluster (full GPU)
#     STEP 3  GROUP    rewrite the data so each cluster's molecules sit together
#     STEP 4  PICK     choose one "typical" molecule to represent each cluster
#     STEP 5  MAP-TOP  build the primary map  (one dot per cluster)
#     STEP 6  MAP-SUB  build the secondary maps (one map per cluster, drill-down)
#
#  Steps 1-4 run in the GPU/compute conda env ("pqkmeans").
#  Steps 5-6 run in the visualisation conda env ("tmap2").
#
#  Everything is RESUMABLE: re-run this file and any step whose output already
#  exists is skipped. Safe to Ctrl-C and restart.
# ----------------------------------------------------------------------------
set -uo pipefail

# --- Which python to use for each half (edit these two paths if needed) ------
PQ=/home/afloresep/miniforge3/envs/pqkmeans/bin/python   # compute / GPU
TM=/home/afloresep/miniforge3/envs/tmap2/bin/python       # tmap2 viz

# --- Where the code lives ----------------------------------------------------
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # this checkout
P=$REPO/scripts/pipeline   # all pipeline scripts (01..06 + helpers) live here

# --- Your data and where results go (EDIT THESE) -----------------------------
SRC=/mnt/10tb_hdd/cleaned_enamine_10b/output_file_0.cxsmiles  # input file(s)
ROOT=/mnt/10tb_hdd/enamine_1b                                  # all output here
SMILES_COL=1     # 0-based column that holds the SMILES in $SRC
ID_COL=0         # 0-based column that holds the molecule id, or empty (see note below)
K=50000          # number of clusters
TRAIN_SAMPLE=100000000   # molecules used to TRAIN (you don't need all 1B)
MAX_CLUSTERS=1000        # how many drill-down maps to build in this run
#
# NOTE on columns: if your file is just ONE SMILES per line with nothing else,
# set SMILES_COL=0 and ID_COL= (empty); the molecules will then be
# auto-numbered (0,1,2,...). Set ID_COL only when your file really has an id
# column you want to keep.
# ----------------------------------------------------------------------------

SAMPLE=$ROOT/train_sample.smi
mkdir -p "$ROOT" "$ROOT/site"
say(){ echo "[$(date '+%F %T')] === $* ==="; }
die(){ echo "[$(date '+%F %T')] !!! FAILED: $* - stopping." >&2; exit 1; }

# ── STEP 1: TRAIN the models on a sample (GPU) ───────────────────────────────
# 01_train_models.py samples MQN fingerprints, trains the PQ encoder, then trains
# the K=$K clusterer, writing $ROOT/models/{encoder,clusterer}.joblib.
# (As its last internal stage it also assigns the *sample*; we ignore that -
#  the real assignment of all 1B molecules is STEP 2.)
if [ ! -f "$ROOT/models/clusterer.joblib" ]; then
  if [ ! -f "$SAMPLE" ]; then
    say "STEP 1a: taking a ${TRAIN_SAMPLE}-line training sample from $SRC"
    head -n "$TRAIN_SAMPLE" "$SRC" > "$SAMPLE" || die "sample"
  fi
  say "STEP 1b: training encoder + clusterer (K=$K) on the sample"
  "$PQ" "$P/01_train_models.py" --input "$SAMPLE" --output "$ROOT" \
        --n-clusters "$K" --smiles-col "$SMILES_COL" --resume || die "train"
else
  say "STEP 1: models already trained, skipping"
fi
say "MODELS READY: $ROOT/models/{encoder,clusterer}.joblib"

# ── STEP 2: ASSIGN a cluster to every molecule (full ~1B GPU pass) ───────────
# Resumable: each output chunk that already exists is skipped on re-run.
say "STEP 2: assigning all molecules to clusters (full pass, --resume)"
"$PQ" "$P/02_assign_clusters.py" --input "$SRC" --output "$ROOT/labels" \
      --encoder "$ROOT/models/encoder.joblib" \
      --clusterer "$ROOT/models/clusterer.joblib" \
      --smiles-col "$SMILES_COL" ${ID_COL:+--id-col "$ID_COL"} \
      --device gpu --resume || die "assign"

# ── STEP 3: GROUP molecules by cluster (DuckDB) ──────────────────────────────
if [ ! -f "$ROOT/clustered/_DONE" ]; then
  say "STEP 3: grouping molecules by cluster"
  "$PQ" "$P/03_group_by_cluster.py" --labels "$ROOT/labels" --output "$ROOT/clustered" \
        --bucket-size 100 --memory-limit 40GB || die "partition"
  touch "$ROOT/clustered/_DONE"
else
  say "STEP 3: clustered/_DONE exists, skipping"
fi

# ── STEP 4: PICK one representative molecule per cluster ─────────────────────
if [ ! -f "$ROOT/representatives.csv" ]; then
  say "STEP 4: picking one representative per cluster"
  "$PQ" "$P/04_pick_representatives.py" --clustered "$ROOT/clustered" \
        --output "$ROOT/representatives.csv" || die "representatives"
else
  say "STEP 4: representatives.csv exists, skipping"
fi

# ── STEP 5: PRIMARY map (one dot per cluster) ────────────────────────────────
if [ ! -f "$ROOT/site/representatives_TMAP.html" ]; then
  say "STEP 5: building the primary map"
  "$TM" "$P/05_primary_map.py" --representatives "$ROOT/representatives.csv" \
        --output "$ROOT/site/representatives_TMAP.html" || die "primary_tmap"
else
  say "STEP 5: primary map exists, skipping"
fi
say "PRIMARY MAP READY: $ROOT/site/representatives_TMAP.html"

# ── STEP 6: SECONDARY maps (drill-down, one per cluster) ─────────────────────
# Build maps for ~$MAX_CLUSTERS clusters now, using 4 workers on THIS machine.
# Resumable, so you can raise MAX_CLUSTERS later to add more without redoing any.
# To build them across MANY machines (a real HPC cluster) use secondary_worker.py
# instead - it takes a plain list of cluster IDs. See the README.
say "STEP 6: building secondary maps (4 local workers, ~$MAX_CLUSTERS clusters)"
N_BUCKETS=$(ls -d "$ROOT/clustered"/bucket=* 2>/dev/null | wc -l)
CAP=$(( (MAX_CLUSTERS + 99) / 100 ))          # bucket = cluster_id // 100
[ "$CAP" -lt "$N_BUCKETS" ] && N_BUCKETS=$CAP
W=4; STEP=$(( (N_BUCKETS + W - 1) / W )); pids=()
for ((i=0; i<W; i++)); do
  START=$(( i * STEP )); END=$(( START + STEP ))
  [ "$START" -ge "$N_BUCKETS" ] && break
  "$TM" "$P/06_secondary_maps.py" --clustered "$ROOT/clustered" --output "$ROOT/site" \
        --max-points 2000 --start-bucket "$START" --end-bucket "$END" \
        > "$ROOT/site/_worker_${i}.log" 2>&1 &
  pids+=($!); echo "  worker $i: buckets [$START,$END)  pid $!"
done
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ "$fail" -eq 0 ] || die "secondary_tmaps (see $ROOT/site/_worker_*.log)"

N_MAPS=$(ls "$ROOT/site"/cluster_*.html 2>/dev/null | wc -l)
say "DONE. $N_MAPS secondary maps built. Serve $ROOT/site/ with any web server."
