import pandas as pd
import numpy as np
from multiprocessing import Pool
import dask.dataframe as dd
import math
import heapq
import numpy as np
import pandas as pd
import joblib
import time
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from typing import List, Tuple, Optional

# ------------------------------------------------------------------------------

def compute_mqn(smiles: str) -> Optional[np.ndarray]:
    """
    Compute MQN fingerprint for a single SMILES string.
    
    :param smiles (str): SMILES representation of the molecule.
    
    :return Optional[np.ndarray]: MQN fingerprint (as numpy array),
     or None if the molecule is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return np.array(rdMolDescriptors.MQNs_(mol))
    except Exception as e:
        # In production, consider logging instead of printing.
        print(f"Error processing SMILES '{smiles}': {e}")
        return None

# ------------------------------------------------------------------------------
# (c) Function to compute Manhattan distance.
def compute_distance(item):
    """
    Given a tuple (smiles, fingerprint) for one molecule,
    compute and return (smiles, Manhattan distance)
    relative to the global target fingerprint.
    """
    smiles, fp = item
    # Use the global target_fp computed from the input SMILES.
    return (smiles, np.sum(np.abs(fp - target_fp)))


# ------------------------------------------------------------------------------
# (c) Function to compute Manhattan distance.
def compute_distance_3d(item):
    """
    Given a tuple (smiles, fingerprint) for one molecule,
    compute and return (smiles, Manhattan distance)
    relative to the global target fingerprint.
    """
    smiles, fp = item
    fp = ipca.transform(fp.reshape(1, -1)) #compute fp
    # Use the global target_fp computed from the input SMILES.

    return (smiles, int(np.sum(np.abs(fp - target_fp_3d))))

# ------------------------------------------------------------------------------
def process_file(file_path):
    """
    Reads a Parquet file and computes the Manhattan distances for all its molecules.
    Returns a list of (smiles, distance) tuples.
    """
    df = pd.read_parquet(file_path)
    df_len = len(df)  # To update the progress bar
    # Assumes that the first column is SMILES and columns 1-42 are the fingerprint.
    smiles_list = df.iloc[:, 0].tolist()
    fingerprint_array = df.iloc[:, 1:43].values  # shape: (num_molecules, 42)
    items = list(zip(smiles_list, fingerprint_array))
    
    # Compute distances using list comprehension.
    results = [compute_distance(item) for item in items]
    return results, df_len


# ------------------------------------------------------------------------------
def process_file_3d(file_path):
    """
    Reads a Parquet file and computes the Manhattan distances for all its molecules.
    Returns a list of (smiles, distance) tuples.
    """
    df = pd.read_parquet(file_path)
    df_len = len(df)
    # Assumes that the first column is SMILES and columns 1-42 are the fingerprint.
    smiles_list = df.iloc[:, 0].tolist()
    fingerprint_array = df.iloc[:, 1:43].values  # shape: (num_molecules, 42)
    items = list(zip(smiles_list, fingerprint_array))
    
    # Compute distances using list comprehension.
    results = [compute_distance_3d(item) for item in items]
    del smiles_list, fingerprint_array, items, df
    return results, df_len

# ------------------------------------------------------------------------------
def main(input_smiles, parquet_dir, top_n=10000):
    """
    For a given input SMILES, iterates through all Parquet files in the directory,
    computes Manhattan distances for each molecule's fingerprint, and maintains
    a global heap of the top_n (lowest distance) molecules.
    
    Returns a sorted list (ascending by Manhattan distance) of tuples: (smiles, distance).
    """
    global target_fp  # Make target_fp available to the compute_distance function
    target_fp = compute_mqn(input_smiles)

    global ipca
    ipca = joblib.load('/mnt/10tb_hdd/clustered_enamine_database-copy/ipca_model.joblib') 
    global target_fp_3d
    target_fp_3d = ipca.transform(target_fp.reshape(1,-1))
    import glob
    # Use glob to list all Parquet files in the directory.
    file_list = glob.glob(parquet_dir)
    
    # global_heap will store tuples of the form (-distance, smiles).
    # We use negative distances so that heapq (a min-heap) behaves like a max-heap.
    global_heap = []
    
    accumulative_rows = 0
    i = 0
    print("Starting search...")
    for file_path in file_list:
        i +=1
        try:
            # Compute distances for molecules in this file.
            results, rows_processed = process_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
        accumulative_rows = rows_processed + accumulative_rows
        print(f"\r Computed: {accumulative_rows:,}/{6742582161:,}. Steps: {i}/{len(file_list)}", flush=True, end="") 
        # Update the global heap with results from this file.
        for smiles, dist in results:
            candidate = (-dist, smiles)  # invert the distance for max-heap behavior
            if len(global_heap) < top_n:
                heapq.heappush(global_heap, candidate)
            else:
                # global_heap[0] is the worst (largest) distance among the current top_n.
                if candidate > global_heap[0]:
                    heapq.heapreplace(global_heap, candidate)
        del results, smiles, dist 
        
    # Convert heap items back to (smiles, distance) with positive distances.
    top_results = [(smiles, -neg_dist) for (neg_dist, smiles) in global_heap]
    # Sort the final list by Manhattan distance in ascending order.
    top_results.sort(key=lambda x: x[1])
    
    return top_results

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    s = time.time()
    # Input SMILES for which you want to find similar molecules.
    input_smiles ="O=C(NC12CCC(C(=O)N3CCC(CO)=C(F)C3)(CC1)C2)C1=C(Cl)C=CS1"
    
    # Path to the Parquet file.
    parquet_file = "/mnt/10tb_hdd/enamine_fingerprints/output_file_*/batch_parquet/fingerprints_chunk_*.parquet"  # Replace with your actual Parquet file path.
    
    # Number of top results to return.
    top_n = 15_000
    
    # Run the main function.
    closest_molecules = main(input_smiles, parquet_file, top_n)

    # Optionally, save the results to a CSV file.
    results_df = pd.DataFrame(closest_molecules, columns=["SMILES", "Manhattan_Distance"])
    results_df.to_csv("top_100_closest_molecules.csv", index=False)
    
    # Also, print a few top results.
    print("Top closest molecules:")
    for smiles, distance in closest_molecules[:10]:
        print(f"SMILES: {smiles}, Manhattan Distance: {distance}")

    e = time.time()
    print(f"{e - s} time")
