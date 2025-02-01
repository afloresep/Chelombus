import math
import heapq
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import clickhouse_connect
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from typing import List, Tuple, Optional

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
        
def compute_similarity_distance(query_fp: np.ndarray, smiles: str, metric: str) -> Tuple[str, Optional[float]]:
    """
    #TODO: Add different fingerprints methods
    Compute the similarity distance of `smiles` relative to `query_fp`.
    using 'tanimoto' or 'manhattan' distances
    
    :param query_fp (np.ndarray): MQN fingerprint for the query molecule.
    :param smiles (str): SMILES for the target molecule.
    
    :return Tuple[str, Optional[float]]:
            - SMILES string.
            - Tanimoto distance (1 - similarity) or None if invalid SMILES.
    """

    # Calculate fingerprint
    #TODO: This should be using `return_fingerprint_method` from helper functions to choose the fingerprint method
    # instead of assuming its mqn
    mol_fp = compute_mqn(smiles)
    if mol_fp is None:
        return smiles, None
    
    if metric.lower()=="tanimoto":
        #TODO: What is the difference between distance and similarity?
        dist = _tanimoto_distance(query_fp, mol_fp)
        return smiles, dist
    elif metric.lower()=="manhattan":
        dist = _manhattan_distance(query_fp, mol_fp)
        return smiles, dist 
    else: 
        raise ValueError("only 'tanimoto' or 'manhattan' distances can be passed as\
                         metric option. instead got: ", metric)


def _tanimoto_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute Tanimoto distance for continuous-valued vectors.
    :param fp1: (np.ndarray): Fingerprint of first molecule.
    :param fp2: (np.ndarray): Fingerprint of second molecule.
    :return float: Tanimoto distance 

    Tanimoto distances are suitable for binary vectors which is not the case for 
    fingerprints like MQN, SMIfp etc. In that case, Manhattan distance should be
    used. 
    
    """
    dot_product = np.dot(fp1, fp2)
    norm0_sq = np.dot(fp1, fp1)
    norm1_sq = np.dot(fp2, fp2)
    denominator = (norm0_sq + norm1_sq - dot_product)
    if denominator == -1:
        return -1.0
    return dot_product / denominator

def _manhattan_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute Manhattan distance for continuous-valued vectors.
    :param fp1: (np.ndarray): Fingerprint of first molecule.
    :param fp2: (np.ndarray): Fingerprint of second molecule.
    :return float: Manhattan distance 

    Manhattan distances are suitable for continuous-valued vectors like 
    MQN or SMIfp fingerprint. In the case of a binary fingerprint/vector
    Tanimoto distance should be used. 
    """
    distance = np.sum(np.abs(fp1 - fp2))
    return distance


def find_neighbors(
    query_smiles: str,
    table_name: str = "clustered_enamine",
    top_n: int = 10_000,
    chunk_size: int = 250_000_000,
    host: str = "localhost",
    port: int = 8123
) -> pd.DataFrame:
    """
    Compute Tanimoto distances for a query molecule across a large table.
    Iterates over the table in chunks so as not to load everything into memory.
    
    Args:
        query_smiles (str): Query molecule in SMILES format.
        table_name (str): Name of the table in ClickHouse containing SMILES data.
        top_n (int): Number of nearest neighbors (lowest Tanimoto distance) to keep.
        chunk_size (int): Number of rows to query at a time from the database.
        host (str): Host where the ClickHouse server is running.
        port (int): Port of the ClickHouse server.
    
    Returns:
        pd.DataFrame: DataFrame of the top-N nearest neighbors with columns:
                      ["smiles", "tanimoto_distance"].
    """
    # Connect to ClickHouse
    client = clickhouse_connect.get_client(host=host, port=port)

    # Count total rows in the table
    count_query = f"SELECT count() FROM {table_name}"
    total_rows = client.query(count_query).result_rows[0][0]
    print(f"Total rows in table '{table_name}': {total_rows:,}")

    # Calculate the query molecule's MQN fingerprint
    query_fp = compute_mqn(query_smiles)
    if query_fp is None:
        raise ValueError("Invalid query SMILES provided.")

    # Decide how many CPU processes to use (75% of available CPU cores)
    processes = cpu_count() # Use all CPU 
    print(f"Using {processes} worker processes out of {cpu_count()} total cores.")

    # This heap will store our best (lowest-distance) molecules.
    # We'll keep a max-heap of (distance, smiles) so that
    # the largest distance can be popped when we exceed top_n.
    neighbors_heap: List[Tuple[float, str]] = []

    # Prepare the partial function for parallel processing

    func = partial(compute_similarity_distance, query_fp, metric="manhattan")

    # Number of chunks
    total_chunks = math.ceil(total_rows / chunk_size)

    with Pool(processes=processes) as pool, tqdm(
        desc="Molecules processed",
        total=total_rows,
        unit="mol"
    ) as pbar:
        for chunk_idx in range(total_chunks):
            offset = chunk_idx * chunk_size

            if offset > 1_000_000:
                break

            query = (
                f"SELECT smiles FROM {table_name} "
                f"LIMIT {chunk_size} OFFSET {offset}"
            )
            query_result = client.query(query)
            chunk_data = query_result.result_rows

            # If no rows returned, break early
            if not chunk_data:
                break

            # Convert result to a list of SMILES
            smiles_list = [row[0] for row in chunk_data]

            # Compute Tanimoto distances in parallel
            distances = pool.map(func, smiles_list)

            # Update progress bar by number of molecules processed in this chunk
            pbar.update(len(distances))

            # Update the max-heap with new results
            for smi, dist in distances:
                # Skip invalid molecules
                if dist is None:
                    continue

                # If we haven't reached top_n, just push
                if len(neighbors_heap) < top_n:
                    heapq.heappush(neighbors_heap, (-dist, smi))
                else:
                    # If the new distance is smaller than the largest in the heap
                    # (heap stores negative distance for max-heap behavior)
                    if dist < -neighbors_heap[0][0]:
                        # Pop the largest distance
                        heapq.heapreplace(neighbors_heap, (-dist, smi))
            del chunk_data, smiles_list, distances

    # Convert final heap to a sorted list (ascending by distance)
    final_results = []
    while neighbors_heap:
        neg_dist, smi = heapq.heappop(neighbors_heap)
        final_results.append((smi, -neg_dist))
    final_results.reverse()  # Because we popped in ascending order of negative distance

    # Create a DataFrame
    df_results = pd.DataFrame(final_results, columns=["smiles", "tanimoto_distance"])
    return df_results

def main():
    """
    Example main function to demonstrate usage.
    """
    query_molecule = "CN(C(=O)CCC1CCCCN1C(=O)C1=NC=CN1C)C1CC2CC2C1"

    print("Starting nearest neighbor search...")
    nearest_neighbors_df = find_neighbors(
        query_smiles=query_molecule,
        table_name="clustered_enamine",  # Adjust if your table name differs
        top_n=120_000,
        chunk_size=50_000,  # Adjust chunk size as per memory constraints
        host="localhost",
        port=8123
    )
    print("Search completed!")

    # Write the top neighbors to a CSV file
    output_csv = "top_neighbors_test.csv"
    nearest_neighbors_df.to_csv(output_csv, index=False)
    print(f"Top 10,000 neighbors saved to '{output_csv}'.")

if __name__ == "__main__":
    main()
