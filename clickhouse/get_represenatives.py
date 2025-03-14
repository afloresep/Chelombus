import time
import pandas as pd
import clickhouse_connect
import numpy as np
from scipy.spatial.distance import cdist
from multiprocessing import Pool

import time
import pandas as pd
import clickhouse_connect
import numpy as np
import os
from scipy.spatial.distance import cdist
import sys
import pandas as pd
import clickhouse_connect
client = clickhouse_connect.get_client(host='localhost', port=8123)

def process_cluster_worker(worker_index, chunk_size=1250, query_chunk_size=125):
    """
    Process a range of cluster IDs for a given worker index.
    
    Each worker:
      - Creates its own ClickHouse connection.
      - Processes cluster IDs from chunk_size*worker_index to chunk_size*(worker_index+1).
      - Divides the IDs into smaller chunks of query_chunk_size.
      - Queries the database for each chunk and calculates the median-based representative.
      - Writes the resulting representatives to a CSV file, named by the range of cluster IDs.
    
    Parameters:
        worker_index (int): The index for this worker, which determines the range of cluster IDs.
        chunk_size (int, optional): The number of cluster IDs to process for each worker. Default is 1250.
        query_chunk_size (int, optional): The number of cluster IDs to query at once. Default is 125.
    """
    from tqdm import tqdm
    # Each worker creates its own ClickHouse connection
    client = clickhouse_connect.get_client(host='130.92.106.202', port=8123)
    
    # Define the range of cluster IDs for this worker
    start_cluster_id = chunk_size * worker_index
    end_cluster_id = chunk_size * (worker_index + 1)
    cluster_ids = list(range(start_cluster_id, end_cluster_id))
    print(f"Worker {worker_index}: [", cluster_ids[0], ",", cluster_ids[-1], "]")
    # Process the cluster IDs in chunks of size query_chunk_size
    num_chunks = len(cluster_ids) // query_chunk_size
    pbar = tqdm(total=chunk_size, desc="Processing clusters") 
    for chunk_index in range(num_chunks):
        # Extract the current chunk of cluster IDs
        cluster_id_chunk = cluster_ids[chunk_index * query_chunk_size : (chunk_index + 1) * query_chunk_size]
        
        # Create a comma-separated string of quoted cluster IDs for the SQL query
        import time
        s = time.time()
        id_list = ",".join(f"'{cid}'" for cid in cluster_id_chunk)
        query = f"SELECT * FROM clustered_enamine_10b WHERE cluster_id IN ({id_list})"
        result = client.query(query)
        e = time.time()
        # print(f"Result queried for {cluster_id_chunk[0]}, {cluster_id_chunk[-1]} took {(e-s):.2f} for {len(result.result_rows)} rows") 
        # Create a DataFrame from the query result
        df = pd.DataFrame(result.result_rows, columns=["smiles", "PCA_1", "PCA_2", "PCA_3", "cluster_id"])
        representative_medians = []

        # For each cluster, calculate the median coordinates and select the closest point
        for cluster_id, sub_df in df.groupby("cluster_id"):
            median_coords = sub_df[['PCA_1', 'PCA_2', 'PCA_3']].median().to_numpy()
            distances = cdist([median_coords], sub_df[['PCA_1', 'PCA_2', 'PCA_3']], metric='euclidean')[0]
            rep_index = np.argmin(distances)
            representative = sub_df.iloc[rep_index]
            representative_medians.append(representative)
        
        # Write the median representatives to a CSV file
        rep_median_df = pd.DataFrame(representative_medians)
        output_filename = f"/mnt/samsung_2tb/enamine_db_10b_processed/representatives/cluster_id_representatives_{cluster_id_chunk[0]}-{cluster_id_chunk[-1]}.csv"
        rep_median_df.to_csv(output_filename, index=False)
        pbar.update(query_chunk_size) 

if __name__ == '__main__':
    # Number of parallel workers to run concurrently
    process_cluster_worker(worker_index=0, chunk_size=125000, query_chunk_size=500)

    # Can also be run in parallel, but main overhead is in the loading the data from clickhouse
    # num_parallel_workers = 5
    # for total in range(num_parallel_workers, 100, num_parallel_workers):
    #     worker_indices = list(range(total-num_parallel_workers, total, 1))
    #     # Launch parallel processing using a pool of workers
    #     with Pool(processes=num_parallel_workers) as pool:
    #         pool.map(process_cluster_worker, worker_indices)