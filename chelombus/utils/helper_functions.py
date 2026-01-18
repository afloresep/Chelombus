import logging
import os
import numpy as np
import pandas as pd
from functools import wraps


def _process_input(input_paths):
    """
    Method to deal with input paths for files and folders. 
    Yields files if tuple or list of paths are provided
    input_paths : str, list, or tuple
        A single file path, a folder path, or a list of file and/or folder paths.
        Folders will be traversed to yield file paths contained within.    Yields:
    -------
    str
        File paths from the provided input(s).
    """
    if isinstance(input_paths, str):
        # Single path (file or folder)
        input_paths = [input_paths]
    elif not isinstance(input_paths, (list, tuple)):
        logging.error(f"Invalid input type: {type(input_paths)}. Provide a string, list, or tuple. Skipping")

    for input_path in input_paths:
        if os.path.isdir(input_path):
            # Process all files in the directory
            for file in os.listdir(input_path):
                file_path = os.path.join(input_path, file)
                if os.path.isfile(file_path):  # Optional: filter by file extensions
                    yield file_path
        elif os.path.isfile(input_path):
            # Process a single file
            yield input_path
        else:
            raise ValueError(f"Invalid input path: {input_path}. Ensure the path exists and is a valid file or directory.")


def format_time(seconds):
    """Simpel script to transform seconds into readable format

    Args:
        seconds (int): Seconds 

    Returns:
        str: (int) h, (int) min, (float) seconds 
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} h; {int(minutes)}min; {seconds:.2f} s" 

def save_chunk(fp_chunk: np.ndarray, output_dir: str, chunk_index: int,
                  file_format: str = 'npy', name:str="fingerprints_chunk", **kwargs) -> str:
    """
    Save a chunk of fingerprint data to a file in the specified format.

    Parameters:
        fp_chunk (np.ndarray): The fingerprint array chunk (each row corresponds to a fingerprint).
        output_dir (str): Directory path where the fingerprint chunk will be saved.
        chunk_index (int): The index number for the current chunk. This is used to generate a unique file name.
        file_format (str): Format to save the data. Options are:
                           - 'npy': Save as a NumPy binary file.
                           - 'parquet': Save as an Apache Parquet file.
                           Default is 'npy'.
        name (str): Name prefix for each the file to be generated. 
        **kwargs: Additional keyword arguments to pass to the saving function.
                  For 'npy', kwargs are passed to `np.save`.
                  For 'parquet', kwargs are passed to `DataFrame.to_parquet`.

    Returns:
        str: The full path of the saved file.
    """
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    if file_format.lower() == 'npy':
        filename = os.path.join(output_dir, f"{name}_{chunk_index:05d}.npy")
        np.save(filename, fp_chunk, **kwargs)
        del fp_chunk
    elif file_format.lower() == 'parquet':
        filename = os.path.join(output_dir, f"{name}_{chunk_index:05d}.parquet")
        # Each row is a fingerprint and each column is a bit.
        df = pd.DataFrame(fp_chunk)
        del fp_chunk
        df.to_parquet(filename, **kwargs)
        del df 
    else:
        raise ValueError("Unsupported file format. Please choose 'npy' or 'parquet'.")
    return filename


def get_time(func): 
    import time
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {format_time(end-start)}")
        return result
    return wrapper